import ctypes
import json
import queue
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import soundRadar


class SoundRadarMappingTests(unittest.TestCase):
    def test_arc_positions_follow_clock_layout(self):
        # Position 0 is the front/top wedge. Positions then move clockwise:
        # 3 = right, 6 = back/bottom, 9 = left.
        self.assertEqual(soundRadar.arc_start_deg_for_position(0), 75)
        self.assertEqual(soundRadar.arc_start_deg_for_position(3), -15)
        self.assertEqual(soundRadar.arc_start_deg_for_position(6), -105)
        self.assertEqual(soundRadar.arc_start_deg_for_position(9), 165)

    def test_stereo_output_uses_stereo_mapping_even_when_blackhole_input_has_16_channels(self):
        mapping, mode = soundRadar.build_channel_mapping(16, output_channel_count=2)

        self.assertIn("stereo", mode)
        self.assertEqual(mapping["g"], 0)    # gauche/left
        self.assertEqual(mapping["avg"], 0)  # avant gauche/front-left
        self.assertEqual(mapping["d"], 1)    # droite/right
        self.assertEqual(mapping["avd"], 1)  # avant droite/front-right

    def test_surround_output_keeps_7_1_mapping(self):
        mapping, mode = soundRadar.build_channel_mapping(16, output_channel_count=16)

        self.assertIn("7.1", mode)
        self.assertEqual(mapping["avg"], 0)
        self.assertEqual(mapping["avd"], 1)
        self.assertEqual(mapping["c"], 2)
        self.assertEqual(mapping["g"], 4)
        self.assertEqual(mapping["d"], 5)
        self.assertEqual(mapping["arg"], 6)
        self.assertEqual(mapping["ard"], 7)

    def test_audio_midi_7_1_center_and_rear_channels_are_direct_sectors(self):
        mapping, _ = soundRadar.build_channel_mapping(16, output_channel_count=16)
        cases = [
            (2, 0),  # center -> front/top sector
            (4, 9),  # left surround -> left sector
            (5, 3),  # right surround -> right sector
            (6, 7),  # left rear surround -> rear-left sector
            (7, 5),  # right rear surround -> rear-right sector
        ]

        for channel, expected_sector in cases:
            with self.subTest(channel=channel):
                values = soundRadar.np.zeros(16)
                values[channel] = 0.8
                levels = soundRadar.compute_direction_levels(values, mapping)
                active = [idx for idx, level in enumerate(levels) if level > 0]

                self.assertEqual(active, [expected_sector])

    def test_center_pair_does_not_light_for_single_rear_channel(self):
        self.assertEqual(soundRadar.centered_pair_strength(0.8, 0.0), 0.0)
        self.assertEqual(soundRadar.centered_pair_strength(0.0, 0.8), 0.0)
        self.assertGreater(soundRadar.centered_pair_strength(0.8, 0.75), 0.0)

    def test_direction_levels_still_follow_runtime_directional_ratio(self):
        mapping, _ = soundRadar.build_channel_mapping(8)
        values = soundRadar.np.zeros(8)
        values[mapping[soundRadar.FRONT_LEFT]] = 1.0
        values[mapping[soundRadar.FRONT_RIGHT]] = 0.99
        original_ratio = soundRadar.maxdifmain

        try:
            soundRadar.maxdifmain = 0.02
            quiet = soundRadar.compute_direction_levels(values, mapping)
            soundRadar.maxdifmain = 0.001
            sensitive = soundRadar.compute_direction_levels(values, mapping)
        finally:
            soundRadar.maxdifmain = original_ratio

        self.assertEqual(quiet[11], 0.0)
        self.assertGreater(sensitive[11], 0.0)

    def test_direction_levels_keep_rear_left_and_rear_right_distinct(self):
        mapping, _ = soundRadar.build_channel_mapping(16, output_channel_count=16)
        values = soundRadar.np.zeros(16)

        values[mapping["arg"]] = 0.8
        levels = soundRadar.compute_direction_levels(values, mapping)
        self.assertGreater(levels[7], 0.0)  # rear-left sector
        self.assertEqual(levels[6], 0.0)    # rear-center should stay dark
        self.assertEqual(levels[5], 0.0)    # rear-right should stay dark

        values[:] = 0.0
        values[mapping["ard"]] = 0.8
        levels = soundRadar.compute_direction_levels(values, mapping)
        self.assertGreater(levels[5], 0.0)  # rear-right sector
        self.assertEqual(levels[6], 0.0)
        self.assertEqual(levels[7], 0.0)

    def test_drain_audio_queue_returns_max_values_and_raw_blocks(self):
        audio_queue = queue.Queue()
        first = soundRadar.np.array([[0.1, -0.2], [0.3, 0.1]], dtype=soundRadar.np.float32)
        second = soundRadar.np.array([[-0.4, 0.05]], dtype=soundRadar.np.float32)
        audio_queue.put(first)
        audio_queue.put(second)

        max_values, blocks = soundRadar.drain_audio_queue(2, audio_queue=audio_queue)

        soundRadar.np.testing.assert_allclose(max_values, [0.4, 0.2])
        self.assertEqual(len(blocks), 2)
        soundRadar.np.testing.assert_array_equal(blocks[0], first)
        soundRadar.np.testing.assert_array_equal(blocks[1], second)

    def test_timed_audio_queue_reports_latest_capture_time_for_latency(self):
        audio_queue = queue.Queue()
        first = soundRadar.np.array([[0.1, -0.2]], dtype=soundRadar.np.float32)
        second = soundRadar.np.array([[0.4, -0.6]], dtype=soundRadar.np.float32)
        audio_queue.put(soundRadar.TimedAudioBlock(first, captured_at=10.0))
        audio_queue.put(soundRadar.TimedAudioBlock(second, captured_at=10.032))

        max_values, blocks, latest_capture = soundRadar.drain_audio_queue_with_timing(2, audio_queue=audio_queue)

        soundRadar.np.testing.assert_allclose(max_values, [0.4, 0.6])
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(latest_capture, 10.032)

    def test_direction_event_runtime_scores_latest_audio_window(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        calls = []

        def score_fn(audio, sample_rate, top_k, source_path):
            calls.append((audio.copy(), sample_rate, top_k, source_path))
            return SimpleNamespace(direction_event_scores={}, active_events_by_direction={})

        runtime = soundRadar.DirectionEventRuntime(
            sample_rate=10,
            channel_count=2,
            window_seconds=0.3,
            interval_seconds=0.5,
            top_k=4,
            executor=ImmediateExecutor(),
            score_fn=score_fn,
        )
        runtime.append_blocks([soundRadar.np.ones((4, 2), dtype=soundRadar.np.float32)])

        prediction = runtime.maybe_submit(now=0.1)
        blocked = runtime.maybe_submit(now=0.2)

        self.assertIs(prediction, runtime.latest_prediction)
        self.assertIsNone(blocked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].shape, (3, 2))
        self.assertEqual(calls[0][1:], (10, 4, "<live>"))

    def test_direction_event_runtime_tracks_ast_latency_from_capture_time(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        clock = [10.75]
        runtime = soundRadar.DirectionEventRuntime(
            sample_rate=10,
            channel_count=2,
            window_seconds=0.3,
            interval_seconds=0.5,
            executor=ImmediateExecutor(),
            score_fn=lambda audio, sample_rate, top_k, source_path: SimpleNamespace(
                direction_event_scores={}, active_events_by_direction={}
            ),
            latency_clock=lambda: clock[0],
        )
        runtime.append_blocks([soundRadar.np.ones((4, 2), dtype=soundRadar.np.float32)], capture_time=10.0)

        runtime.maybe_submit(now=0.1)

        self.assertAlmostEqual(runtime.latest_latency_ms, 750.0)

    def test_direction_event_runtime_uses_auto_device_by_default(self):
        # ast_teacher resolves "auto" to MPS on Apple Silicon, and CPU on Macs
        # without a Metal-backed PyTorch device.
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_DEVICE, "auto")

    def test_direction_event_runtime_warms_ast_teacher_in_background(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        class FakeTeacher:
            instances = []

            def __init__(self, model_id, *, device, dtype, **kwargs):
                self.model_id = model_id
                self.device = "mps:0"
                self.dtype = "torch.float16"
                self.warmups = []
                FakeTeacher.instances.append(self)

            def warmup_direction_batch(self, **kwargs):
                self.warmups.append(kwargs)

        with patch("sound_model.ast_teacher.AstAudioSetTeacher", FakeTeacher):
            runtime = soundRadar.DirectionEventRuntime(
                sample_rate=16_000,
                channel_count=8,
                executor=ImmediateExecutor(),
                teacher_model="ast",
            )

        self.assertEqual(len(FakeTeacher.instances), 1)
        self.assertEqual(FakeTeacher.instances[0].warmups[0]["direction_count"], 7)
        self.assertEqual(FakeTeacher.instances[0].warmups[0]["sample_rate"], 16_000)
        self.assertEqual(runtime.resolved_device, "mps:0")
        self.assertEqual(runtime.resolved_dtype, "float16")

    def test_direction_event_runtime_passes_compile_and_attention_options_to_teacher(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        class FakeTeacher:
            init_kwargs = None

            def __init__(self, model_id, **kwargs):
                FakeTeacher.init_kwargs = kwargs
                self.device = "cuda:0"
                self.dtype = "torch.float16"

            def warmup_direction_batch(self, **kwargs):
                pass

        with patch("sound_model.ast_teacher.AstAudioSetTeacher", FakeTeacher):
            soundRadar.DirectionEventRuntime(
                sample_rate=16_000,
                channel_count=8,
                executor=ImmediateExecutor(),
                teacher_model="ast",
                compile_model="reduce-overhead",
                attn_implementation="sdpa",
            )

        self.assertEqual(FakeTeacher.init_kwargs["compile_model"], "reduce-overhead")
        self.assertEqual(FakeTeacher.init_kwargs["attn_implementation"], "sdpa")

    def test_direction_event_runtime_passes_selected_teacher_model_to_factory(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        factory_calls = []

        class FakeTeacher:
            device = "cpu"
            dtype = "torch.float32"

            def warmup_direction_batch(self, **kwargs):
                pass

        def fake_factory(teacher_model, **kwargs):
            factory_calls.append((teacher_model, kwargs))
            return FakeTeacher()

        with patch("sound_model.ast_teacher.create_audio_event_teacher", fake_factory):
            soundRadar.DirectionEventRuntime(
                sample_rate=16_000,
                channel_count=8,
                executor=ImmediateExecutor(),
                teacher_model="efficientat-mn10",
                model_id="mn10_as",
            )

        self.assertEqual(factory_calls[0][0], "efficientat-mn10")
        self.assertEqual(factory_calls[0][1]["model_id"], "mn10_as")


class SoundRadarPulseTests(unittest.TestCase):
    def test_pulse_opacity_fades_to_zero(self):
        pulse = soundRadar.SoundPulse(sector=3, strength=0.8, created_at=10.0, duration=1.0)

        self.assertGreater(soundRadar.pulse_opacity(pulse, now=10.0), 0.7)
        self.assertGreater(soundRadar.pulse_opacity(pulse, now=10.5), 0.0)
        self.assertEqual(soundRadar.pulse_opacity(pulse, now=11.1), 0.0)

    def test_create_pulses_from_levels_ignores_low_values(self):
        levels = soundRadar.np.zeros(soundRadar.RADAR_SECTORS)
        levels[3] = 0.004

        pulses = soundRadar.create_pulses_from_levels(levels, now=10.0, threshold=0.02)

        self.assertEqual(pulses, [])

    def test_create_pulses_from_levels_uses_active_sector(self):
        levels = soundRadar.np.zeros(soundRadar.RADAR_SECTORS)
        levels[5] = 0.7

        pulses = soundRadar.create_pulses_from_levels(levels, now=10.0, threshold=0.02)

        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0].sector, 5)
        self.assertAlmostEqual(pulses[0].strength, 0.7)
        self.assertEqual(pulses[0].kind, "sharp")

    def test_create_pulses_respects_per_sector_cooldown(self):
        levels = soundRadar.np.zeros(soundRadar.RADAR_SECTORS)
        levels[5] = 0.7
        last_pulse_times = soundRadar.np.zeros(soundRadar.RADAR_SECTORS)
        last_pulse_times[5] = 10.0

        blocked = soundRadar.create_pulses_from_levels(
            levels,
            now=10.05,
            threshold=0.02,
            cooldown=0.18,
            last_pulse_times=last_pulse_times,
        )
        allowed = soundRadar.create_pulses_from_levels(
            levels,
            now=10.2,
            threshold=0.02,
            cooldown=0.18,
            last_pulse_times=last_pulse_times,
        )

        self.assertEqual(blocked, [])
        self.assertEqual(len(allowed), 1)

    def test_sector_mid_angles_are_clock_like(self):
        self.assertEqual(soundRadar.sector_mid_angle_deg(0), 90)
        self.assertEqual(soundRadar.sector_mid_angle_deg(3), 0)
        self.assertEqual(soundRadar.sector_mid_angle_deg(6), -90)
        self.assertEqual(soundRadar.sector_mid_angle_deg(9), 180)

    def test_ripple_radius_stays_outer_and_moves_outward(self):
        pulse = soundRadar.SoundPulse(sector=3, strength=0.8, created_at=10.0, duration=1.0)

        start_radius = soundRadar.pulse_ripple_radius(pulse, now=10.0, min_side=720)
        later_radius = soundRadar.pulse_ripple_radius(pulse, now=10.5, min_side=720)

        self.assertGreaterEqual(start_radius, 720 * 0.38)
        self.assertGreater(later_radius, start_radius)

    def test_basic_sound_kind_is_ready_for_future_classification(self):
        self.assertEqual(soundRadar.classify_basic_sound_event(0.9, 0.1), "impact")
        self.assertEqual(soundRadar.classify_basic_sound_event(0.5, 0.1), "sharp")
        self.assertEqual(soundRadar.classify_basic_sound_event(0.2, 0.19), "unknown")

    def test_dominant_active_event_uses_strongest_score_not_priority_order(self):
        event_name, score = soundRadar.dominant_active_event(
            {"explosion": 0.42, "gunshot": 0.99, "vehicle": 0.51, "footstep": 0.48},
            active_events=["explosion", "gunshot", "vehicle", "footstep"],
            threshold=0.1,
        )

        self.assertEqual(event_name, "gunshot")
        self.assertAlmostEqual(score, 0.99)

    def test_dominant_active_event_biases_ambiguous_game_guns_to_gunshot(self):
        event_name, score = soundRadar.dominant_active_event(
            {"explosion": 0.89, "gunshot": 0.72, "vehicle": 0.58, "footstep": 0.76},
            active_events=["explosion", "gunshot", "vehicle", "footstep"],
            threshold=0.1,
        )

        self.assertEqual(event_name, "gunshot")
        self.assertAlmostEqual(score, 0.72)

    def test_dominant_active_event_requires_high_confidence_for_step_icons(self):
        weak_footstep, weak_score = soundRadar.dominant_active_event(
            {"footstep": 0.74, "gunshot": 0.0},
            active_events=["footstep"],
            threshold=0.1,
        )
        strong_footstep, strong_score = soundRadar.dominant_active_event(
            {"footstep": 0.93, "gunshot": 0.0},
            active_events=["footstep"],
            threshold=0.1,
        )

        self.assertIsNone(weak_footstep)
        self.assertEqual(weak_score, 0.0)
        self.assertEqual(strong_footstep, "footstep")
        self.assertAlmostEqual(strong_score, 0.93)

    def test_dominant_active_event_allows_sustained_vehicle_icon(self):
        weak_vehicle, weak_score = soundRadar.dominant_active_event(
            {"vehicle": 0.59, "gunshot": 0.0},
            active_events=["vehicle"],
            threshold=0.1,
        )
        vehicle, score = soundRadar.dominant_active_event(
            {"vehicle": 0.70, "gunshot": 0.0},
            active_events=["vehicle"],
            threshold=0.1,
        )

        self.assertIsNone(weak_vehicle)
        self.assertEqual(weak_score, 0.0)
        self.assertEqual(vehicle, "vehicle")
        self.assertAlmostEqual(score, 0.70)

    def test_dominant_active_event_does_not_bias_close_vehicle_to_gunshot(self):
        vehicle, score = soundRadar.dominant_active_event(
            {"vehicle": 0.70, "gunshot": 0.55, "explosion": 0.0, "footstep": 0.0},
            active_events=["vehicle", "gunshot"],
            threshold=0.1,
        )

        self.assertEqual(vehicle, "vehicle")
        self.assertAlmostEqual(score, 0.70)

    def test_displayed_active_events_suppresses_gunshot_secondary_for_vehicle(self):
        events = soundRadar.displayed_active_events(
            {"vehicle": 0.70, "gunshot": 0.55, "explosion": 0.0, "footstep": 0.0},
            active_events=["vehicle", "gunshot"],
            threshold=0.1,
        )

        self.assertEqual(events, [("vehicle", 0.70)])

    def test_displayed_active_events_can_show_vehicle_secondary_for_gunshot(self):
        events = soundRadar.displayed_active_events(
            {"vehicle": 0.70, "gunshot": 0.75, "explosion": 0.0, "footstep": 0.0},
            active_events=["vehicle", "gunshot"],
            threshold=0.1,
        )

        self.assertEqual(events, [("gunshot", 0.75), ("vehicle", 0.70)])

    def test_displayed_active_events_does_not_show_tiny_secondary_gunshot(self):
        events = soundRadar.displayed_active_events(
            {"vehicle": 0.70, "gunshot": 0.10, "explosion": 0.0, "footstep": 0.0},
            active_events=["vehicle", "gunshot"],
            threshold=0.1,
        )

        self.assertEqual(events, [("vehicle", 0.70)])

    def test_direction_event_prediction_creates_event_kind_pulse(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {
                    "background": 0.2,
                    "footstep": 0.0,
                    "gunshot": 0.8,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                }
            },
            active_events_by_direction={"right": ["gunshot"]},
        )

        pulses = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)

        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0].sector, 3)
        self.assertEqual(pulses[0].kind, "gunshot")
        self.assertAlmostEqual(pulses[0].strength, 0.8)
        self.assertEqual(pulses[0].duration, soundRadar.EVENT_ICON_DURATION)

    def test_direction_event_prediction_suppresses_adjacent_gunshot_bleed(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {
                    "background": 0.2,
                    "footstep": 0.0,
                    "gunshot": 0.31,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
                "front_right": {
                    "background": 0.1,
                    "footstep": 0.0,
                    "gunshot": 0.72,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
                "right": {
                    "background": 0.3,
                    "footstep": 0.0,
                    "gunshot": 0.26,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "front_right": ["gunshot"],
                "right": ["gunshot"],
            },
        )

        pulses = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)

        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0].sector, soundRadar.DIRECTION_EVENT_SECTORS["front_right"])
        self.assertEqual(pulses[0].kind, "gunshot")
        self.assertAlmostEqual(pulses[0].strength, 0.72)

    def test_direction_event_pulses_follow_runtime_spatial_limit(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {"gunshot": 0.8},
                "rear_right": {"gunshot": 0.7},
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "rear_right": ["gunshot"],
            },
        )
        original_limit = soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS

        try:
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = 1
            limited = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = 2
            expanded = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)
        finally:
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = original_limit

        self.assertEqual([pulse.sector for pulse in limited], [soundRadar.DIRECTION_EVENT_SECTORS["front"]])
        self.assertEqual(
            {pulse.sector for pulse in expanded},
            {soundRadar.DIRECTION_EVENT_SECTORS["front"], soundRadar.DIRECTION_EVENT_SECTORS["rear_right"]},
        )

    def test_direction_event_prediction_keeps_distant_low_score_gunshot_local_maximum(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {
                    "background": 0.9,
                    "footstep": 0.0,
                    "gunshot": 0.10,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
                "front_right": {
                    "background": 0.85,
                    "footstep": 0.0,
                    "gunshot": 0.13,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
                "right": {
                    "background": 0.9,
                    "footstep": 0.0,
                    "gunshot": 0.11,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                },
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "front_right": ["gunshot"],
                "right": ["gunshot"],
            },
        )

        pulses = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)

        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0].sector, soundRadar.DIRECTION_EVENT_SECTORS["front_right"])
        self.assertAlmostEqual(pulses[0].strength, 0.13)

    def test_direction_event_prediction_applies_global_gunshot_cooldown(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {
                    "background": 0.2,
                    "footstep": 0.0,
                    "gunshot": 0.8,
                    "vehicle": 0.0,
                    "explosion": 0.0,
                }
            },
            active_events_by_direction={"right": ["gunshot"]},
        )
        last_global_event_times = {}

        first = soundRadar.create_pulses_from_direction_events(
            prediction,
            now=10.0,
            threshold=0.1,
            last_global_event_times=last_global_event_times,
        )
        blocked = soundRadar.create_pulses_from_direction_events(
            prediction,
            now=10.05,
            threshold=0.1,
            last_global_event_times=last_global_event_times,
        )
        allowed = soundRadar.create_pulses_from_direction_events(
            prediction,
            now=10.25,
            threshold=0.1,
            last_global_event_times=last_global_event_times,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(blocked, [])
        self.assertEqual(len(allowed), 1)

    def test_direction_event_prediction_reports_gunshot_display_debug(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {"gunshot": 0.31, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "front_right": {"gunshot": 0.72, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "right": {"gunshot": 0.26, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "front_right": ["gunshot"],
                "right": ["gunshot"],
            },
        )
        display_debug = {}

        soundRadar.create_pulses_from_direction_events(
            prediction,
            now=10.0,
            threshold=0.1,
            display_debug=display_debug,
        )

        debug = display_debug["debug"]
        self.assertEqual(debug.gunshot_emitted_directions, ("front_right",))
        self.assertEqual(debug.gunshot_decision.allowed_directions, frozenset(("front_right",)))
        self.assertEqual(debug.gunshot_decision.spatially_suppressed_directions, frozenset(("front", "right")))

    def test_direction_event_prediction_reports_global_gunshot_cooldown_debug(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"]},
        )
        display_debug = {}

        pulses = soundRadar.create_pulses_from_direction_events(
            prediction,
            now=10.05,
            threshold=0.1,
            last_global_event_times={"gunshot": 10.0},
            display_debug=display_debug,
        )

        self.assertEqual(pulses, [])
        self.assertEqual(display_debug["debug"].gunshot_global_cooldown_blocked_directions, ("right",))

    def test_direction_event_prediction_creates_multiple_event_icons_per_sector(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {
                    "background": 0.1,
                    "footstep": 0.0,
                    "gunshot": 0.75,
                    "vehicle": 0.70,
                    "explosion": 0.0,
                }
            },
            active_events_by_direction={"right": ["vehicle", "gunshot"]},
        )

        pulses = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)

        self.assertEqual(
            [(pulse.kind, pulse.lane_index, pulse.lane_count) for pulse in pulses],
            [("gunshot", 0, 2), ("vehicle", 1, 2)],
        )

    def test_direction_event_prediction_suppresses_secondary_gunshot_for_vehicle(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {
                    "background": 0.1,
                    "footstep": 0.0,
                    "gunshot": 0.55,
                    "vehicle": 0.70,
                    "explosion": 0.0,
                }
            },
            active_events_by_direction={"right": ["vehicle", "gunshot"]},
        )

        pulses = soundRadar.create_pulses_from_direction_events(prediction, now=10.0, threshold=0.1)

        self.assertEqual([(pulse.kind, pulse.lane_index, pulse.lane_count) for pulse in pulses], [("vehicle", 0, 1)])

    def test_event_icon_center_uses_sector_mid_angle_and_outer_ring(self):
        center = soundRadar.event_icon_center(sector=3, min_side=720, center_x=360, center_y=360)

        self.assertAlmostEqual(center.x(), 360 + 720 * soundRadar.EVENT_ICON_DISTANCE_RATIO)
        self.assertAlmostEqual(center.y(), 360)

        front_center = soundRadar.event_icon_center(sector=0, min_side=720, center_x=360, center_y=360)

        self.assertAlmostEqual(front_center.x(), 360)
        self.assertAlmostEqual(front_center.y(), 360 - 720 * soundRadar.EVENT_ICON_DISTANCE_RATIO)

    def test_event_icon_center_offsets_multiple_lanes_tangentially(self):
        first = soundRadar.event_icon_center(
            sector=0,
            min_side=720,
            center_x=360,
            center_y=360,
            lane_index=0,
            lane_count=2,
        )
        second = soundRadar.event_icon_center(
            sector=0,
            min_side=720,
            center_x=360,
            center_y=360,
            lane_index=1,
            lane_count=2,
        )

        self.assertLess(min(first.x(), second.x()), 360)
        self.assertGreater(max(first.x(), second.x()), 360)
        self.assertAlmostEqual(first.y(), second.y())

    def test_event_icon_size_ratios_are_compact_for_dual_icons(self):
        self.assertLessEqual(soundRadar.EVENT_ICON_MAX_SIZE_RATIO, 0.044)
        self.assertLessEqual(soundRadar.EVENT_ICON_MIN_SIZE_RATIO, 0.026)
        self.assertFalse(soundRadar.EVENT_ICON_SHOW_LABELS)

    def test_event_icon_size_uses_strength_without_moving_direction(self):
        weak = soundRadar.SoundPulse(sector=1, strength=0.2, created_at=10.0, kind="gunshot")
        strong = soundRadar.SoundPulse(sector=1, strength=0.9, created_at=10.0, kind="gunshot")

        self.assertGreater(soundRadar.event_icon_size(strong, now=10.1, min_side=720), soundRadar.event_icon_size(weak, now=10.1, min_side=720))
        self.assertLessEqual(soundRadar.event_icon_size(strong, now=10.1, min_side=720), 720 * 0.048)
        self.assertEqual(
            soundRadar.event_icon_center(weak.sector, 720, 360, 360),
            soundRadar.event_icon_center(strong.sector, 720, 360, 360),
        )

    def test_event_icon_size_follows_runtime_scale(self):
        pulse = soundRadar.SoundPulse(sector=1, strength=0.8, created_at=10.0, kind="gunshot")
        original_scale = soundRadar.EVENT_ICON_SIZE_SCALE

        try:
            soundRadar.EVENT_ICON_SIZE_SCALE = 1.0
            normal = soundRadar.event_icon_size(pulse, now=10.1, min_side=720)
            soundRadar.EVENT_ICON_SIZE_SCALE = 1.25
            enlarged = soundRadar.event_icon_size(pulse, now=10.1, min_side=720)
        finally:
            soundRadar.EVENT_ICON_SIZE_SCALE = original_scale

        self.assertAlmostEqual(enlarged, normal * 1.25)

    def test_event_icon_opacity_holds_before_fading(self):
        pulse = soundRadar.SoundPulse(sector=1, strength=0.8, created_at=10.0, duration=soundRadar.EVENT_ICON_DURATION, kind="gunshot")

        self.assertGreater(soundRadar.event_icon_opacity(pulse, now=12.0), 0.85)
        self.assertGreater(soundRadar.event_icon_opacity(pulse, now=13.0), 0.0)
        self.assertEqual(soundRadar.event_icon_opacity(pulse, now=14.0), 0.0)

    def test_event_kind_does_not_recolor_watercolor_ripple(self):
        gunshot = soundRadar.watercolor_color(0.8, 1.0, "gunshot")
        vehicle = soundRadar.watercolor_color(0.8, 1.0, "vehicle")

        self.assertEqual(gunshot.getRgb(), vehicle.getRgb())

    def test_direction_event_debug_lines_show_compass_hud_and_device(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "left": {"gunshot": 0.0, "footstep": 0.92, "vehicle": 0.0, "explosion": 0.0},
                "rear_right": {"gunshot": 0.0, "footstep": 0.0, "vehicle": 0.88, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"], "left": ["footstep"], "rear_right": ["vehicle"]},
        )

        lines = soundRadar.direction_event_debug_lines(
            prediction,
            threshold=0.1,
            status="idle",
            teacher_model="efficientat-mn10",
            requested_device="auto",
            resolved_device="mps",
            radar_latency_ms=18.4,
            ast_latency_ms=255.0,
        )

        self.assertEqual(lines[0], "efficientat-mn10 idle auto→mps")
        self.assertEqual(lines[1], "lag radar 18ms model 255ms Δ+237ms")
        self.assertIn("gun cand R .80", lines[2])
        self.assertIn("show R", lines[2])
        self.assertIn("F: --", lines[3])
        self.assertIn("L: FOOT .92", lines[5])
        self.assertIn("R: GUN .80", lines[5])
        self.assertIn("RR: VEH .88", lines[6])

    def test_direction_event_debug_lines_show_gunshot_suppression_and_cooldown(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {"gunshot": 0.31, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "front_right": {"gunshot": 0.72, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "right": {"gunshot": 0.26, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "front_right": ["gunshot"],
                "right": ["gunshot"],
            },
        )
        display_debug = {
            "debug": soundRadar.DirectionEventPulseDebug(
                gunshot_decision=soundRadar.gunshot_display_decision(
                    prediction.direction_event_scores,
                    prediction.active_events_by_direction,
                    threshold=0.1,
                ),
                gunshot_emitted_directions=(),
                gunshot_global_cooldown_blocked_directions=("front_right",),
                gunshot_sector_cooldown_blocked_directions=(),
            )
        }

        lines = soundRadar.direction_event_debug_lines(
            prediction,
            threshold=0.1,
            display_debug=display_debug["debug"],
        )

        self.assertIn("cand FR .72,F .31,R .26", lines[2])
        self.assertIn("show --", lines[2])
        self.assertIn("sup F/R", lines[2])
        self.assertIn("cd G:FR", lines[2])

    def test_direction_event_debug_line_follows_runtime_spatial_limit(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {"gunshot": 0.8},
                "rear_right": {"gunshot": 0.7},
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "rear_right": ["gunshot"],
            },
        )
        original_limit = soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS

        try:
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = 1
            limited = soundRadar.direction_event_gunshot_debug_line(prediction, threshold=0.1)
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = 2
            expanded = soundRadar.direction_event_gunshot_debug_line(prediction, threshold=0.1)
        finally:
            soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS = original_limit

        self.assertIn("show F ", limited)
        self.assertIn("show F/RR ", expanded)

    def test_direction_event_runtime_device_label_shows_mps_resolution(self):
        runtime = soundRadar.DirectionEventRuntime(sample_rate=16_000, channel_count=8, device="auto", warmup=False)
        runtime.resolved_device = "mps"

        self.assertEqual(soundRadar.direction_event_device_label(runtime), "auto→mps")

    def test_default_ripple_style_is_watercolor(self):
        self.assertEqual(soundRadar.RIPPLE_STYLE, "watercolor")

    def test_watercolor_blobs_stay_near_outer_sector(self):
        pulse = soundRadar.SoundPulse(sector=3, strength=0.8, created_at=10.0, duration=1.0)

        specs = soundRadar.watercolor_blob_specs(pulse, now=10.0, min_side=720)

        self.assertEqual(len(specs), soundRadar.WATERCOLOR_BLOBS)
        for spec in specs:
            self.assertGreaterEqual(spec.distance, 720 * soundRadar.WATERCOLOR_INNER_SAFE_RATIO)
            angle_offset = abs(soundRadar.normalize_degrees(spec.angle_deg - soundRadar.sector_mid_angle_deg(3)))
            self.assertLessEqual(angle_offset, soundRadar.WATERCOLOR_ANGLE_SPREAD)
            self.assertGreater(spec.radius, 0.0)
            self.assertGreaterEqual(spec.opacity, 0.0)
            self.assertLessEqual(spec.opacity, 1.0)

    def test_watercolor_blobs_are_stable_and_spread_outward(self):
        pulse = soundRadar.SoundPulse(sector=5, strength=0.8, created_at=10.0, duration=1.0)

        start_specs = soundRadar.watercolor_blob_specs(pulse, now=10.0, min_side=720)
        repeated_start_specs = soundRadar.watercolor_blob_specs(pulse, now=10.0, min_side=720)
        later_specs = soundRadar.watercolor_blob_specs(pulse, now=10.5, min_side=720)

        self.assertEqual(start_specs, repeated_start_specs)
        start_distance = sum(spec.distance for spec in start_specs) / len(start_specs)
        later_distance = sum(spec.distance for spec in later_specs) / len(later_specs)
        start_opacity = sum(spec.opacity for spec in start_specs) / len(start_specs)
        later_opacity = sum(spec.opacity for spec in later_specs) / len(later_specs)
        self.assertGreater(later_distance, start_distance)
        self.assertLess(later_opacity, start_opacity)


class SoundRadarThresholdProfileTests(unittest.TestCase):
    def tearDown(self):
        soundRadar.apply_threshold_profile("default")

    def test_threshold_profile_names_include_runtime_choices(self):
        self.assertEqual(soundRadar.threshold_profile_names(), ("default", "quiet", "aggressive", "debug"))

    def test_parse_threshold_profile_args_uses_env_and_removes_qt_arg(self):
        qt_args, profile_name = soundRadar.parse_threshold_profile_args(
            ["soundRadar.py", "--threshold-profile", "quiet", "-platform", "offscreen"],
            environ={soundRadar.THRESHOLD_PROFILE_ENV: "aggressive"},
        )

        self.assertEqual(qt_args, ["soundRadar.py", "-platform", "offscreen"])
        self.assertEqual(profile_name, "quiet")

    def test_parse_threshold_profile_args_accepts_equals_form(self):
        qt_args, profile_name = soundRadar.parse_threshold_profile_args(
            ["soundRadar.py", "--threshold-profile=debug"],
            environ={},
        )

        self.assertEqual(qt_args, ["soundRadar.py"])
        self.assertEqual(profile_name, "debug")

    def test_apply_threshold_profile_updates_thresholds_without_code_edits(self):
        profile = soundRadar.apply_threshold_profile("quiet")

        self.assertEqual(profile.name, "quiet")
        self.assertEqual(soundRadar.RIPPLE_THRESHOLD, profile.ripple_threshold)
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_THRESHOLD, profile.direction_event_threshold)
        self.assertEqual(soundRadar.DIRECTION_EVENT_DISPLAY_THRESHOLDS["gunshot"], 0.16)
        self.assertEqual(soundRadar.GUNSHOT_SPATIAL_MAX_DIRECTIONS, 1)


class SoundRadarRuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self._settings = {
            "ENABLE_AST_DIRECTION_EVENTS": soundRadar.ENABLE_AST_DIRECTION_EVENTS,
            "AST_DIRECTION_EVENT_TEACHER_MODEL": soundRadar.AST_DIRECTION_EVENT_TEACHER_MODEL,
            "AST_DIRECTION_EVENT_MODEL_ID": soundRadar.AST_DIRECTION_EVENT_MODEL_ID,
            "AST_DIRECTION_EVENT_DEVICE": soundRadar.AST_DIRECTION_EVENT_DEVICE,
            "AST_DIRECTION_EVENT_DTYPE": soundRadar.AST_DIRECTION_EVENT_DTYPE,
            "AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION": soundRadar.AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION,
            "AST_DIRECTION_EVENT_COMPILE": soundRadar.AST_DIRECTION_EVENT_COMPILE,
            "AST_DIRECTION_EVENT_TOP_K": soundRadar.AST_DIRECTION_EVENT_TOP_K,
            "AST_DIRECTION_EVENT_WINDOW_SECONDS": soundRadar.AST_DIRECTION_EVENT_WINDOW_SECONDS,
            "AST_DIRECTION_EVENT_INTERVAL": soundRadar.AST_DIRECTION_EVENT_INTERVAL,
            "AST_DIRECTION_EVENT_WARMUP": soundRadar.AST_DIRECTION_EVENT_WARMUP,
            "ROLLING_CAPTURE_ENABLED": soundRadar.ROLLING_CAPTURE_ENABLED,
            "ROLLING_CAPTURE_SECONDS": soundRadar.ROLLING_CAPTURE_SECONDS,
            "ROLLING_CAPTURE_DIR": soundRadar.ROLLING_CAPTURE_DIR,
            "ROLLING_CAPTURE_TRIGGER_PATH": soundRadar.ROLLING_CAPTURE_TRIGGER_PATH,
            "EVENT_ICON_SHOW_LABELS": soundRadar.EVENT_ICON_SHOW_LABELS,
            "EVENT_ICON_SIZE_SCALE": soundRadar.EVENT_ICON_SIZE_SCALE,
            "EVENT_ICON_ALPHA_SCALE": soundRadar.EVENT_ICON_ALPHA_SCALE,
            "DIRECTION_EVENT_SMOOTHING_ENABLED": soundRadar.DIRECTION_EVENT_SMOOTHING_ENABLED,
            "DIRECTION_EVENT_SMOOTHING_WINDOW": soundRadar.DIRECTION_EVENT_SMOOTHING_WINDOW,
        }

    def tearDown(self):
        for name, value in self._settings.items():
            setattr(soundRadar, name, value)
        soundRadar.apply_threshold_profile("default")

    def test_load_runtime_config_reads_json_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/soundradar.local.json"
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump({"teacher_model": "ast", "threshold_profile": "quiet"}, config_file)

            config = soundRadar.load_runtime_config(path)

        self.assertEqual(config["teacher_model"], "ast")
        self.assertEqual(config["threshold_profile"], "quiet")

    def test_apply_runtime_config_updates_runtime_settings_without_code_edits(self):
        soundRadar.apply_runtime_config(
            {
                "enable_direction_events": False,
                "teacher_model": "ast",
                "model_id": "custom/model",
                "device": "mps",
                "dtype": "float16",
                "attn_implementation": "sdpa",
                "compile_model": "reduce-overhead",
                "top_k": 7,
                "window_seconds": 1.5,
                "interval_seconds": 0.75,
                "warmup": "off",
                "rolling_capture_enabled": True,
                "rolling_capture_seconds": 3.5,
                "rolling_capture_dir": "/tmp/soundradar-roll",
                "rolling_capture_trigger_path": "/tmp/soundradar-trigger",
                "event_icon_labels": True,
                "event_icon_scale": 0.8,
                "event_icon_opacity": 0.6,
                "event_smoothing_enabled": False,
                "event_smoothing_window": 2,
            }
        )

        self.assertFalse(soundRadar.ENABLE_AST_DIRECTION_EVENTS)
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_TEACHER_MODEL, "ast")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_MODEL_ID, "custom/model")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_DEVICE, "mps")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_DTYPE, "float16")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION, "sdpa")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_COMPILE, "reduce-overhead")
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_TOP_K, 7)
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_WINDOW_SECONDS, 1.5)
        self.assertEqual(soundRadar.AST_DIRECTION_EVENT_INTERVAL, 0.75)
        self.assertFalse(soundRadar.AST_DIRECTION_EVENT_WARMUP)
        self.assertTrue(soundRadar.ROLLING_CAPTURE_ENABLED)
        self.assertEqual(soundRadar.ROLLING_CAPTURE_SECONDS, 3.5)
        self.assertEqual(soundRadar.ROLLING_CAPTURE_DIR, "/tmp/soundradar-roll")
        self.assertEqual(soundRadar.ROLLING_CAPTURE_TRIGGER_PATH, "/tmp/soundradar-trigger")
        self.assertTrue(soundRadar.EVENT_ICON_SHOW_LABELS)
        self.assertEqual(soundRadar.EVENT_ICON_SIZE_SCALE, 0.8)
        self.assertEqual(soundRadar.EVENT_ICON_ALPHA_SCALE, 0.6)
        self.assertFalse(soundRadar.DIRECTION_EVENT_SMOOTHING_ENABLED)
        self.assertEqual(soundRadar.DIRECTION_EVENT_SMOOTHING_WINDOW, 2)

    def test_parse_threshold_profile_args_uses_config_then_env_then_cli(self):
        qt_args, env_profile = soundRadar.parse_threshold_profile_args(
            ["soundRadar.py", "-platform", "offscreen"],
            environ={soundRadar.THRESHOLD_PROFILE_ENV: "aggressive"},
            runtime_config={"threshold_profile": "quiet"},
        )
        cli_args, cli_profile = soundRadar.parse_threshold_profile_args(
            ["soundRadar.py", "--threshold-profile=debug"],
            environ={soundRadar.THRESHOLD_PROFILE_ENV: "aggressive"},
            runtime_config={"threshold_profile": "quiet"},
        )

        self.assertEqual(qt_args, ["soundRadar.py", "-platform", "offscreen"])
        self.assertEqual(env_profile, "aggressive")
        self.assertEqual(cli_args, ["soundRadar.py"])
        self.assertEqual(cli_profile, "debug")


class SoundRadarRollingCaptureTests(unittest.TestCase):
    def test_rolling_audio_capture_keeps_recent_window(self):
        capture = soundRadar.RollingAudioCapture(sample_rate=10, channel_count=2, seconds=0.3)
        capture.append_blocks([soundRadar.np.ones((4, 2), dtype=soundRadar.np.float32)], capture_time=10.0)
        capture.append_blocks([soundRadar.np.full((2, 2), 2.0, dtype=soundRadar.np.float32)], capture_time=10.2)

        snapshot = capture.snapshot()

        self.assertEqual(snapshot.audio.shape, (3, 2))
        soundRadar.np.testing.assert_array_equal(snapshot.audio[-1], [2.0, 2.0])
        self.assertAlmostEqual(snapshot.end_capture_time, 10.2)
        self.assertAlmostEqual(snapshot.start_capture_time, 9.9)

    def test_write_rolling_capture_snapshot_writes_wav_and_metadata(self):
        capture = soundRadar.RollingAudioCapture(sample_rate=10, channel_count=8, seconds=1.0)
        audio = soundRadar.np.zeros((5, 8), dtype=soundRadar.np.float32)
        audio[:, 0] = 0.5
        audio[:, 1] = 0.5
        capture.append_blocks([audio], capture_time=10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, metadata_path = soundRadar.write_rolling_capture_snapshot(
                capture.snapshot(),
                directory=tmpdir,
                now=10.25,
                threshold_profile_name="default",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertTrue(str(audio_path).endswith(".wav"))
        self.assertEqual(metadata["sample_rate"], 10)
        self.assertEqual(metadata["channel_count"], 8)
        self.assertEqual(metadata["threshold_profile"], "default")
        self.assertIn("ch0=0.500", metadata["peak_summary"])
        self.assertTrue(any("stereo/downmixed" in line for line in metadata["sanity_lines"]))

    def test_write_and_consume_rolling_capture_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/trigger"

            written = soundRadar.write_rolling_capture_trigger(path)
            consumed = soundRadar.consume_rolling_capture_trigger(path)

        self.assertEqual(str(written), path)
        self.assertTrue(consumed)


class SoundRadarEventSmoothingTests(unittest.TestCase):
    def tearDown(self):
        soundRadar.apply_runtime_config(
            {
                "event_smoothing_enabled": True,
                "event_smoothing_window": 3,
                "event_icon_labels": False,
                "event_icon_scale": 1.0,
                "event_icon_opacity": 1.0,
            }
        )

    def test_smooth_direction_event_predictions_weights_recent_scores(self):
        first = SimpleNamespace(
            sample_rate=48000,
            direction_event_scores={"right": {"gunshot": 0.90, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0}},
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={"right": [{"label": "Gunshot", "score": 0.90}]},
            source_path="<live>",
        )
        second = SimpleNamespace(
            sample_rate=48000,
            direction_event_scores={"right": {"gunshot": 0.0, "vehicle": 0.30, "footstep": 0.0, "explosion": 0.0}},
            active_events_by_direction={"right": ["vehicle"]},
            top_labels_by_direction={"right": [{"label": "Vehicle", "score": 0.30}]},
            source_path="<live>",
        )

        smoothed = soundRadar.smooth_direction_event_predictions([first, second], window=2)

        self.assertAlmostEqual(smoothed.direction_event_scores["right"]["gunshot"], 0.30)
        self.assertAlmostEqual(smoothed.direction_event_scores["right"]["vehicle"], 0.20)
        self.assertEqual(smoothed.active_events_by_direction["right"], ["gunshot", "vehicle"])
        self.assertEqual(smoothed.top_labels_by_direction["right"][0]["label"], "Vehicle")

    def test_update_direction_event_runtime_returns_smoothed_prediction(self):
        class ImmediateExecutor:
            def submit(self, fn, *args):
                class DoneFuture:
                    def done(self):
                        return True

                    def result(self):
                        return fn(*args)

                return DoneFuture()

        predictions = [
            SimpleNamespace(
                sample_rate=10,
                direction_event_scores={"right": {"gunshot": 0.90, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0}},
                active_events_by_direction={"right": ["gunshot"]},
                top_labels_by_direction={},
            ),
            SimpleNamespace(
                sample_rate=10,
                direction_event_scores={"right": {"gunshot": 0.0, "vehicle": 0.30, "footstep": 0.0, "explosion": 0.0}},
                active_events_by_direction={"right": ["vehicle"]},
                top_labels_by_direction={},
            ),
        ]

        def score_fn(audio, sample_rate, top_k, source_path):
            _ = audio, sample_rate, top_k, source_path
            return predictions.pop(0)

        soundRadar.apply_runtime_config({"event_smoothing_enabled": True, "event_smoothing_window": 2})
        radar = SimpleNamespace(
            direction_event_runtime=soundRadar.DirectionEventRuntime(
                sample_rate=10,
                channel_count=2,
                window_seconds=0.1,
                interval_seconds=0.0,
                executor=ImmediateExecutor(),
                score_fn=score_fn,
            ),
            _direction_event_prediction_history=[],
            ast_latency_ms=None,
            latest_direction_event_prediction=None,
        )
        blocks = [soundRadar.np.ones((1, 2), dtype=soundRadar.np.float32)]

        first = soundRadar.update_direction_event_runtime(radar, blocks, now=0.0, capture_time=0.0)
        second = soundRadar.update_direction_event_runtime(radar, blocks, now=0.1, capture_time=0.1)

        self.assertAlmostEqual(first.direction_event_scores["right"]["gunshot"], 0.90)
        self.assertAlmostEqual(second.direction_event_scores["right"]["gunshot"], 0.30)
        self.assertIs(radar.latest_direction_event_prediction, second)


class SoundRadarWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = soundRadar.QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = soundRadar.QtWidgets.QApplication([])

    def test_overlay_does_not_raise_periodically(self):
        window = soundRadar.ParentWidget()

        self.assertFalse(hasattr(window, "_top_timer") and window._top_timer.isActive())

    def test_overlay_has_native_keep_on_top_pulse_without_qt_raise(self):
        class ProbeWindow(soundRadar.ParentWidget):
            def __init__(self):
                self.raise_called = False
                super().__init__()

            def raise_(self):
                self.raise_called = True
                raise AssertionError("raise_() should not be used")

        window = ProbeWindow()
        self.assertTrue(hasattr(window, "_native_top_timer"))
        self.assertTrue(window._native_top_timer.isActive())
        window.ensure_on_top()
        self.assertFalse(window.raise_called)

    def test_ripple_overlay_has_watercolor_renderer(self):
        window = soundRadar.ParentWidget()

        self.assertTrue(hasattr(window.ripple_overlay, "_paint_watercolor_pulse"))

    def test_overlay_flags_are_cross_platform_no_activate_topmost(self):
        for platform_name in ("darwin", "win32"):
            flags = soundRadar.overlay_window_flags(platform_name)

            self.assertTrue(flags & soundRadar.QtCore.Qt.WindowStaysOnTopHint)
            self.assertTrue(flags & soundRadar.QtCore.Qt.WindowDoesNotAcceptFocus)
            self.assertTrue(flags & soundRadar.QtCore.Qt.Tool)

    def test_native_overlay_dispatch_returns_false_for_unsupported_platform(self):
        window = soundRadar.ParentWidget()

        self.assertFalse(soundRadar.apply_native_overlay_level(window, platform_name="linux"))

    def test_macos_overlay_level_available(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only overlay level")

        self.assertGreaterEqual(soundRadar.macos_overlay_window_level(), 1000)

    def test_native_overlay_level_can_be_applied_without_raise(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only overlay level")

        class ProbeWindow(soundRadar.ParentWidget):
            def __init__(self):
                self.raise_called = False
                super().__init__()

            def raise_(self):
                self.raise_called = True
                raise AssertionError("raise_() should not be used")

        window = ProbeWindow()
        window.show()
        self.app.processEvents()

        self.assertTrue(soundRadar.apply_native_overlay_level(window))
        self.assertFalse(window.raise_called)

        ns_window = soundRadar._objc_msg_send(int(window.winId()), "window", restype=ctypes.c_void_p)
        hides_on_deactivate = soundRadar._objc_msg_send(ns_window, "hidesOnDeactivate", restype=ctypes.c_bool)
        ignores_mouse = soundRadar._objc_msg_send(ns_window, "ignoresMouseEvents", restype=ctypes.c_bool)
        self.assertFalse(hides_on_deactivate)
        self.assertTrue(ignores_mouse)


if __name__ == "__main__":
    unittest.main()
