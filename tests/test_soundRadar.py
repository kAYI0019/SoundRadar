import ctypes
import queue
from types import SimpleNamespace
import sys
import unittest

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

    def test_direction_event_debug_lines_show_compass_hud_and_device(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "left": {"gunshot": 0.0, "footstep": 0.45, "vehicle": 0.0, "explosion": 0.0},
                "rear_right": {"gunshot": 0.0, "footstep": 0.0, "vehicle": 0.39, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"], "left": ["footstep"], "rear_right": ["vehicle"]},
        )

        lines = soundRadar.direction_event_debug_lines(
            prediction,
            threshold=0.1,
            status="idle",
            requested_device="auto",
            resolved_device="mps",
            radar_latency_ms=18.4,
            ast_latency_ms=255.0,
        )

        self.assertEqual(lines[0], "AST idle auto→mps")
        self.assertEqual(lines[1], "lag radar 18ms ast 255ms Δ+237ms")
        self.assertIn("F: --", lines[2])
        self.assertIn("L: FOOT .45", lines[4])
        self.assertIn("R: GUN .80", lines[4])
        self.assertIn("RR: VEH .39", lines[5])

    def test_direction_event_runtime_device_label_shows_mps_resolution(self):
        runtime = soundRadar.DirectionEventRuntime(sample_rate=16_000, channel_count=8, device="auto")
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
