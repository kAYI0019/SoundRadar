from types import SimpleNamespace
import unittest

import numpy as np

from sound_model.direction_events import (
    DIRECTION_NAMES,
    build_parser,
    extract_direction_waveforms,
    score_direction_events,
    top_labels_include_road_vehicle,
)


class DirectionEventScoreTests(unittest.TestCase):
    def test_extract_direction_waveforms_uses_audio_midi_7_1_order(self):
        audio = np.stack([np.full(5, channel, dtype=np.float32) for channel in range(8)], axis=1)

        waveforms = extract_direction_waveforms(audio)

        self.assertEqual(tuple(waveforms), DIRECTION_NAMES)
        expected_channels = {
            "front_left": 0,
            "front": 2,
            "front_right": 1,
            "left": 4,
            "right": 5,
            "rear_left": 6,
            "rear_right": 7,
        }
        for direction, channel in expected_channels.items():
            np.testing.assert_array_equal(waveforms[direction], audio[:, channel])

    def test_extract_direction_waveforms_uses_stereo_fallback_when_7_1_is_missing(self):
        audio = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

        waveforms = extract_direction_waveforms(audio)

        for direction in ("front_left", "left", "rear_left"):
            np.testing.assert_array_equal(waveforms[direction], audio[:, 0])
        for direction in ("front_right", "right", "rear_right"):
            np.testing.assert_array_equal(waveforms[direction], audio[:, 1])
        np.testing.assert_array_equal(waveforms["front"], audio.mean(axis=1))

    def test_extract_direction_waveforms_can_follow_soundradar_stereo_channel_map(self):
        audio = np.stack([np.full(5, channel, dtype=np.float32) for channel in range(8)], axis=1)
        stereo_map = {
            "avg": 0,
            "avd": 1,
            "c": None,
            "g": 0,
            "d": 1,
            "arg": 0,
            "ard": 1,
        }

        waveforms = extract_direction_waveforms(audio, channel_map=stereo_map)

        for direction in ("front_left", "left", "rear_left"):
            np.testing.assert_array_equal(waveforms[direction], audio[:, 0])
        for direction in ("front_right", "right", "rear_right"):
            np.testing.assert_array_equal(waveforms[direction], audio[:, 1])
        np.testing.assert_array_equal(waveforms["front"], audio[:, :2].mean(axis=1))

    def test_score_direction_events_falls_back_to_one_teacher_call_per_direction(self):
        audio = np.stack([np.full(4, channel / 10.0, dtype=np.float32) for channel in range(8)], axis=1)

        class ProbeTeacher:
            def __init__(self):
                self.calls = []

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                self.calls.append((audio_path, sample_rate, top_k, float(np.mean(waveform))))
                score = float(np.mean(waveform))
                return SimpleNamespace(
                    top_labels=[{"label": "probe", "score": score}],
                    soundradar_events={
                        "background": 1.0 - score,
                        "footstep": 0.0,
                        "gunshot": score,
                        "vehicle": 0.0,
                        "explosion": 0.0,
                    },
                    active_events=["gunshot"] if score >= 0.5 else [],
                )

        teacher = ProbeTeacher()
        prediction = score_direction_events(audio, 16000, teacher, top_k=3)

        self.assertEqual([call[0] for call in teacher.calls], list(DIRECTION_NAMES))
        self.assertTrue(all(call[1] == 16000 and call[2] == 3 for call in teacher.calls))
        self.assertAlmostEqual(prediction.direction_event_scores["right"]["gunshot"], 0.5)
        self.assertEqual(prediction.active_events_by_direction["right"], ["gunshot"])
        self.assertEqual(prediction.active_events_by_direction["front_left"], [])

    def test_score_direction_events_batches_teacher_when_available(self):
        audio = np.stack([np.full(4, channel / 10.0, dtype=np.float32) for channel in range(8)], axis=1)

        class BatchTeacher:
            def __init__(self):
                self.batch_calls = []
                self.single_calls = []

            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                self.batch_calls.append((len(waveforms), sample_rate, top_k, tuple(audio_paths or ())))
                predictions = []
                for waveform in waveforms:
                    score = float(np.mean(waveform))
                    predictions.append(
                        SimpleNamespace(
                            top_labels=[{"label": "probe", "score": score}],
                            soundradar_events={
                                "background": 1.0 - score,
                                "footstep": 0.0,
                                "gunshot": score,
                                "vehicle": 0.0,
                                "explosion": 0.0,
                            },
                            active_events=["gunshot"] if score >= 0.5 else [],
                        )
                    )
                return predictions

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                self.single_calls.append(audio_path)
                raise AssertionError("batched teacher should not use single-waveform fallback")

        teacher = BatchTeacher()
        prediction = score_direction_events(audio, 16000, teacher, top_k=3)

        self.assertEqual(teacher.batch_calls, [(7, 16000, 3, tuple(DIRECTION_NAMES))])
        self.assertEqual(teacher.single_calls, [])
        self.assertAlmostEqual(prediction.direction_event_scores["right"]["gunshot"], 0.5)
        self.assertEqual(prediction.active_events_by_direction["right"], ["gunshot"])
        self.assertEqual(prediction.active_events_by_direction["front_left"], [])

    def test_score_direction_events_gates_silent_directions_before_teacher_scores(self):
        audio = np.zeros((32, 8), dtype=np.float32)

        class SaturatingTeacher:
            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                return [
                    SimpleNamespace(
                        top_labels=[{"label": "Boom", "score": 1.0}],
                        soundradar_events={
                            "background": 1.0,
                            "footstep": 1.0,
                            "gunshot": 1.0,
                            "vehicle": 1.0,
                            "explosion": 1.0,
                        },
                        active_events=["footstep", "gunshot", "vehicle", "explosion"],
                    )
                    for _ in waveforms
                ]

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                raise AssertionError("batched teacher should not use single-waveform fallback")

        prediction = score_direction_events(audio, 32000, SaturatingTeacher(), top_k=3)

        for direction in DIRECTION_NAMES:
            self.assertEqual(prediction.direction_event_scores[direction]["background"], 1.0)
            self.assertEqual(prediction.direction_event_scores[direction]["explosion"], 0.0)
            self.assertEqual(prediction.active_events_by_direction[direction], [])

    def test_score_direction_events_promotes_transient_vehicle_false_positive_to_gunshot(self):
        sample_rate = 32_000
        audio = np.zeros((sample_rate, 8), dtype=np.float32)
        burst_len = int(sample_rate * 0.03)
        burst = np.hanning(burst_len).astype(np.float32) * 0.6
        audio[int(sample_rate * 0.4) : int(sample_rate * 0.4) + burst_len, 0] = burst

        class VehicleFalsePositiveTeacher:
            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                return [
                    SimpleNamespace(
                        top_labels=[{"label": "Train wheels squealing", "score": 0.67}],
                        soundradar_events={
                            "background": 0.3,
                            "footstep": 0.0,
                            "gunshot": 0.02,
                            "vehicle": 0.67,
                            "explosion": 0.0,
                        },
                        active_events=["vehicle"],
                    )
                    for _ in waveforms
                ]

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                raise AssertionError("batched teacher should not use single-waveform fallback")

        prediction = score_direction_events(audio, sample_rate, VehicleFalsePositiveTeacher(), top_k=3)

        self.assertGreaterEqual(prediction.direction_event_scores["front_left"]["gunshot"], 0.35)
        self.assertLess(prediction.direction_event_scores["front_left"]["vehicle"], 0.10)
        self.assertEqual(prediction.active_events_by_direction["front_left"], ["gunshot"])

    def test_score_direction_events_keeps_transient_road_vehicle_as_vehicle(self):
        sample_rate = 32_000
        audio = np.zeros((sample_rate, 8), dtype=np.float32)
        burst_len = int(sample_rate * 0.08)
        t = np.arange(burst_len, dtype=np.float32) / sample_rate
        burst = (0.35 * np.sin(2 * np.pi * 130 * t) * np.hanning(burst_len)).astype(np.float32)
        audio[int(sample_rate * 0.3) : int(sample_rate * 0.3) + burst_len, 0] = burst

        class RoadVehicleTeacher:
            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                return [
                    SimpleNamespace(
                        top_labels=[{"label": "Accelerating, revving, vroom", "score": 0.70}],
                        soundradar_events={
                            "background": 0.2,
                            "footstep": 0.0,
                            "gunshot": 0.10,
                            "vehicle": 0.70,
                            "explosion": 0.0,
                        },
                        active_events=["vehicle", "gunshot"],
                    )
                    for _ in waveforms
                ]

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                raise AssertionError("batched teacher should not use single-waveform fallback")

        prediction = score_direction_events(audio, sample_rate, RoadVehicleTeacher(), top_k=3)

        self.assertEqual(prediction.direction_event_scores["front_left"]["vehicle"], 0.70)
        self.assertEqual(prediction.direction_event_scores["front_left"]["gunshot"], 0.0)
        self.assertEqual(prediction.raw_direction_event_scores["front_left"]["gunshot"], 0.10)
        self.assertEqual(prediction.active_events_by_direction["front_left"], ["vehicle"])
        self.assertEqual(prediction.vehicle_gun_decisions_by_direction["front_left"].label, "vehicle")

    def test_top_labels_include_road_vehicle_excludes_non_road_engine_noise(self):
        self.assertTrue(top_labels_include_road_vehicle([{"label": "Engine", "score": 0.8}]))
        self.assertTrue(top_labels_include_road_vehicle([{"label": "Accelerating, revving, vroom", "score": 0.8}]))
        self.assertFalse(top_labels_include_road_vehicle([{"label": "Engine", "score": 0.05}]))
        self.assertFalse(top_labels_include_road_vehicle([{"label": "Light engine (high frequency)", "score": 0.8}]))

    def test_score_direction_events_promotes_unclassified_transient_to_gunshot(self):
        sample_rate = 32_000
        audio = np.zeros((sample_rate, 8), dtype=np.float32)
        burst_len = int(sample_rate * 0.03)
        burst = np.hanning(burst_len).astype(np.float32) * 0.6
        audio[int(sample_rate * 0.4) : int(sample_rate * 0.4) + burst_len, 0] = burst

        class BackgroundTeacher:
            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                return [
                    SimpleNamespace(
                        top_labels=[{"label": "Change ringing (campanology)", "score": 1.0}],
                        soundradar_events={
                            "background": 0.95,
                            "footstep": 0.0,
                            "gunshot": 0.02,
                            "vehicle": 0.02,
                            "explosion": 0.0,
                        },
                        active_events=[],
                    )
                    for _ in waveforms
                ]

            def predict_waveform(self, waveform, sample_rate, *, top_k=12, audio_path="<waveform>"):
                raise AssertionError("batched teacher should not use single-waveform fallback")

        prediction = score_direction_events(audio, sample_rate, BackgroundTeacher(), top_k=3)

        self.assertGreaterEqual(prediction.direction_event_scores["front_left"]["gunshot"], 0.35)
        self.assertEqual(prediction.active_events_by_direction["front_left"], ["gunshot"])

    def test_parser_accepts_efficientat_teacher_selection(self):
        args = build_parser().parse_args(["sample.wav", "--teacher-model", "efficientat-mn10"])

        self.assertEqual(args.teacher_model, "efficientat-mn10")


if __name__ == "__main__":
    unittest.main()
