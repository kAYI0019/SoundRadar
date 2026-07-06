from types import SimpleNamespace
import unittest

import numpy as np

from sound_model.direction_events import (
    DIRECTION_NAMES,
    extract_direction_waveforms,
    score_direction_events,
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


if __name__ == "__main__":
    unittest.main()
