import unittest
from types import SimpleNamespace

import numpy as np

from sound_model.direction_runtime import DirectionEventRuntime


class ImmediateExecutor:
    def submit(self, fn, *args):
        class DoneFuture:
            def done(self):
                return True

            def result(self):
                return fn(*args)

        return DoneFuture()


class DirectionRuntimeTests(unittest.TestCase):
    def test_runtime_scores_only_the_latest_audio_window_without_qt(self):
        calls = []

        def score_fn(audio, sample_rate, top_k, source_path):
            calls.append((audio.copy(), sample_rate, top_k, source_path))
            return SimpleNamespace(direction_event_scores={}, active_events_by_direction={})

        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=2,
            window_seconds=0.3,
            interval_seconds=0.5,
            top_k=4,
            executor=ImmediateExecutor(),
            score_fn=score_fn,
        )
        runtime.append_blocks([np.ones((4, 2), dtype=np.float32)])

        prediction = runtime.maybe_submit(now=0.1)
        blocked = runtime.maybe_submit(now=0.2)

        self.assertIs(prediction, runtime.latest_prediction)
        self.assertIsNone(blocked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].shape, (3, 2))
        self.assertEqual(calls[0][1:], (10, 4, "<live>"))

    def test_runtime_tracks_latency_from_the_latest_capture_time(self):
        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=2,
            window_seconds=0.3,
            interval_seconds=0.5,
            executor=ImmediateExecutor(),
            score_fn=lambda audio, sample_rate, top_k, source_path: object(),
            latency_clock=lambda: 10.75,
        )
        runtime.append_blocks([np.ones((4, 2), dtype=np.float32)], capture_time=10.0)

        runtime.maybe_submit(now=0.1)

        self.assertAlmostEqual(runtime.latest_latency_ms, 750.0)

    def test_runtime_disables_itself_after_scoring_failure(self):
        def fail_score(audio, sample_rate, top_k, source_path):
            raise RuntimeError("teacher unavailable")

        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=2,
            window_seconds=0.2,
            executor=ImmediateExecutor(),
            score_fn=fail_score,
        )
        runtime.append_blocks([np.ones((2, 2), dtype=np.float32)])

        prediction = runtime.maybe_submit(now=0.1)

        self.assertIsNone(prediction)
        self.assertEqual(runtime.disabled_reason, "teacher unavailable")
        self.assertIsNone(runtime.maybe_submit(now=10.0))


if __name__ == "__main__":
    unittest.main()
