import unittest
from types import SimpleNamespace

import numpy as np

from sound_model.direction_runtime import DirectionEventRuntime, normalize_analysis_timing


class ImmediateExecutor:
    def submit(self, fn, *args):
        class DoneFuture:
            def done(self):
                return True

            def result(self):
                return fn(*args)

        return DoneFuture()


class DirectionRuntimeTests(unittest.TestCase):
    def test_interval_is_capped_to_window_to_prevent_analysis_gaps(self):
        self.assertEqual(normalize_analysis_timing(1.0, 1.25), (1.0, 1.0))
        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=1,
            window_seconds=0.5,
            interval_seconds=0.75,
            executor=ImmediateExecutor(),
            score_fn=lambda *args: SimpleNamespace(direction_event_scores={}),
        )

        self.assertEqual(runtime.interval_seconds, 0.5)
        self.assertTrue(runtime.interval_was_capped)

    def test_overlapping_windows_keep_boundary_impulse_in_analyzed_audio(self):
        calls = []

        def score_fn(audio, sample_rate, top_k, source_path):
            calls.append(audio.copy())
            return SimpleNamespace(direction_event_scores={})

        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=1,
            window_seconds=1.0,
            interval_seconds=0.75,
            executor=ImmediateExecutor(),
            score_fn=score_fn,
        )
        runtime.append_blocks([np.zeros((10, 1), dtype=np.float32)])
        runtime.maybe_submit(now=1.0)
        boundary_block = np.zeros((8, 1), dtype=np.float32)
        boundary_block[0, 0] = 1.0
        runtime.append_blocks([boundary_block])

        runtime.maybe_submit(now=1.75)

        self.assertEqual(len(calls), 2)
        self.assertEqual(float(np.max(calls[1])), 1.0)

    def test_busy_inference_is_skipped_instead_of_queued(self):
        class PendingFuture:
            def done(self):
                return False

        class PendingExecutor:
            def __init__(self):
                self.submissions = 0

            def submit(self, fn, *args):
                self.submissions += 1
                return PendingFuture()

        executor = PendingExecutor()
        runtime = DirectionEventRuntime(
            sample_rate=10,
            channel_count=1,
            window_seconds=1.0,
            interval_seconds=0.5,
            executor=executor,
            score_fn=lambda *args: None,
        )
        runtime.append_blocks([np.ones((10, 1), dtype=np.float32)])

        runtime.maybe_submit(now=1.0)
        runtime.maybe_submit(now=1.5)
        runtime.maybe_submit(now=2.0)

        self.assertEqual(executor.submissions, 1)
        self.assertEqual(runtime.submitted_inference_count, 1)
        self.assertEqual(runtime.busy_skip_count, 2)

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
