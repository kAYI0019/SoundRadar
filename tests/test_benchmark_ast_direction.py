from types import SimpleNamespace
import unittest

import numpy as np

from sound_model.benchmark_ast_direction import (
    benchmark_direction_teacher,
    build_parser,
    make_synthetic_audio,
    summarize_timing_rows,
)


class BenchmarkAstDirectionTests(unittest.TestCase):
    def test_make_synthetic_audio_has_requested_shape_and_is_deterministic(self):
        first = make_synthetic_audio(sample_rate=10, seconds=0.5, channel_count=8, seed=123)
        second = make_synthetic_audio(sample_rate=10, seconds=0.5, channel_count=8, seed=123)

        self.assertEqual(first.shape, (5, 8))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)

    def test_summarize_timing_rows_reports_min_median_max(self):
        summary = summarize_timing_rows(
            [
                {"total_ms": 30.0, "model_ms": 20.0},
                {"total_ms": 10.0, "model_ms": 8.0},
                {"total_ms": 20.0, "model_ms": 12.0},
            ]
        )

        self.assertEqual(summary["total_ms"], {"min": 10.0, "median": 20.0, "max": 30.0})
        self.assertEqual(summary["model_ms"], {"min": 8.0, "median": 12.0, "max": 20.0})

    def test_benchmark_direction_teacher_uses_full_seven_direction_batch(self):
        class FakeTeacher:
            def __init__(self):
                self.calls = []

            def profile_predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None, synchronize=False):
                self.calls.append((len(waveforms), sample_rate, top_k, tuple(audio_paths or ())))
                return [], SimpleNamespace(
                    to_jsonable=lambda: {
                        "prepare_ms": 1.0,
                        "feature_ms": 2.0,
                        "to_device_ms": 3.0,
                        "model_ms": 4.0,
                        "postprocess_ms": 5.0,
                        "total_ms": 15.0,
                    }
                )

        teacher = FakeTeacher()
        audio = np.zeros((16, 8), dtype=np.float32)

        result = benchmark_direction_teacher(teacher, audio, sample_rate=16, runs=1, warmups=1, top_k=3)

        self.assertEqual(teacher.calls, [(7, 16, 3, tuple(result["directions"])), (7, 16, 3, tuple(result["directions"]))])
        self.assertEqual(result["batch_size"], 7)
        self.assertEqual(result["summary"]["model_ms"], {"min": 4.0, "median": 4.0, "max": 4.0})

    def test_parser_exposes_cuda_compile_and_attention_options(self):
        args = build_parser().parse_args(
            [
                "sample.wav",
                "--teacher-model",
                "efficientat-mn20",
                "--device",
                "cuda",
                "--dtype",
                "float16",
                "--compile-model",
                "--compile-mode",
                "max-autotune",
                "--attn-implementation",
                "sdpa",
            ]
        )

        self.assertEqual(args.teacher_model, "efficientat-mn20")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.dtype, "float16")
        self.assertTrue(args.compile_model)
        self.assertEqual(args.compile_mode, "max-autotune")
        self.assertEqual(args.attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
