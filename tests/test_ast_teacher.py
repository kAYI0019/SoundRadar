import unittest
import warnings

from sound_model.audio_features import DEFAULT_CLASSES
from sound_model.ast_teacher import (
    active_soundradar_events,
    _from_pretrained_prefer_cache,
    _load_ast_feature_extractor,
    _resolve_torch_dtype,
    _resolve_torch_device,
    map_audioset_scores_to_events,
)


class AstTeacherMappingTests(unittest.TestCase):
    def test_auto_device_prefers_cuda_then_mps_then_cpu(self):
        class FakeDevice:
            def __init__(self, name):
                self.type = name

            def __str__(self):
                return self.type

        class Availability:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        class FakeBackends:
            def __init__(self, mps_available):
                self.mps = Availability(mps_available)

        class FakeTorch:
            def __init__(self, cuda_available, mps_available):
                self.cuda = Availability(cuda_available)
                self.backends = FakeBackends(mps_available)

            def device(self, name):
                return FakeDevice(name)

        self.assertEqual(_resolve_torch_device(FakeTorch(True, True), "auto").type, "cuda")
        self.assertEqual(_resolve_torch_device(FakeTorch(False, True), "auto").type, "mps")
        self.assertEqual(_resolve_torch_device(FakeTorch(False, False), "auto").type, "cpu")

    def test_auto_dtype_uses_half_precision_on_gpu_and_float32_on_cpu(self):
        class FakeDType:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

        class FakeDevice:
            def __init__(self, name):
                self.type = name

        class FakeTorch:
            float16 = FakeDType("torch.float16")
            bfloat16 = FakeDType("torch.bfloat16")
            float32 = FakeDType("torch.float32")

        self.assertIs(_resolve_torch_dtype(FakeTorch, "auto", FakeDevice("cuda")), FakeTorch.float16)
        self.assertIs(_resolve_torch_dtype(FakeTorch, "auto", FakeDevice("mps")), FakeTorch.float16)
        self.assertIs(_resolve_torch_dtype(FakeTorch, "auto", FakeDevice("cpu")), FakeTorch.float32)
        self.assertIs(_resolve_torch_dtype(FakeTorch, "bfloat16", FakeDevice("cuda")), FakeTorch.bfloat16)

    def test_requested_cuda_requires_available_cuda_backend(self):
        class FakeDevice:
            def __init__(self, name):
                self.type = name

        class Cuda:
            def is_available(self):
                return False

        class FakeTorch:
            cuda = Cuda()

            def device(self, name):
                return FakeDevice(name)

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            _resolve_torch_device(FakeTorch(), "cuda")

    def test_maps_audioset_labels_to_soundradar_events(self):
        mapped = map_audioset_scores_to_events(
            {
                "Gunshot, gunfire": 0.91,
                "Explosion": 0.72,
                "Car": 0.64,
                "Walk, footsteps": 0.55,
                "Speech": 0.99,
            }
        )

        self.assertEqual(set(mapped), set(DEFAULT_CLASSES))
        self.assertAlmostEqual(mapped["gunshot"], 0.91)
        self.assertAlmostEqual(mapped["explosion"], 0.72)
        self.assertAlmostEqual(mapped["vehicle"], 0.64)
        self.assertAlmostEqual(mapped["footstep"], 0.55)

    def test_background_means_no_target_sound_event(self):
        mapped = map_audioset_scores_to_events({"Speech": 0.95, "Music": 0.88})

        self.assertEqual(mapped["gunshot"], 0.0)
        self.assertEqual(mapped["explosion"], 0.0)
        self.assertEqual(mapped["vehicle"], 0.0)
        self.assertEqual(mapped["footstep"], 0.0)
        self.assertAlmostEqual(mapped["background"], 1.0)

    def test_background_drops_when_target_event_is_confident(self):
        mapped = map_audioset_scores_to_events({"Gunshot, gunfire": 0.84})

        self.assertAlmostEqual(mapped["background"], 0.16)

    def test_ast_teacher_uses_lower_default_threshold_for_gunshot(self):
        active = active_soundradar_events(
            {
                "background": 0.89,
                "footstep": 0.0,
                "gunshot": 0.11,
                "vehicle": 0.0,
                "explosion": 0.0,
            }
        )

        self.assertEqual(active, ["gunshot"])

    def test_feature_extractor_load_suppresses_only_known_ast_mel_warning(self):
        sentinel = object()
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            self.assertEqual(model_id, "test-model")
            calls.append(kwargs)
            warnings.warn(
                "At least one mel filter has all zero values. "
                "The value for `num_mel_filters` (128) may be set too high. "
                "Or, the value for `num_frequency_bins` (257) may be set too low.",
                UserWarning,
            )
            warnings.warn("different warning", UserWarning)
            return sentinel

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = _load_ast_feature_extractor(
                "test-model",
                from_pretrained=fake_from_pretrained,
            )

        self.assertIs(loaded, sentinel)
        self.assertEqual(calls, [{"local_files_only": True}])
        self.assertEqual([str(warning.message) for warning in caught], ["different warning"])

    def test_from_pretrained_prefers_cache_before_hub_lookup(self):
        sentinel = object()
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            self.assertEqual(model_id, "test-model")
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return sentinel

        loaded = _from_pretrained_prefer_cache(
            fake_from_pretrained,
            "test-model",
            revision="main",
        )

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [{"local_files_only": True, "revision": "main"}, {"revision": "main"}],
        )


if __name__ == "__main__":
    unittest.main()
