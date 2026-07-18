from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from sound_model.audio_features import DEFAULT_CLASSES
from sound_model.ast_teacher import (
    AstAudioSetTeacher,
    EfficientATAudioSetTeacher,
    active_soundradar_events,
    build_audioset_background_indices,
    build_audioset_event_indices,
    create_audio_event_teacher,
    event_label_evidence_from_array,
    configure_torch_runtime_for_device,
    compile_torch_model_if_requested,
    normalize_teacher_model_choice,
    _label_matches,
    _from_pretrained_prefer_cache,
    _load_ast_feature_extractor,
    _resolve_torch_dtype,
    _resolve_torch_device,
    map_audioset_logits_to_events,
    map_audioset_scores_to_events,
    map_audioset_probabilities_to_events,
    top_k_label_scores,
)


class AstTeacherMappingTests(unittest.TestCase):
    def test_normalize_teacher_model_choice_keeps_ast_and_maps_efficientat_aliases(self):
        self.assertEqual(
            normalize_teacher_model_choice("ast"),
            ("ast", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        )
        self.assertEqual(normalize_teacher_model_choice("efficientat-mn10"), ("efficientat", "mn10_as"))
        self.assertEqual(normalize_teacher_model_choice("efficientat-mn20"), ("efficientat", "mn20_as"))
        self.assertEqual(normalize_teacher_model_choice("mn10_as"), ("efficientat", "mn10_as"))
        self.assertEqual(normalize_teacher_model_choice("mn20_as"), ("efficientat", "mn20_as"))

    def test_create_audio_event_teacher_can_select_ast_or_efficientat(self):
        class FakeAstTeacher:
            def __init__(self, model_id, **kwargs):
                self.model_id = model_id
                self.kwargs = kwargs

        class FakeEfficientTeacher:
            def __init__(self, model_id, **kwargs):
                self.model_id = model_id
                self.kwargs = kwargs

        ast_teacher = create_audio_event_teacher(
            "ast",
            device="cpu",
            dtype="float32",
            ast_cls=FakeAstTeacher,
            efficientat_cls=FakeEfficientTeacher,
        )
        efficient_teacher = create_audio_event_teacher(
            "efficientat-mn20",
            device="cpu",
            dtype="float32",
            ast_cls=FakeAstTeacher,
            efficientat_cls=FakeEfficientTeacher,
        )

        self.assertIsInstance(ast_teacher, FakeAstTeacher)
        self.assertEqual(ast_teacher.model_id, "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.assertIsInstance(efficient_teacher, FakeEfficientTeacher)
        self.assertEqual(efficient_teacher.model_id, "mn20_as")
        self.assertEqual(efficient_teacher.kwargs["device"], "cpu")

    def test_efficientat_warmup_direction_batch_runs_seven_zero_waveforms(self):
        class ProbeTeacher(EfficientATAudioSetTeacher):
            def __init__(self):
                self.calls = []

            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                self.calls.append((waveforms, sample_rate, top_k, audio_paths))
                return []

        teacher = ProbeTeacher()

        teacher.warmup_direction_batch(sample_rate=10, seconds=0.5, direction_count=7)

        waveforms, sample_rate, top_k, audio_paths = teacher.calls[0]
        self.assertEqual(sample_rate, 10)
        self.assertEqual(top_k, 1)
        self.assertEqual(audio_paths, ["warmup_0", "warmup_1", "warmup_2", "warmup_3", "warmup_4", "warmup_5", "warmup_6"])
        self.assertEqual(len(waveforms), 7)
        for waveform in waveforms:
            np.testing.assert_array_equal(waveform, np.zeros(5, dtype=np.float32))

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

    def test_configure_torch_runtime_enables_cuda_fast_paths_only_for_cuda(self):
        class FakeDevice:
            def __init__(self, name):
                self.type = name

        class FakeCudnn:
            benchmark = False

        class FakeBackends:
            cudnn = FakeCudnn()

        class FakeTorch:
            backends = FakeBackends()

            def __init__(self):
                self.precision_calls = []

            def set_float32_matmul_precision(self, value):
                self.precision_calls.append(value)

        torch_module = FakeTorch()

        configure_torch_runtime_for_device(torch_module, FakeDevice("cuda"))

        self.assertTrue(torch_module.backends.cudnn.benchmark)
        self.assertEqual(torch_module.precision_calls, ["high"])

        torch_module.backends.cudnn.benchmark = False
        torch_module.precision_calls.clear()
        configure_torch_runtime_for_device(torch_module, FakeDevice("mps"))

        self.assertFalse(torch_module.backends.cudnn.benchmark)
        self.assertEqual(torch_module.precision_calls, [])

    def test_compile_torch_model_if_requested_uses_reduce_overhead_by_default(self):
        class FakeTorch:
            def __init__(self):
                self.calls = []

            def compile(self, model, *, mode=None):
                self.calls.append((model, mode))
                return f"compiled:{mode}:{model}"

        torch_module = FakeTorch()

        self.assertEqual(compile_torch_model_if_requested(torch_module, "model", True), "compiled:reduce-overhead:model")
        self.assertEqual(torch_module.calls, [("model", "reduce-overhead")])

    def test_compile_torch_model_if_requested_accepts_explicit_mode(self):
        class FakeTorch:
            def compile(self, model, *, mode=None):
                return (model, mode)

        self.assertEqual(compile_torch_model_if_requested(FakeTorch(), "model", "max-autotune"), ("model", "max-autotune"))

    def test_compile_torch_model_if_requested_rejects_missing_compile(self):
        with self.assertRaisesRegex(RuntimeError, "torch.compile"):
            compile_torch_model_if_requested(object(), "model", True)

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

    def test_label_matching_does_not_match_keyword_inside_unrelated_words(self):
        self.assertTrue(_label_matches("Car alarm", ("car",)))
        self.assertTrue(_label_matches("Gunshot, gunfire", ("gunfire",)))
        self.assertTrue(_label_matches("Walk, footsteps", ("walk, footsteps",)))

        self.assertFalse(_label_matches("Carnatic music", ("car",)))
        self.assertFalse(_label_matches("Grunt", ("run",)))
        self.assertFalse(_label_matches("Grunge", ("run",)))

    def test_efficientat_logit_mapping_uses_relative_evidence_not_saturated_sigmoid(self):
        labels = (
            "Motor vehicle (road)",
            "Speech",
            "Gunshot, gunfire",
            "Explosion",
            "Boom",
        )
        # EfficientAT can emit very large positive logits where sigmoid(logit)
        # would be ~1.0 for every target class. The mapping should keep the top
        # relative event while not promoting distant positive logits.
        logits = np.array([120.0, 110.0, 95.0, 80.0, 70.0], dtype=np.float32)

        mapped = map_audioset_logits_to_events(logits, labels, temperature=8.0)

        self.assertGreater(mapped["vehicle"], 0.9)
        self.assertLess(mapped["gunshot"], 0.1)
        self.assertLess(mapped["explosion"], 0.1)

    def test_vehicle_mapping_includes_engine_and_road_vehicle_motion_labels(self):
        mapped = map_audioset_scores_to_events(
            {
                "Accelerating, revving, vroom": 0.74,
                "Idling": 0.70,
                "Engine": 0.66,
                "Traffic noise, roadway noise": 0.62,
                "Speech": 0.95,
            }
        )

        self.assertAlmostEqual(mapped["vehicle"], 0.74)

    def test_vehicle_mapping_ignores_broad_mechanical_false_positives(self):
        labels = (
            "Train wheels squealing",
            "Gunshot, gunfire",
        )
        logits = np.array([120.0, 80.0], dtype=np.float32)

        mapped = map_audioset_logits_to_events(logits, labels, temperature=8.0)

        self.assertEqual(mapped["vehicle"], 0.0)

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

    def test_probability_mapping_matches_label_score_mapping(self):
        labels = (
            "Speech",
            "Gunshot, gunfire",
            "Walk, footsteps",
            "Car",
            "Explosion",
            "Silence",
        )
        probabilities = np.array([0.95, 0.42, 0.31, 0.22, 0.55, 0.77], dtype=np.float32)

        event_indices = build_audioset_event_indices(labels)
        background_indices = build_audioset_background_indices(labels)
        indexed = map_audioset_probabilities_to_events(
            probabilities,
            labels,
            event_indices=event_indices,
            background_indices=background_indices,
        )
        dict_mapped = map_audioset_scores_to_events(dict(zip(labels, probabilities)))

        for event_name in DEFAULT_CLASSES:
            self.assertAlmostEqual(indexed[event_name], dict_mapped[event_name])

    def test_top_k_label_scores_returns_only_requested_sorted_labels(self):
        labels = ("quiet", "gun", "foot", "vehicle")
        probabilities = np.array([0.1, 0.8, 0.3, 0.6], dtype=np.float32)

        top_labels = top_k_label_scores(probabilities, labels, top_k=2)

        self.assertEqual(top_labels, [{"label": "gun", "score": 0.8}, {"label": "vehicle", "score": 0.6}])

    def test_vehicle_gun_evidence_preserves_relevant_scores_outside_top_k(self):
        labels = (
            "Speech",
            "Music",
            "Gunshot, gunfire",
            "Motor vehicle (road)",
            "Light engine (high frequency)",
        )
        scores = np.array([0.99, 0.95, 0.42, 0.31, 0.88], dtype=np.float32)

        self.assertEqual([item["label"] for item in top_k_label_scores(scores, labels, top_k=2)], ["Speech", "Music"])
        evidence = event_label_evidence_from_array(scores, labels)

        self.assertAlmostEqual(evidence["gunshot"]["Gunshot, gunfire"], 0.42)
        self.assertAlmostEqual(evidence["vehicle"]["Motor vehicle (road)"], 0.31)
        self.assertNotIn("Light engine (high frequency)", evidence["vehicle"])

    def test_predict_waveforms_uses_torch_inference_mode(self):
        class FakeInput:
            def is_floating_point(self):
                return True

            def to(self, *args, **kwargs):
                self.to_args = args
                self.to_kwargs = kwargs
                return self

        class FakeProbabilities:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0.0, 0.6, 0.0, 0.0]], dtype=np.float32)

        class FakeContext:
            def __init__(self, torch_module, name):
                self.torch_module = torch_module
                self.name = name

            def __enter__(self):
                setattr(self.torch_module, f"{self.name}_entered", True)

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            inference_mode_entered = False
            no_grad_entered = False

            def inference_mode(self):
                return FakeContext(self, "inference_mode")

            def no_grad(self):
                return FakeContext(self, "no_grad")

            def sigmoid(self, logits):
                return FakeProbabilities()

        class FakeModel:
            config = SimpleNamespace(id2label={0: "Speech", 1: "Gunshot, gunfire", 2: "Car", 3: "Silence"})

            def __call__(self, **inputs):
                return SimpleNamespace(logits=object())

        fake_torch = FakeTorch()
        teacher = AstAudioSetTeacher.__new__(AstAudioSetTeacher)
        teacher.model_id = "fake-model"
        teacher.torch = fake_torch
        teacher.feature_extractor = lambda waveforms, sampling_rate, return_tensors: {"input_values": FakeInput()}
        teacher.model = FakeModel()
        teacher.dtype = "float16"
        teacher.device = "cuda"
        teacher._labels = ("Speech", "Gunshot, gunfire", "Car", "Silence")
        teacher._event_label_indices = build_audioset_event_indices(teacher._labels)
        teacher._background_label_indices = build_audioset_background_indices(teacher._labels)

        teacher.predict_waveforms([np.zeros(4, dtype=np.float32)], 16_000, top_k=2)

        self.assertTrue(fake_torch.inference_mode_entered)
        self.assertFalse(fake_torch.no_grad_entered)

    def test_warmup_direction_batch_runs_seven_zero_waveforms(self):
        class ProbeTeacher(AstAudioSetTeacher):
            def __init__(self):
                self.calls = []

            def predict_waveforms(self, waveforms, sample_rate, *, top_k=12, audio_paths=None):
                self.calls.append((waveforms, sample_rate, top_k, audio_paths))
                return []

        teacher = ProbeTeacher()

        teacher.warmup_direction_batch(sample_rate=10, seconds=0.5, direction_count=7)

        waveforms, sample_rate, top_k, audio_paths = teacher.calls[0]
        self.assertEqual(sample_rate, 10)
        self.assertEqual(top_k, 1)
        self.assertEqual(audio_paths, ["warmup_0", "warmup_1", "warmup_2", "warmup_3", "warmup_4", "warmup_5", "warmup_6"])
        self.assertEqual(len(waveforms), 7)
        for waveform in waveforms:
            np.testing.assert_array_equal(waveform, np.zeros(5, dtype=np.float32))

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
