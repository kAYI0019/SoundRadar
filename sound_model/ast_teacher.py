"""AST AudioSet teacher inference for SoundRadar event bootstrapping.

The model requested for the first teacher pass is
``MIT/ast-finetuned-audioset-10-10-0.4593``.  This module keeps the heavy
Hugging Face imports lazy so the rest of SoundRadar and the unit tests still run
when optional ML dependencies are not installed.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import warnings
from typing import Mapping

import numpy as np

from .audio_features import DEFAULT_CLASSES, read_wav, resample_linear

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16_000
DEFAULT_TEACHER_MODEL = "ast"
EFFICIENTAT_REPO_URL = "https://github.com/fschmid56/EfficientAT.git"
EFFICIENTAT_SAMPLE_RATE = 32_000
EFFICIENTAT_DEFAULT_MODEL = "mn10_as"
EFFICIENTAT_MODEL_ALIASES: dict[str, str] = {
    "efficientat-mn10": "mn10_as",
    "efficientat_mn10": "mn10_as",
    "mn10": "mn10_as",
    "mn10_as": "mn10_as",
    "efficientat-mn20": "mn20_as",
    "efficientat_mn20": "mn20_as",
    "mn20": "mn20_as",
    "mn20_as": "mn20_as",
}
TEACHER_MODEL_CHOICES = ("ast", "efficientat-mn10", "efficientat-mn20", "mn10_as", "mn20_as")
AST_MEL_FILTER_WARNING_PATTERN = (
    r"At least one mel filter has all zero values\..*"
    r"num_mel_filters.*num_frequency_bins.*"
)

# Keep vehicle/gun label routing in one place.  The same mapping is used for
# coarse event scores and for the full (not Top-K-limited) resolver evidence.
GUNSHOT_LABEL_KEYWORDS = (
    "gunshot",
    "gunfire",
    "firearm",
    "machine gun",
    "fusillade",
    "artillery fire",
    "cap gun",
)
ROAD_VEHICLE_LABEL_KEYWORDS = (
    "vehicle",
    "motor vehicle",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "engine",
    "engine starting",
    "idling",
    "accelerating, revving, vroom",
    "traffic noise",
    "roadway noise",
    "tire squeal",
)
NON_ROAD_ENGINE_LABEL_KEYWORDS = (
    "light engine",
    "medium engine",
    "heavy engine",
    "aircraft engine",
    "jet engine",
)
VEHICLE_GUN_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gunshot": GUNSHOT_LABEL_KEYWORDS,
    "vehicle": ROAD_VEHICLE_LABEL_KEYWORDS,
}

# These map AudioSet labels into the coarse events used by the accessibility
# overlay, not into exact PUBG semantic classes.
EVENT_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "footstep": (
        "footstep",
        "footsteps",
        "walk, footsteps",
        "walking",
        "run",
        "running",
        "shuffle",
    ),
    "gunshot": GUNSHOT_LABEL_KEYWORDS,
    "vehicle": ROAD_VEHICLE_LABEL_KEYWORDS,
    "explosion": (
        "explosion",
        "blast",
        "boom",
        "fireworks",
        "grenade",
    ),
}

BACKGROUND_HINT_KEYWORDS = (
    "silence",
    "inside, small room",
    "inside, large room or hall",
    "outside, rural or natural",
    "outside, urban or manmade",
    "ambient noise",
)

DEFAULT_EVENT_THRESHOLDS: dict[str, float] = {
    "footstep": 0.25,
    # AST AudioSet sigmoid scores for in-game gunshots can be well below 0.25
    # while still ranking gunshot/machine-gun at the top.  Keep this teacher
    # threshold permissive; realtime UI should still apply smoothing/cooldown.
    "gunshot": 0.10,
    "vehicle": 0.25,
    "explosion": 0.25,
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
EFFICIENTAT_RELATIVE_LOGIT_TEMPERATURE = 8.0


def _clamp_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _label_matches(label: str, keywords: tuple[str, ...]) -> bool:
    """Return True when a keyword matches label tokens, not arbitrary substrings."""

    label_tokens = tuple(TOKEN_PATTERN.findall(label.lower()))
    if not label_tokens:
        return False
    for keyword in keywords:
        keyword_tokens = tuple(TOKEN_PATTERN.findall(keyword.lower()))
        if not keyword_tokens:
            continue
        if len(keyword_tokens) == 1:
            if keyword_tokens[0] in label_tokens:
                return True
            continue
        window = len(keyword_tokens)
        for offset in range(0, len(label_tokens) - window + 1):
            if label_tokens[offset : offset + window] == keyword_tokens:
                return True
    return False


def map_audioset_scores_to_events(
    label_scores: Mapping[str, float],
    *,
    event_keywords: Mapping[str, tuple[str, ...]] = EVENT_LABEL_KEYWORDS,
) -> dict[str, float]:
    """Map AudioSet label probabilities to SoundRadar V0 event probabilities.

    ``background`` means "none of the target SoundRadar events is likely".  A
    high unrelated AudioSet class such as Speech therefore still maps to high
    background unless one of footstep/gunshot/vehicle/explosion is also high.
    """

    mapped = {name: 0.0 for name in DEFAULT_CLASSES}
    for label, raw_score in label_scores.items():
        score = _clamp_probability(float(raw_score))
        for event_name, keywords in event_keywords.items():
            if event_name in mapped and _label_matches(label, keywords):
                mapped[event_name] = max(mapped[event_name], score)

    target_max = max(mapped[event] for event in mapped if event != "background")
    explicit_background = 0.0
    for label, raw_score in label_scores.items():
        if _label_matches(label, BACKGROUND_HINT_KEYWORDS):
            explicit_background = max(explicit_background, _clamp_probability(float(raw_score)))
    mapped["background"] = max(explicit_background, 1.0 - target_max)
    return {name: _clamp_probability(mapped[name]) for name in DEFAULT_CLASSES}


def event_label_evidence_from_scores(
    label_scores: Mapping[str, float],
    *,
    label_keywords: Mapping[str, tuple[str, ...]] = VEHICLE_GUN_LABEL_KEYWORDS,
) -> dict[str, dict[str, float]]:
    """Preserve all vehicle/gun AudioSet label scores, independent of Top-K."""

    evidence: dict[str, dict[str, float]] = {event_name: {} for event_name in label_keywords}
    for label, raw_score in label_scores.items():
        for event_name, keywords in label_keywords.items():
            if event_name == "vehicle" and _label_matches(label, NON_ROAD_ENGINE_LABEL_KEYWORDS):
                continue
            if _label_matches(label, keywords):
                evidence[event_name][str(label)] = _clamp_probability(float(raw_score))
    return evidence


def event_label_evidence_from_array(
    scores: np.ndarray,
    labels: tuple[str, ...] | list[str],
) -> dict[str, dict[str, float]]:
    """Vector-friendly adapter used by both AST and EfficientAT backends."""

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    return event_label_evidence_from_scores(
        {
            _label_from_sequence(labels, index): float(values[index])
            for index in range(min(len(labels), int(values.size)))
        }
    )


def active_soundradar_events(
    mapped_scores: Mapping[str, float],
    thresholds: Mapping[str, float] = DEFAULT_EVENT_THRESHOLDS,
) -> list[str]:
    """Return non-background events whose mapped score passes its threshold."""

    active: list[str] = []
    for event_name, threshold in thresholds.items():
        if mapped_scores.get(event_name, 0.0) >= threshold:
            active.append(event_name)
    return active


def build_audioset_event_indices(
    labels: tuple[str, ...] | list[str],
    *,
    event_keywords: Mapping[str, tuple[str, ...]] = EVENT_LABEL_KEYWORDS,
) -> dict[str, np.ndarray]:
    """Precompute AudioSet label indices for each SoundRadar event."""

    return {
        event_name: np.array(
            [index for index, label in enumerate(labels) if _label_matches(label, keywords)],
            dtype=np.int64,
        )
        for event_name, keywords in event_keywords.items()
    }


def build_audioset_background_indices(labels: tuple[str, ...] | list[str]) -> np.ndarray:
    """Precompute AudioSet label indices that explicitly mean background/ambient."""

    return np.array(
        [index for index, label in enumerate(labels) if _label_matches(label, BACKGROUND_HINT_KEYWORDS)],
        dtype=np.int64,
    )


def map_audioset_probabilities_to_events(
    probabilities: np.ndarray,
    labels: tuple[str, ...] | list[str],
    *,
    event_indices: Mapping[str, np.ndarray] | None = None,
    background_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Map model probabilities to SoundRadar events without a full label-score dict."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    event_indices = event_indices or build_audioset_event_indices(labels)
    if background_indices is None:
        background_indices = build_audioset_background_indices(labels)

    mapped = {name: 0.0 for name in DEFAULT_CLASSES}
    for event_name, indices in event_indices.items():
        if event_name in mapped and len(indices):
            mapped[event_name] = _clamp_probability(float(np.max(probabilities[indices])))

    target_max = max(mapped[event] for event in mapped if event != "background")
    explicit_background = 0.0
    if len(background_indices):
        explicit_background = _clamp_probability(float(np.max(probabilities[background_indices])))
    mapped["background"] = max(explicit_background, 1.0 - target_max)
    return {name: _clamp_probability(mapped[name]) for name in DEFAULT_CLASSES}


def relative_logit_scores(logits: np.ndarray, *, temperature: float = EFFICIENTAT_RELATIVE_LOGIT_TEMPERATURE) -> np.ndarray:
    """Convert logits to relative 0..1 evidence without sigmoid saturation.

    EfficientAT MN models can produce very large positive logits on live/silent
    game-routing windows. Applying sigmoid makes nearly every AudioSet class
    look like probability 1.0, so for EfficientAT we score labels by their
    distance from the best label in the same clip instead.
    """

    logits = np.asarray(logits, dtype=np.float32)
    if logits.size == 0:
        return logits.astype(np.float32, copy=False)
    scale = max(float(temperature), 1.0e-6)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    relative = np.clip((logits - max_logits) / scale, -80.0, 0.0)
    return np.exp(relative).astype(np.float32, copy=False)


def map_audioset_logits_to_events(
    logits: np.ndarray,
    labels: tuple[str, ...] | list[str],
    *,
    event_indices: Mapping[str, np.ndarray] | None = None,
    background_indices: np.ndarray | None = None,
    temperature: float = EFFICIENTAT_RELATIVE_LOGIT_TEMPERATURE,
) -> dict[str, float]:
    """Map AudioSet logits to SoundRadar events using relative evidence.

    This is intended for EfficientAT-style logits whose absolute sigmoid values
    are not calibrated enough for thresholding in the overlay. A target event is
    strong only when one of its AudioSet labels is close to the clip's best label.
    """

    logits = np.asarray(logits, dtype=np.float32)
    event_indices = event_indices or build_audioset_event_indices(labels)
    if background_indices is None:
        background_indices = build_audioset_background_indices(labels)
    relative_scores = relative_logit_scores(logits, temperature=temperature)

    mapped = {name: 0.0 for name in DEFAULT_CLASSES}
    for event_name, indices in event_indices.items():
        if event_name in mapped and len(indices):
            mapped[event_name] = _clamp_probability(float(np.max(relative_scores[indices])))

    target_max = max(mapped[event] for event in mapped if event != "background")
    explicit_background = 0.0
    if len(background_indices):
        explicit_background = _clamp_probability(float(np.max(relative_scores[background_indices])))
    mapped["background"] = max(explicit_background, 1.0 - target_max)
    return {name: _clamp_probability(mapped[name]) for name in DEFAULT_CLASSES}


def _label_from_sequence(labels: tuple[str, ...] | list[str], index: int) -> str:
    if 0 <= index < len(labels):
        return str(labels[index])
    return str(index)


def top_k_label_scores(probabilities: np.ndarray, labels: tuple[str, ...] | list[str], top_k: int) -> list[dict[str, object]]:
    """Return top-k label scores using partial sort for the hot path."""

    probabilities = np.asarray(probabilities)
    if top_k <= 0 or probabilities.size == 0:
        return []
    k = min(int(top_k), int(probabilities.size))
    if k == probabilities.size:
        top_indices = np.argsort(probabilities)[::-1]
    else:
        candidates = np.argpartition(probabilities, -k)[-k:]
        top_indices = candidates[np.argsort(probabilities[candidates])[::-1]]
    return [
        {"label": _label_from_sequence(labels, int(index)), "score": float(round(float(probabilities[int(index)]), 6))}
        for index in top_indices[:k]
    ]


def _torch_inference_context(torch_module):
    inference_mode = getattr(torch_module, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    return torch_module.no_grad()


def normalize_teacher_model_choice(teacher_model: str | None, model_id: str | None = None) -> tuple[str, str]:
    """Normalize public teacher aliases into ``(backend, model_id)``.

    ``ast`` keeps the existing Hugging Face AST teacher.  EfficientAT aliases map
    to the upstream release names used by fschmid56/EfficientAT, currently
    ``mn10_as`` and ``mn20_as``.
    """

    requested = (teacher_model or DEFAULT_TEACHER_MODEL).strip().lower()
    if requested in {"ast", "ast-audioset", "mit-ast"}:
        return "ast", model_id or AST_MODEL_ID
    if requested in EFFICIENTAT_MODEL_ALIASES:
        return "efficientat", model_id or EFFICIENTAT_MODEL_ALIASES[requested]
    if requested.startswith("efficientat:"):
        efficientat_model_id = requested.split(":", 1)[1].strip()
        if efficientat_model_id:
            return "efficientat", model_id or efficientat_model_id
    raise ValueError(
        f"Unsupported teacher model {teacher_model!r}. Choose one of: {', '.join(TEACHER_MODEL_CHOICES)}"
    )


def create_audio_event_teacher(
    teacher_model: str | None = DEFAULT_TEACHER_MODEL,
    *,
    model_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    compile_model=False,
    ast_cls=None,
    efficientat_cls=None,
):
    """Create the selected AudioSet teacher while keeping AST as the default."""

    backend, resolved_model_id = normalize_teacher_model_choice(teacher_model, model_id=model_id)
    ast_cls = AstAudioSetTeacher if ast_cls is None else ast_cls
    efficientat_cls = EfficientATAudioSetTeacher if efficientat_cls is None else efficientat_cls
    if backend == "ast":
        return ast_cls(
            resolved_model_id,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            compile_model=compile_model,
        )
    return efficientat_cls(
        resolved_model_id,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
    )


def synchronize_torch_device(torch_module, device) -> None:
    """Synchronize CUDA/MPS for accurate profiling; no-op on CPU/unsupported backends."""

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    try:
        if device_type == "cuda" and getattr(torch_module, "cuda", None) is not None:
            torch_module.cuda.synchronize(device)
        elif device_type == "mps" and getattr(torch_module, "mps", None) is not None:
            torch_module.mps.synchronize()
    except Exception:
        # Synchronization is a profiling aid only; never break inference because
        # a backend does not support explicit synchronization in this runtime.
        return


def configure_torch_runtime_for_device(
    torch_module,
    device,
    *,
    matmul_precision: str | None = "high",
    cudnn_benchmark: bool = True,
) -> None:
    """Enable safe CUDA runtime knobs without affecting CPU/MPS behavior."""

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type != "cuda":
        return

    cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "benchmark"):
        cudnn.benchmark = bool(cudnn_benchmark)

    set_precision = getattr(torch_module, "set_float32_matmul_precision", None)
    if matmul_precision and callable(set_precision):
        set_precision(matmul_precision)


def compile_torch_model_if_requested(torch_module, model, compile_model=False):
    """Optionally compile the model for CUDA benchmarking/runtime experiments."""

    if not compile_model:
        return model
    compiler = getattr(torch_module, "compile", None)
    if not callable(compiler):
        raise RuntimeError("Requested torch.compile, but this torch build does not expose torch.compile")
    mode = "reduce-overhead" if compile_model is True else str(compile_model)
    return compiler(model, mode=mode)


def load_audio_channels(path: str | Path) -> tuple[np.ndarray, int]:
    """Load WAV/MP3/etc. while preserving available channels."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".wav":
        return read_wav(path)
    return _read_with_afconvert(path)


def load_audio_mono(path: str | Path, *, target_rate: int = AST_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load WAV/MP3/etc. as mono float32 audio at ``target_rate``.

    WAV is read directly with the local lightweight loader.  Non-WAV files are
    converted through macOS ``afconvert`` when available; this keeps MP3 support
    out of the Python dependency set.
    """

    audio, sample_rate = load_audio_channels(path)

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio.astype(np.float32, copy=False)
    mono = resample_linear(mono[:, None], sample_rate, target_rate)[:, 0]
    return mono.astype(np.float32, copy=False), target_rate


def _read_with_afconvert(path: Path) -> tuple[np.ndarray, int]:
    afconvert = shutil.which("afconvert")
    if afconvert is None:
        raise RuntimeError(
            f"Cannot read non-WAV audio without afconvert on PATH: {path}. "
            "Convert it to WAV first or install an audio decoder."
        )

    with tempfile.TemporaryDirectory(prefix="soundradar-ast-") as tmpdir:
        wav_path = Path(tmpdir) / f"{path.stem}.wav"
        command = [afconvert, "-f", "WAVE", "-d", "LEI16", str(path), str(wav_path)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"afconvert failed for {path}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return read_wav(wav_path)


@dataclass
class AstPrediction:
    audio_path: str
    model_id: str
    duration_sec: float
    top_labels: list[dict[str, object]]
    soundradar_events: dict[str, float]
    active_events: list[str]
    event_label_evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    label_score_semantics: str = "unknown"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "audio": self.audio_path,
            "model_id": self.model_id,
            "duration_sec": self.duration_sec,
            "top_labels": self.top_labels,
            "soundradar_events": self.soundradar_events,
            "active_events": self.active_events,
            "event_label_evidence": self.event_label_evidence,
            "label_score_semantics": self.label_score_semantics,
        }


@dataclass(frozen=True)
class AstBatchProfile:
    prepare_ms: float
    feature_ms: float
    to_device_ms: float
    model_ms: float
    postprocess_ms: float
    total_ms: float

    def to_jsonable(self) -> dict[str, float]:
        return {
            "prepare_ms": round(self.prepare_ms, 3),
            "feature_ms": round(self.feature_ms, 3),
            "to_device_ms": round(self.to_device_ms, 3),
            "model_ms": round(self.model_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


class AstAudioSetTeacher:
    """Lazy wrapper around Hugging Face AST AudioSet inference."""

    def __init__(
        self,
        model_id: str = AST_MODEL_ID,
        *,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str | None = None,
        compile_model=False,
    ) -> None:
        self.model_id = model_id
        self.torch, self.feature_extractor, self.model, self.dtype = self._load_model(
            model_id,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            compile_model=compile_model,
        )
        self.device = next(self.model.parameters()).device
        self._labels = self._model_labels()
        self._event_label_indices = build_audioset_event_indices(self._labels)
        self._background_label_indices = build_audioset_background_indices(self._labels)

    @staticmethod
    def _load_model(
        model_id: str,
        *,
        device: str,
        dtype: str,
        attn_implementation: str | None = None,
        compile_model=False,
    ):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:  # pragma: no cover - exercised in real env, not unit tests
            raise RuntimeError(
                "AST teacher requires optional dependencies. Install them with: "
                ".venv/bin/python -m pip install torch transformers huggingface_hub"
            ) from exc

        resolved_device = _resolve_torch_device(torch, device)
        configure_torch_runtime_for_device(torch, resolved_device)
        resolved_dtype = _resolve_torch_dtype(torch, dtype, resolved_device)
        feature_extractor = _load_ast_feature_extractor(
            model_id,
            from_pretrained=AutoFeatureExtractor.from_pretrained,
        )
        model_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
        model = _from_pretrained_prefer_cache(AutoModelForAudioClassification.from_pretrained, model_id, **model_kwargs)
        model.to(device=resolved_device, dtype=resolved_dtype)
        model.eval()
        model = compile_torch_model_if_requested(torch, model, compile_model)
        return torch, feature_extractor, model, resolved_dtype

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_path: str = "<waveform>",
    ) -> AstPrediction:
        return self.predict_waveforms([waveform], sample_rate, top_k=top_k, audio_paths=[audio_path])[0]

    def predict_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_paths: list[str] | None = None,
    ) -> list[AstPrediction]:
        predictions, _ = self.profile_predict_waveforms(
            waveforms,
            sample_rate,
            top_k=top_k,
            audio_paths=audio_paths,
        )
        return predictions

    def profile_predict_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_paths: list[str] | None = None,
        timer=None,
        synchronize: bool = False,
    ) -> tuple[list[AstPrediction], AstBatchProfile]:
        """Predict a waveform batch and return stage timings for benchmarking."""

        timer = timer or time.perf_counter
        t0 = timer()
        prepared = []
        for waveform in waveforms:
            mono = np.asarray(waveform, dtype=np.float32)
            if mono.ndim != 1:
                raise ValueError("AST teacher expects mono waveforms for batched prediction")
            if sample_rate != AST_SAMPLE_RATE:
                mono = resample_linear(mono[:, None], sample_rate, AST_SAMPLE_RATE)[:, 0]
            prepared.append(mono.astype(np.float32, copy=False))
        t1 = timer()

        if not prepared:
            profile = AstBatchProfile(0.0, 0.0, 0.0, 0.0, 0.0, max(0.0, (t1 - t0) * 1000.0))
            return [], profile
        if audio_paths is None:
            audio_paths = ["<waveform>"] * len(prepared)
        if len(audio_paths) != len(prepared):
            raise ValueError("audio_paths length must match waveforms length")

        model_sample_rate = AST_SAMPLE_RATE if sample_rate != AST_SAMPLE_RATE else sample_rate
        inputs = self.feature_extractor(
            prepared,
            sampling_rate=model_sample_rate,
            return_tensors="pt",
        )
        t2 = timer()
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }
        if synchronize:
            synchronize_torch_device(self.torch, self.device)
        t3 = timer()
        with _torch_inference_context(self.torch):
            logits = self.model(**inputs).logits
            probabilities = self.torch.sigmoid(logits)
        if synchronize:
            synchronize_torch_device(self.torch, self.device)
        t4 = timer()
        probabilities_batch = np.atleast_2d(probabilities.detach().cpu().numpy())
        predictions = [
            self._prediction_from_probabilities(
                probabilities,
                duration_sec=float(len(waveform) / model_sample_rate),
                top_k=top_k,
                audio_path=audio_path,
            )
            for probabilities, waveform, audio_path in zip(probabilities_batch, prepared, audio_paths)
        ]
        t5 = timer()
        return predictions, AstBatchProfile(
            prepare_ms=(t1 - t0) * 1000.0,
            feature_ms=(t2 - t1) * 1000.0,
            to_device_ms=(t3 - t2) * 1000.0,
            model_ms=(t4 - t3) * 1000.0,
            postprocess_ms=(t5 - t4) * 1000.0,
            total_ms=(t5 - t0) * 1000.0,
        )

    def _prediction_from_probabilities(
        self,
        probabilities: np.ndarray,
        *,
        duration_sec: float,
        top_k: int,
        audio_path: str,
    ) -> AstPrediction:
        labels = getattr(self, "_labels", ()) or self._model_labels()
        event_indices = getattr(self, "_event_label_indices", None)
        background_indices = getattr(self, "_background_label_indices", None)
        top_labels = top_k_label_scores(probabilities, labels, top_k)
        mapped = map_audioset_probabilities_to_events(
            probabilities,
            labels,
            event_indices=event_indices,
            background_indices=background_indices,
        )
        return AstPrediction(
            audio_path=audio_path,
            model_id=self.model_id,
            duration_sec=duration_sec,
            top_labels=top_labels,
            soundradar_events=mapped,
            active_events=active_soundradar_events(mapped),
            event_label_evidence=event_label_evidence_from_array(probabilities, labels),
            label_score_semantics="sigmoid_probability",
        )

    def predict_file(self, path: str | Path, *, top_k: int = 12) -> AstPrediction:
        path = Path(path)
        waveform, sample_rate = load_audio_mono(path, target_rate=AST_SAMPLE_RATE)
        return self.predict_waveform(waveform, sample_rate, top_k=top_k, audio_path=str(path))

    def warmup_direction_batch(
        self,
        *,
        sample_rate: int = AST_SAMPLE_RATE,
        seconds: float = 1.0,
        direction_count: int = 7,
    ) -> None:
        """Run one fixed-shape zero batch to pay backend/kernel warmup before live audio."""

        sample_count = max(1, int(round(float(sample_rate) * float(seconds))))
        waveforms = [np.zeros(sample_count, dtype=np.float32) for _ in range(int(direction_count))]
        audio_paths = [f"warmup_{index}" for index in range(int(direction_count))]
        self.predict_waveforms(waveforms, sample_rate, top_k=1, audio_paths=audio_paths)

    def _model_labels(self) -> tuple[str, ...]:
        id2label = getattr(self.model.config, "id2label", {})
        numeric_keys = []
        for key in id2label:
            try:
                numeric_keys.append(int(key))
            except (TypeError, ValueError):
                continue
        if not numeric_keys:
            return ()
        max_index = max(numeric_keys)
        return tuple(str(id2label.get(index, id2label.get(str(index), index))) for index in range(max_index + 1))

    def _label_for_index(self, index: int) -> str:
        return _label_from_sequence(getattr(self, "_labels", ()), index)


class EfficientATAudioSetTeacher:
    """Wrapper for fschmid56/EfficientAT MobileNet AudioSet teachers.

    EfficientAT is not a Transformers model, so it is loaded from the upstream
    repository code.  Use ``SOUNDRADAR_EFFICIENTAT_REPO=/path/to/EfficientAT`` to
    point at an existing clone; otherwise the wrapper lazily clones the repo into
    ``~/.cache/soundradar/EfficientAT`` when first used.
    """

    def __init__(
        self,
        model_id: str = EFFICIENTAT_DEFAULT_MODEL,
        *,
        device: str = "auto",
        dtype: str = "auto",
        compile_model=False,
    ) -> None:
        self.model_id = model_id
        self.torch, self.mel, self.model, self.dtype, self._labels = self._load_model(
            model_id,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
        )
        self.device = next(self.model.parameters()).device
        self._event_label_indices = build_audioset_event_indices(self._labels)
        self._background_label_indices = build_audioset_background_indices(self._labels)

    @staticmethod
    def _load_model(
        model_id: str,
        *,
        device: str,
        dtype: str,
        compile_model=False,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised in real envs
            raise RuntimeError(
                "EfficientAT teacher requires PyTorch. Install optional ML deps with: "
                ".venv/bin/python -m pip install torch torchvision torchaudio"
            ) from exc

        resolved_device = _resolve_torch_device(torch, device)
        configure_torch_runtime_for_device(torch, resolved_device)
        resolved_dtype = _resolve_torch_dtype(torch, dtype, resolved_device)
        try:
            efficientat_repo, get_mobilenet, augment_mel_stft, name_to_width, labels = _load_efficientat_components()
        except ImportError as exc:  # pragma: no cover - depends on optional upstream deps
            raise RuntimeError(
                "EfficientAT teacher requires the upstream EfficientAT inference dependencies. "
                "Install at least torchvision and torchaudio in the repo venv, e.g. "
                ".venv/bin/python -m pip install torchvision torchaudio, then retry."
            ) from exc

        with _temporary_sys_path_and_cwd(efficientat_repo), redirect_stdout(io.StringIO()):
            model = get_mobilenet(width_mult=name_to_width(model_id), pretrained_name=model_id)
            mel = augment_mel_stft(
                n_mels=128,
                sr=EFFICIENTAT_SAMPLE_RATE,
                win_length=800,
                hopsize=320,
                freqm=0,
                timem=0,
            )
        model.to(device=resolved_device)
        mel.to(device=resolved_device)
        model.eval()
        mel.eval()
        model = compile_torch_model_if_requested(torch, model, compile_model)
        return torch, mel, model, resolved_dtype, tuple(str(label) for label in labels)

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_path: str = "<waveform>",
    ) -> AstPrediction:
        return self.predict_waveforms([waveform], sample_rate, top_k=top_k, audio_paths=[audio_path])[0]

    def predict_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_paths: list[str] | None = None,
    ) -> list[AstPrediction]:
        predictions, _ = self.profile_predict_waveforms(
            waveforms,
            sample_rate,
            top_k=top_k,
            audio_paths=audio_paths,
        )
        return predictions

    def profile_predict_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
        *,
        top_k: int = 12,
        audio_paths: list[str] | None = None,
        timer=None,
        synchronize: bool = False,
    ) -> tuple[list[AstPrediction], AstBatchProfile]:
        """Predict a waveform batch and return stage timings for benchmarking."""

        timer = timer or time.perf_counter
        t0 = timer()
        prepared: list[np.ndarray] = []
        durations: list[float] = []
        for waveform in waveforms:
            mono = np.asarray(waveform, dtype=np.float32)
            if mono.ndim != 1:
                raise ValueError("EfficientAT teacher expects mono waveforms for batched prediction")
            if sample_rate != EFFICIENTAT_SAMPLE_RATE:
                mono = resample_linear(mono[:, None], sample_rate, EFFICIENTAT_SAMPLE_RATE)[:, 0]
            mono = mono.astype(np.float32, copy=False)
            prepared.append(mono)
            durations.append(float(len(mono) / EFFICIENTAT_SAMPLE_RATE))
        t1 = timer()

        if not prepared:
            profile = AstBatchProfile(0.0, 0.0, 0.0, 0.0, 0.0, max(0.0, (t1 - t0) * 1000.0))
            return [], profile
        if audio_paths is None:
            audio_paths = ["<waveform>"] * len(prepared)
        if len(audio_paths) != len(prepared):
            raise ValueError("audio_paths length must match waveforms length")

        max_samples = max(len(waveform) for waveform in prepared)
        batch = np.zeros((len(prepared), max_samples), dtype=np.float32)
        for index, waveform in enumerate(prepared):
            batch[index, : len(waveform)] = waveform

        waveform_tensor = self.torch.from_numpy(batch).to(self.device)
        if synchronize:
            synchronize_torch_device(self.torch, self.device)
        t2 = timer()
        with _torch_inference_context(self.torch), _torch_autocast_context(self.torch, self.device, self.dtype):
            spec = self.mel(waveform_tensor)
        if synchronize:
            synchronize_torch_device(self.torch, self.device)
        t3 = timer()
        with _torch_inference_context(self.torch), _torch_autocast_context(self.torch, self.device, self.dtype):
            outputs = self.model(spec.unsqueeze(1))
            logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "logits", outputs)
            logits = logits.float()
        if synchronize:
            synchronize_torch_device(self.torch, self.device)
        t4 = timer()

        logits_batch = np.atleast_2d(logits.detach().cpu().numpy())
        relative_scores_batch = np.atleast_2d(relative_logit_scores(logits_batch))
        predictions = [
            self._prediction_from_logits(
                logits,
                relative_scores,
                duration_sec=duration_sec,
                top_k=top_k,
                audio_path=audio_path,
            )
            for logits, relative_scores, duration_sec, audio_path in zip(logits_batch, relative_scores_batch, durations, audio_paths)
        ]
        t5 = timer()
        return predictions, AstBatchProfile(
            prepare_ms=(t1 - t0) * 1000.0,
            feature_ms=(t3 - t2) * 1000.0,
            to_device_ms=(t2 - t1) * 1000.0,
            model_ms=(t4 - t3) * 1000.0,
            postprocess_ms=(t5 - t4) * 1000.0,
            total_ms=(t5 - t0) * 1000.0,
        )

    def _prediction_from_logits(
        self,
        logits: np.ndarray,
        relative_scores: np.ndarray,
        *,
        duration_sec: float,
        top_k: int,
        audio_path: str,
    ) -> AstPrediction:
        labels = getattr(self, "_labels", ())
        top_labels = top_k_label_scores(relative_scores, labels, top_k)
        mapped = map_audioset_logits_to_events(
            logits,
            labels,
            event_indices=getattr(self, "_event_label_indices", None),
            background_indices=getattr(self, "_background_label_indices", None),
        )
        return AstPrediction(
            audio_path=audio_path,
            model_id=self.model_id,
            duration_sec=duration_sec,
            top_labels=top_labels,
            soundradar_events=mapped,
            active_events=active_soundradar_events(mapped),
            event_label_evidence=event_label_evidence_from_array(relative_scores, labels),
            label_score_semantics="relative_logit_evidence",
        )

    def _prediction_from_probabilities(
        self,
        probabilities: np.ndarray,
        *,
        duration_sec: float,
        top_k: int,
        audio_path: str,
    ) -> AstPrediction:
        labels = getattr(self, "_labels", ())
        top_labels = top_k_label_scores(probabilities, labels, top_k)
        mapped = map_audioset_probabilities_to_events(
            probabilities,
            labels,
            event_indices=getattr(self, "_event_label_indices", None),
            background_indices=getattr(self, "_background_label_indices", None),
        )
        return AstPrediction(
            audio_path=audio_path,
            model_id=self.model_id,
            duration_sec=duration_sec,
            top_labels=top_labels,
            soundradar_events=mapped,
            active_events=active_soundradar_events(mapped),
            event_label_evidence=event_label_evidence_from_array(probabilities, labels),
            label_score_semantics="probability",
        )

    def predict_file(self, path: str | Path, *, top_k: int = 12) -> AstPrediction:
        path = Path(path)
        waveform, sample_rate = load_audio_mono(path, target_rate=EFFICIENTAT_SAMPLE_RATE)
        return self.predict_waveform(waveform, sample_rate, top_k=top_k, audio_path=str(path))

    def warmup_direction_batch(
        self,
        *,
        sample_rate: int = EFFICIENTAT_SAMPLE_RATE,
        seconds: float = 1.0,
        direction_count: int = 7,
    ) -> None:
        """Run one fixed-shape zero batch to pay backend/kernel warmup before live audio."""

        sample_count = max(1, int(round(float(sample_rate) * float(seconds))))
        waveforms = [np.zeros(sample_count, dtype=np.float32) for _ in range(int(direction_count))]
        audio_paths = [f"warmup_{index}" for index in range(int(direction_count))]
        self.predict_waveforms(waveforms, sample_rate, top_k=1, audio_paths=audio_paths)


def _torch_autocast_context(torch_module, device, dtype):
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    autocast = getattr(torch_module, "autocast", None)
    float32 = getattr(torch_module, "float32", None)
    if device_type == "cuda" and callable(autocast) and dtype is not float32:
        return autocast(device_type=device_type, dtype=dtype)
    return nullcontext()


def _load_efficientat_components():
    repo = _resolve_efficientat_repo()
    with _temporary_sys_path_and_cwd(repo):
        from helpers.utils import NAME_TO_WIDTH, labels
        from models.mn.model import get_model as get_mobilenet
        from models.preprocess import AugmentMelSTFT

    return repo, get_mobilenet, AugmentMelSTFT, NAME_TO_WIDTH, tuple(labels)


def _resolve_efficientat_repo(repo_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if repo_path is not None:
        candidates.append(Path(repo_path).expanduser())
    for env_name in ("SOUNDRADAR_EFFICIENTAT_REPO", "EFFICIENTAT_REPO"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.append(Path.home() / ".cache" / "soundradar" / "EfficientAT")

    for candidate in candidates:
        if _looks_like_efficientat_repo(candidate):
            return candidate

    target = candidates[-1]
    _clone_efficientat_repo(target)
    if _looks_like_efficientat_repo(target):
        return target
    raise RuntimeError(f"EfficientAT repo was not available at {target}")


def _looks_like_efficientat_repo(path: Path) -> bool:
    return (path / "models" / "mn" / "model.py").exists() and (path / "metadata" / "class_labels_indices.csv").exists()


def _clone_efficientat_repo(target: Path) -> None:
    if target.exists():
        return
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "EfficientAT repo is required but git is not available. Clone "
            f"{EFFICIENTAT_REPO_URL} and set SOUNDRADAR_EFFICIENTAT_REPO."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [git, "clone", "--depth", "1", EFFICIENTAT_REPO_URL, str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to clone EfficientAT repo. Clone it manually from "
            f"{EFFICIENTAT_REPO_URL} and set SOUNDRADAR_EFFICIENTAT_REPO. "
            f"git said: {completed.stderr.strip() or completed.stdout.strip()}"
        )


@contextmanager
def _temporary_sys_path_and_cwd(path: Path):
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    active_site_paths = [
        site_path
        for site_path in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib"))
        if site_path
    ]
    sys.path[:0] = [str(path), *active_site_paths]
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def _resolve_torch_device(torch_module, requested: str):
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = cuda is not None and cuda.is_available()
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = mps_backend is not None and mps_backend.is_available()
    if requested == "auto":
        if cuda_available:
            return torch_module.device("cuda")
        if mps_available:
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("Requested CUDA device, but torch.cuda is not available")
    if requested == "mps" and not mps_available:
        raise RuntimeError("Requested MPS device, but torch.backends.mps is not available")
    return torch_module.device(requested)


def _resolve_torch_dtype(torch_module, requested: str, resolved_device):
    if requested == "auto":
        if getattr(resolved_device, "type", str(resolved_device)) in {"cuda", "mps"}:
            return torch_module.float16
        return torch_module.float32

    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return dtype_map[requested]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype {requested!r}") from exc


def _load_ast_feature_extractor(model_id: str, *, from_pretrained=None):
    if from_pretrained is None:  # pragma: no cover - exercised through _load_model in real envs
        from transformers import AutoFeatureExtractor

        from_pretrained = AutoFeatureExtractor.from_pretrained

    # The MIT AST feature extractor currently emits this Transformers warning
    # with its default 128 mel bins over 257 FFT bins. The pretrained model was
    # built with that setup, so changing it would break model compatibility.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=AST_MEL_FILTER_WARNING_PATTERN,
            category=UserWarning,
        )
        return _from_pretrained_prefer_cache(from_pretrained, model_id)


def _from_pretrained_prefer_cache(from_pretrained, model_id: str, **kwargs):
    try:
        return from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return from_pretrained(model_id, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AudioSet teacher inference on an audio file")
    parser.add_argument("audio", type=Path, help="WAV/MP3 audio file")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL, choices=TEACHER_MODEL_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Torch model/input dtype")
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation, e.g. sdpa")
    parser.add_argument("--compile-model", action="store_true", help="Wrap the teacher model with torch.compile")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode when --compile-model is set")
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    teacher = create_audio_event_teacher(
        args.teacher_model,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        compile_model=args.compile_mode if args.compile_model else False,
    )
    prediction = teacher.predict_file(args.audio, top_k=args.top_k)
    print(json.dumps(prediction.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
