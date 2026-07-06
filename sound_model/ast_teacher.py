"""AST AudioSet teacher inference for SoundRadar event bootstrapping.

The model requested for the first teacher pass is
``MIT/ast-finetuned-audioset-10-10-0.4593``.  This module keeps the heavy
Hugging Face imports lazy so the rest of SoundRadar and the unit tests still run
when optional ML dependencies are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from typing import Mapping

import numpy as np

from .audio_features import DEFAULT_CLASSES, read_wav, resample_linear

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16_000
AST_MEL_FILTER_WARNING_PATTERN = (
    r"At least one mel filter has all zero values\..*"
    r"num_mel_filters.*num_frequency_bins.*"
)

# Keep these conservative: they map AudioSet labels into the coarse events used
# by the accessibility overlay, not into exact PUBG semantic classes.
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
    "gunshot": (
        "gunshot",
        "gunfire",
        "firearm",
        "machine gun",
        "fusillade",
        "artillery fire",
        "cap gun",
    ),
    "vehicle": (
        "vehicle",
        "motor vehicle",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "engine",
        "aircraft",
        "helicopter",
        "boat",
        "ship",
        "train",
    ),
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


def _clamp_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _label_matches(label: str, keywords: tuple[str, ...]) -> bool:
    normalized = label.lower()
    return any(keyword in normalized for keyword in keywords)


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
    top_labels: list[dict[str, float]]
    soundradar_events: dict[str, float]
    active_events: list[str]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "audio": self.audio_path,
            "model_id": self.model_id,
            "duration_sec": self.duration_sec,
            "top_labels": self.top_labels,
            "soundradar_events": self.soundradar_events,
            "active_events": self.active_events,
        }


class AstAudioSetTeacher:
    """Lazy wrapper around Hugging Face AST AudioSet inference."""

    def __init__(self, model_id: str = AST_MODEL_ID, *, device: str = "auto", dtype: str = "auto") -> None:
        self.model_id = model_id
        self.torch, self.feature_extractor, self.model, self.dtype = self._load_model(model_id, device=device, dtype=dtype)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _load_model(model_id: str, *, device: str, dtype: str):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:  # pragma: no cover - exercised in real env, not unit tests
            raise RuntimeError(
                "AST teacher requires optional dependencies. Install them with: "
                ".venv/bin/python -m pip install torch transformers huggingface_hub"
            ) from exc

        resolved_device = _resolve_torch_device(torch, device)
        resolved_dtype = _resolve_torch_dtype(torch, dtype, resolved_device)
        feature_extractor = _load_ast_feature_extractor(
            model_id,
            from_pretrained=AutoFeatureExtractor.from_pretrained,
        )
        model = _from_pretrained_prefer_cache(AutoModelForAudioClassification.from_pretrained, model_id)
        model.to(device=resolved_device, dtype=resolved_dtype)
        model.eval()
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
        prepared = []
        for waveform in waveforms:
            mono = np.asarray(waveform, dtype=np.float32)
            if mono.ndim != 1:
                raise ValueError("AST teacher expects mono waveforms for batched prediction")
            if sample_rate != AST_SAMPLE_RATE:
                mono = resample_linear(mono[:, None], sample_rate, AST_SAMPLE_RATE)[:, 0]
            prepared.append(mono.astype(np.float32, copy=False))

        if not prepared:
            return []
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
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities_batch = self.torch.sigmoid(logits).detach().cpu().numpy()

        probabilities_batch = np.atleast_2d(probabilities_batch)
        return [
            self._prediction_from_probabilities(
                probabilities,
                duration_sec=float(len(waveform) / model_sample_rate),
                top_k=top_k,
                audio_path=audio_path,
            )
            for probabilities, waveform, audio_path in zip(probabilities_batch, prepared, audio_paths)
        ]

    def _prediction_from_probabilities(
        self,
        probabilities: np.ndarray,
        *,
        duration_sec: float,
        top_k: int,
        audio_path: str,
    ) -> AstPrediction:
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        label_scores = {
            self._label_for_index(int(index)): float(probabilities[int(index)])
            for index in range(len(probabilities))
        }
        top_labels = [
            {"label": self._label_for_index(int(index)), "score": float(probabilities[int(index)])}
            for index in top_indices
        ]
        mapped = map_audioset_scores_to_events(label_scores)
        return AstPrediction(
            audio_path=audio_path,
            model_id=self.model_id,
            duration_sec=duration_sec,
            top_labels=top_labels,
            soundradar_events=mapped,
            active_events=active_soundradar_events(mapped),
        )

    def predict_file(self, path: str | Path, *, top_k: int = 12) -> AstPrediction:
        path = Path(path)
        waveform, sample_rate = load_audio_mono(path, target_rate=AST_SAMPLE_RATE)
        return self.predict_waveform(waveform, sample_rate, top_k=top_k, audio_path=str(path))

    def _label_for_index(self, index: int) -> str:
        id2label = getattr(self.model.config, "id2label", {})
        return str(id2label.get(index, id2label.get(str(index), index)))


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
    parser = argparse.ArgumentParser(description="Run MIT AST AudioSet teacher inference on an audio file")
    parser.add_argument("audio", type=Path, help="WAV/MP3 audio file")
    parser.add_argument("--model-id", default=AST_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Torch model/input dtype")
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    teacher = AstAudioSetTeacher(args.model_id, device=args.device, dtype=args.dtype)
    prediction = teacher.predict_file(args.audio, top_k=args.top_k)
    print(json.dumps(prediction.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
