"""Direction-wise event scoring without additional training.

This is the Method-B prototype: split available audio into seven coarse
speaker/direction waveforms, run the same teacher classifier for those
directions, and return a ``direction x event`` score matrix.  Batch-capable
teachers are called once with all seven waveforms.  It intentionally does not
train or fine-tune anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import json
import time
from typing import Any, Mapping, Protocol, cast

import numpy as np

from .ast_teacher import (
    DEFAULT_TEACHER_MODEL,
    TEACHER_MODEL_CHOICES,
    active_soundradar_events,
    create_audio_event_teacher,
    event_label_evidence_from_scores,
    load_audio_channels,
)
from .audio_features import DEFAULT_CLASSES
from .vehicle_gun_resolver import (
    VehicleGunDecision,
    VehicleGunEvidence,
    apply_vehicle_gun_decision,
    resolve_vehicle_gun,
    strong_road_vehicle_evidence,
)
from .vehicle_gun_features import extract_vehicle_gun_acoustic_features

DIRECTION_CHANNELS: tuple[tuple[str, int], ...] = (
    ("front_left", 0),
    ("front", 2),
    ("front_right", 1),
    ("left", 4),
    ("right", 5),
    ("rear_left", 6),
    ("rear_right", 7),
)
SOUNDRADAR_CHANNEL_KEYS: dict[str, str] = {
    "front_left": "avg",
    "front": "c",
    "front_right": "avd",
    "left": "g",
    "right": "d",
    "rear_left": "arg",
    "rear_right": "ard",
}
DIRECTION_NAMES: tuple[str, ...] = tuple(direction for direction, _ in DIRECTION_CHANNELS)
DIRECTION_SILENCE_PEAK = 1.0e-4


def background_event_scores() -> dict[str, float]:
    return {event_name: 1.0 if event_name == "background" else 0.0 for event_name in DEFAULT_CLASSES}


def direction_waveform_has_signal(waveform: np.ndarray, *, min_peak: float = DIRECTION_SILENCE_PEAK) -> bool:
    if waveform.size == 0:
        return False
    return float(np.nanmax(np.abs(waveform))) >= float(min_peak)


def _linear_score(value: float, start: float, full: float) -> float:
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


def transient_gunshot_score(waveform: np.ndarray, sample_rate: int) -> float:
    """Return a conservative score for short, loud impulses.

    EfficientAT can label sharp in-game gunshots as mechanical/vehicle-like
    sounds (for example train-wheel squeal).  This feature gate only triggers
    for waveforms whose energy is concentrated in a short burst, so sustained
    engine-like audio keeps its model label.
    """

    samples = np.asarray(waveform, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    abs_samples = np.abs(samples)
    peak = float(np.nanmax(abs_samples))
    if not np.isfinite(peak) or peak < 0.04:
        return 0.0
    rms = float(np.sqrt(np.nanmean(samples * samples)))
    if not np.isfinite(rms) or rms <= 1.0e-9:
        return 0.0

    crest = peak / rms
    window = max(1, min(samples.size, int(sample_rate * 0.05)))
    energy = samples * samples
    if window >= samples.size:
        short_rms = rms
    else:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        short_rms = float(np.sqrt(np.nanmax(np.convolve(energy, kernel, mode="valid"))))
    concentration = short_rms / (rms + 1.0e-9)

    peak_score = _linear_score(peak, 0.04, 0.35)
    crest_score = _linear_score(crest, 8.0, 24.0)
    concentration_score = _linear_score(concentration, 1.8, 3.5)
    if crest_score <= 0.0 or concentration_score <= 0.0:
        return 0.0
    return float(min(1.0, 0.4 * peak_score + 0.3 * crest_score + 0.3 * concentration_score))


def _max_label_evidence(event_label_evidence, event_name: str) -> float:
    scores = (event_label_evidence or {}).get(event_name, {}) or {}
    if isinstance(scores, Mapping):
        return max((float(score) for score in scores.values()), default=0.0)
    return 0.0


def _top_label_evidence(top_labels) -> dict[str, dict[str, float]]:
    label_scores = {}
    for item in top_labels or ():
        if not isinstance(item, Mapping):
            continue
        label_scores[str(item.get("label", ""))] = float(item.get("score", 0.0) or 0.0)
    return event_label_evidence_from_scores(label_scores)


def top_labels_include_road_vehicle(top_labels) -> bool:
    """Compatibility helper using the resolver's score-and-margin vehicle gate."""

    evidence = _top_label_evidence(top_labels)
    return strong_road_vehicle_evidence(
        _max_label_evidence(evidence, "vehicle"),
        _max_label_evidence(evidence, "gunshot"),
    )


def waveform_vehicle_gun_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    samples = np.asarray(waveform, dtype=np.float32)
    acoustic = asdict(extract_vehicle_gun_acoustic_features(samples, sample_rate))
    if samples.size == 0:
        return {
            "transient_score": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "crest_factor": 0.0,
            **acoustic,
        }
    peak = float(np.nanmax(np.abs(samples)))
    rms = float(np.sqrt(np.nanmean(samples * samples)))
    if not np.isfinite(peak):
        peak = 0.0
    if not np.isfinite(rms) or rms <= 1.0e-9:
        rms = 0.0
    crest_factor = peak / rms if rms > 0.0 else 0.0
    return {
        "transient_score": transient_gunshot_score(samples, sample_rate),
        "peak": peak,
        "rms": rms,
        "crest_factor": crest_factor,
        **acoustic,
    }


def resolve_waveform_event_scores(
    scores: dict[str, float],
    waveform: np.ndarray,
    sample_rate: int,
    *,
    event_label_evidence=None,
    top_labels=None,
) -> tuple[dict[str, float], VehicleGunEvidence, VehicleGunDecision]:
    """Apply the resolver and return adjusted scores plus its audit trail."""

    raw_scores = dict(scores)
    label_evidence = event_label_evidence or _top_label_evidence(top_labels)
    waveform_features = waveform_vehicle_gun_features(waveform, sample_rate)
    evidence = VehicleGunEvidence(
        gunshot_teacher_score=float(raw_scores.get("gunshot", 0.0)),
        vehicle_teacher_score=float(raw_scores.get("vehicle", 0.0)),
        gunshot_label_score=_max_label_evidence(label_evidence, "gunshot"),
        road_vehicle_label_score=_max_label_evidence(label_evidence, "vehicle"),
        **waveform_features,
    )

    if float(raw_scores.get("explosion", 0.0)) >= 0.25:
        decision = VehicleGunDecision(
            label="unknown",
            confidence=1.0,
            gunshot_evidence=float(raw_scores.get("gunshot", 0.0)),
            vehicle_evidence=float(raw_scores.get("vehicle", 0.0)),
            reason="resolver skipped; explosion evidence is active",
        )
        return raw_scores, evidence, decision

    decision = resolve_vehicle_gun(evidence)
    return apply_vehicle_gun_decision(raw_scores, evidence, decision), evidence, decision


def apply_waveform_event_heuristics(
    scores: dict[str, float],
    waveform: np.ndarray,
    sample_rate: int,
    top_labels=None,
) -> dict[str, float]:
    adjusted, _, _ = resolve_waveform_event_scores(
        scores,
        waveform,
        sample_rate,
        top_labels=top_labels,
    )
    return adjusted


class DirectionTeacher(Protocol):
    def predict_waveform(self, waveform, sample_rate: int, *, top_k: int = 12, audio_path: str = "<waveform>") -> Any:
        ...


@dataclass
class DirectionEventPrediction:
    sample_rate: int
    direction_event_scores: dict[str, dict[str, float]]
    active_events_by_direction: dict[str, list[str]]
    top_labels_by_direction: dict[str, list[dict[str, float]]]
    source_path: str | None = None
    mode: str = "7-direction teacher inference"
    raw_direction_event_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    resolver_direction_event_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    temporal_direction_event_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    event_label_evidence_by_direction: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    vehicle_gun_evidence_by_direction: dict[str, VehicleGunEvidence] = field(default_factory=dict)
    vehicle_gun_decisions_by_direction: dict[str, VehicleGunDecision] = field(default_factory=dict)
    label_score_semantics_by_direction: dict[str, str] = field(default_factory=dict)
    inference_latency_ms: float | None = None

    def to_jsonable(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "mode": self.mode,
            "directions": list(DIRECTION_NAMES),
            "classes": list(DEFAULT_CLASSES),
            "direction_event_scores": self.direction_event_scores,
            "active_events_by_direction": self.active_events_by_direction,
            "top_labels_by_direction": self.top_labels_by_direction,
            "raw_direction_event_scores": self.raw_direction_event_scores,
            "resolver_direction_event_scores": self.resolver_direction_event_scores,
            "temporal_direction_event_scores": self.temporal_direction_event_scores,
            "event_label_evidence_by_direction": self.event_label_evidence_by_direction,
            "vehicle_gun_evidence_by_direction": {
                direction: asdict(evidence)
                for direction, evidence in self.vehicle_gun_evidence_by_direction.items()
            },
            "vehicle_gun_decisions_by_direction": {
                direction: asdict(decision)
                for direction, decision in self.vehicle_gun_decisions_by_direction.items()
            },
            "label_score_semantics_by_direction": self.label_score_semantics_by_direction,
            "inference_latency_ms": self.inference_latency_ms,
        }


def _as_samples_by_channels(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio[:, None]
    if audio.ndim != 2:
        raise ValueError("audio must be a 1D mono array or 2D samples x channels array")
    if audio.shape[1] == 0:
        raise ValueError("audio has no channels")
    return audio


def _mapped_channel(channels: np.ndarray, channel_map: Mapping[str, int | None], key: str) -> np.ndarray | None:
    index = channel_map.get(key)
    if index is None:
        return None
    index = int(index)
    if index < 0 or index >= channels.shape[1]:
        return None
    return channels[:, index]


def _average_or_zeros(channels: np.ndarray, waveforms: tuple[np.ndarray | None, ...]) -> np.ndarray:
    available = [waveform for waveform in waveforms if waveform is not None]
    if not available:
        return np.zeros(channels.shape[0], dtype=np.float32)
    return np.mean(np.stack(available, axis=1), axis=1).astype(np.float32, copy=False)


def extract_direction_waveforms(audio: np.ndarray, *, channel_map: Mapping[str, int | None] | None = None) -> dict[str, np.ndarray]:
    """Return seven direction waveforms from 7.1, stereo, mono, or SoundRadar mapping.

    7.1 follows the Audio MIDI / SoundRadar order: FL=0, FR=1, C=2, LFE=3
    ignored, side-left=4, side-right=5, rear-left=6, rear-right=7.
    Passing ``channel_map`` makes classifier directions follow the same mapping
    the live radar uses, including stereo fallback for 16ch virtual devices.
    """

    channels = _as_samples_by_channels(audio)
    channel_count = channels.shape[1]

    if channel_map is not None:
        front_left = _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["front_left"])
        front_right = _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["front_right"])
        front = _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["front"])
        if front is None:
            front = _average_or_zeros(channels, (front_left, front_right))
        mapped = {
            "front_left": front_left,
            "front": front,
            "front_right": front_right,
            "left": _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["left"]),
            "right": _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["right"]),
            "rear_left": _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["rear_left"]),
            "rear_right": _mapped_channel(channels, channel_map, SOUNDRADAR_CHANNEL_KEYS["rear_right"]),
        }
        return {
            direction: (waveform.copy() if waveform is not None else np.zeros(channels.shape[0], dtype=np.float32))
            for direction, waveform in mapped.items()
        }

    if channel_count >= 8:
        selected = channels[:, [index for _, index in DIRECTION_CHANNELS]]
        return {direction: selected[:, offset] for offset, (direction, _) in enumerate(DIRECTION_CHANNELS)}

    if channel_count >= 2:
        left = channels[:, 0]
        right = channels[:, 1]
        front = ((left + right) * 0.5).astype(np.float32, copy=False)
        return {
            "front_left": left.copy(),
            "front": front.copy(),
            "front_right": right.copy(),
            "left": left.copy(),
            "right": right.copy(),
            "rear_left": left.copy(),
            "rear_right": right.copy(),
        }

    mono = channels[:, 0]
    return {direction: mono.copy() for direction in DIRECTION_NAMES}


def score_direction_events(
    audio: np.ndarray,
    sample_rate: int,
    teacher: DirectionTeacher,
    *,
    top_k: int = 12,
    source_path: str | None = None,
    channel_map: Mapping[str, int | None] | None = None,
) -> DirectionEventPrediction:
    """Collect direction-event scores, batching teacher inference when supported."""

    waveforms = extract_direction_waveforms(audio, channel_map=channel_map)
    directions = list(waveforms)
    direction_waveforms = [waveforms[direction] for direction in directions]
    direction_scores: dict[str, dict[str, float]] = {}
    raw_direction_scores: dict[str, dict[str, float]] = {}
    active_by_direction: dict[str, list[str]] = {}
    top_labels_by_direction: dict[str, list[dict[str, float]]] = {}
    label_evidence_by_direction: dict[str, dict[str, dict[str, float]]] = {}
    resolver_evidence_by_direction: dict[str, VehicleGunEvidence] = {}
    resolver_decisions_by_direction: dict[str, VehicleGunDecision] = {}
    score_semantics_by_direction: dict[str, str] = {}

    predict_waveforms = getattr(teacher, "predict_waveforms", None)
    inference_started_at = time.perf_counter()
    if callable(predict_waveforms):
        batch_predict = cast(Any, predict_waveforms)
        predictions = list(
            batch_predict(
                direction_waveforms,
                sample_rate,
                top_k=top_k,
                audio_paths=directions,
            )
        )
        if len(predictions) != len(directions):
            raise RuntimeError(
                f"batched teacher returned {len(predictions)} predictions for {len(directions)} directions"
            )
    else:
        predictions = [
            teacher.predict_waveform(waveform, sample_rate, top_k=top_k, audio_path=direction)
            for direction, waveform in zip(directions, direction_waveforms)
        ]
    for direction, waveform, prediction in zip(directions, direction_waveforms, predictions):
        if not direction_waveform_has_signal(waveform):
            direction_scores[direction] = background_event_scores()
            raw_direction_scores[direction] = background_event_scores()
            active_by_direction[direction] = []
            top_labels_by_direction[direction] = []
            label_evidence_by_direction[direction] = {"gunshot": {}, "vehicle": {}}
            continue
        raw_scores = dict(prediction.soundradar_events)
        label_evidence = getattr(prediction, "event_label_evidence", None) or _top_label_evidence(
            getattr(prediction, "top_labels", ())
        )
        adjusted_scores, resolver_evidence, resolver_decision = resolve_waveform_event_scores(
            raw_scores,
            waveform,
            sample_rate,
            event_label_evidence=label_evidence,
            top_labels=getattr(prediction, "top_labels", ()),
        )
        raw_direction_scores[direction] = raw_scores
        direction_scores[direction] = adjusted_scores
        active_by_direction[direction] = active_soundradar_events(adjusted_scores)
        top_labels_by_direction[direction] = list(prediction.top_labels)
        label_evidence_by_direction[direction] = label_evidence
        resolver_evidence_by_direction[direction] = resolver_evidence
        resolver_decisions_by_direction[direction] = resolver_decision
        score_semantics_by_direction[direction] = str(getattr(prediction, "label_score_semantics", "unknown"))

    inference_latency_ms = max(0.0, (time.perf_counter() - inference_started_at) * 1000.0)
    return DirectionEventPrediction(
        sample_rate=sample_rate,
        direction_event_scores=direction_scores,
        active_events_by_direction=active_by_direction,
        top_labels_by_direction=top_labels_by_direction,
        source_path=source_path,
        raw_direction_event_scores=raw_direction_scores,
        resolver_direction_event_scores={
            direction: dict(scores) for direction, scores in direction_scores.items()
        },
        event_label_evidence_by_direction=label_evidence_by_direction,
        vehicle_gun_evidence_by_direction=resolver_evidence_by_direction,
        vehicle_gun_decisions_by_direction=resolver_decisions_by_direction,
        label_score_semantics_by_direction=score_semantics_by_direction,
        inference_latency_ms=inference_latency_ms,
    )


def predict_direction_events_file(
    path: str | Path,
    *,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    model_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    compile_model=False,
    top_k: int = 8,
) -> DirectionEventPrediction:
    path = Path(path)
    audio, sample_rate = load_audio_channels(path)
    teacher = create_audio_event_teacher(
        teacher_model,
        model_id=model_id,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        compile_model=compile_model,
    )
    return score_direction_events(audio, sample_rate, teacher, top_k=top_k, source_path=str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AudioSet teacher once per coarse direction and emit direction x event scores")
    parser.add_argument("audio", type=Path, help="WAV/MP3 audio file; 8ch WAV gives true 7-direction input")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL, choices=TEACHER_MODEL_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation, e.g. sdpa")
    parser.add_argument("--compile-model", action="store_true", help="Wrap the teacher model with torch.compile")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode when --compile-model is set")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction = predict_direction_events_file(
        args.audio,
        teacher_model=args.teacher_model,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        compile_model=args.compile_mode if args.compile_model else False,
        top_k=args.top_k,
    )
    print(json.dumps(prediction.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
