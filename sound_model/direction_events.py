"""Direction-wise event scoring without additional training.

This is the Method-B prototype: split available audio into seven coarse
speaker/direction waveforms, run the same teacher classifier for those
directions, and return a ``direction x event`` score matrix.  Batch-capable
teachers are called once with all seven waveforms.  It intentionally does not
train or fine-tune anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Any, Protocol, cast

import numpy as np

from .ast_teacher import AST_MODEL_ID, AstAudioSetTeacher, load_audio_channels
from .audio_features import DEFAULT_CLASSES

DIRECTION_CHANNELS: tuple[tuple[str, int], ...] = (
    ("front_left", 0),
    ("front", 2),
    ("front_right", 1),
    ("left", 4),
    ("right", 5),
    ("rear_left", 6),
    ("rear_right", 7),
)
DIRECTION_NAMES: tuple[str, ...] = tuple(direction for direction, _ in DIRECTION_CHANNELS)


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


def extract_direction_waveforms(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Return seven direction waveforms from 7.1, stereo, or mono audio.

    7.1 follows the Audio MIDI / SoundRadar order: FL=0, FR=1, C=2, LFE=3
    ignored, side-left=4, side-right=5, rear-left=6, rear-right=7.
    Stereo/mono fallback preserves API shape but is less directional.
    """

    channels = _as_samples_by_channels(audio)
    channel_count = channels.shape[1]

    if channel_count >= 8:
        return {direction: channels[:, index].copy() for direction, index in DIRECTION_CHANNELS}

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
) -> DirectionEventPrediction:
    """Collect direction-event scores, batching teacher inference when supported."""

    waveforms = extract_direction_waveforms(audio)
    directions = list(waveforms)
    direction_waveforms = [waveforms[direction] for direction in directions]
    direction_scores: dict[str, dict[str, float]] = {}
    active_by_direction: dict[str, list[str]] = {}
    top_labels_by_direction: dict[str, list[dict[str, float]]] = {}

    predict_waveforms = getattr(teacher, "predict_waveforms", None)
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

    for direction, prediction in zip(directions, predictions):
        direction_scores[direction] = dict(prediction.soundradar_events)
        active_by_direction[direction] = list(prediction.active_events)
        top_labels_by_direction[direction] = list(prediction.top_labels)

    return DirectionEventPrediction(
        sample_rate=sample_rate,
        direction_event_scores=direction_scores,
        active_events_by_direction=active_by_direction,
        top_labels_by_direction=top_labels_by_direction,
        source_path=source_path,
    )


def predict_direction_events_file(
    path: str | Path,
    *,
    model_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    top_k: int = 8,
) -> DirectionEventPrediction:
    path = Path(path)
    audio, sample_rate = load_audio_channels(path)
    teacher = AstAudioSetTeacher(model_id or AST_MODEL_ID, device=device, dtype=dtype)
    return score_direction_events(audio, sample_rate, teacher, top_k=top_k, source_path=str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AST teacher once per coarse direction and emit direction x event scores")
    parser.add_argument("audio", type=Path, help="WAV/MP3 audio file; 8ch WAV gives true 7-direction input")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction = predict_direction_events_file(
        args.audio,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        top_k=args.top_k,
    )
    print(json.dumps(prediction.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
