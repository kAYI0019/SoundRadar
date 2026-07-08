"""Benchmark full 7-direction AST teacher inference.

This CLI keeps the current full-direction design intact and reports where time is
spent: direction extraction, AST feature preparation, device transfer, model
forward, and postprocess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from .ast_teacher import DEFAULT_TEACHER_MODEL, TEACHER_MODEL_CHOICES, create_audio_event_teacher, load_audio_channels
from .direction_events import DIRECTION_NAMES, extract_direction_waveforms


def make_synthetic_audio(*, sample_rate: int, seconds: float, channel_count: int = 8, seed: int = 1234) -> np.ndarray:
    """Return deterministic full-scale-safe synthetic multichannel audio."""

    sample_count = max(1, int(round(int(sample_rate) * float(seconds))))
    rng = np.random.default_rng(seed)
    audio = rng.normal(0.0, 0.02, size=(sample_count, int(channel_count))).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def fit_audio_duration(audio: np.ndarray, sample_rate: int, seconds: float | None) -> np.ndarray:
    if seconds is None:
        return np.asarray(audio, dtype=np.float32)
    target_samples = max(1, int(round(int(sample_rate) * float(seconds))))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if len(audio) == target_samples:
        return audio
    if len(audio) > target_samples:
        return audio[:target_samples]
    padding = np.zeros((target_samples - len(audio), audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio, padding], axis=0)


def load_or_synthesize_audio(
    audio_path: str | Path | None,
    *,
    sample_rate: int,
    seconds: float,
    channel_count: int,
    seed: int,
) -> tuple[np.ndarray, int, str]:
    if audio_path is None:
        return (
            make_synthetic_audio(sample_rate=sample_rate, seconds=seconds, channel_count=channel_count, seed=seed),
            sample_rate,
            "<synthetic>",
        )
    audio, loaded_rate = load_audio_channels(audio_path)
    return fit_audio_duration(audio, loaded_rate, seconds), loaded_rate, str(audio_path)


def summarize_timing_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize numeric benchmark rows as min/median/max per timing key."""

    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = {
            "min": round(min(values), 3),
            "median": round(float(statistics.median(values)), 3),
            "max": round(max(values), 3),
        }
    return summary


def _direction_waveform_batch(audio: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    waveforms = extract_direction_waveforms(audio)
    directions = list(waveforms)
    return directions, [waveforms[direction] for direction in directions]


def benchmark_direction_teacher(
    teacher: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    runs: int = 5,
    warmups: int = 1,
    top_k: int = 5,
    synchronize: bool = True,
) -> dict[str, object]:
    """Benchmark full 7-direction teacher inference with optional warmup runs."""

    rows: list[dict[str, float]] = []
    directions = list(DIRECTION_NAMES)
    total_iterations = max(0, int(warmups)) + max(1, int(runs))
    for iteration in range(total_iterations):
        t0 = time.perf_counter()
        directions, direction_waveforms = _direction_waveform_batch(audio)
        t1 = time.perf_counter()
        _, profile = teacher.profile_predict_waveforms(
            direction_waveforms,
            sample_rate,
            top_k=top_k,
            audio_paths=directions,
            synchronize=synchronize,
        )
        timing_row = {"direction_extract_ms": round((t1 - t0) * 1000.0, 3), **profile.to_jsonable()}
        timing_row["total_with_extract_ms"] = round(timing_row["direction_extract_ms"] + timing_row["total_ms"], 3)
        if iteration >= warmups:
            rows.append(timing_row)
    return {
        "directions": directions,
        "batch_size": len(directions),
        "runs": len(rows),
        "warmups": int(warmups),
        "rows": rows,
        "summary": summarize_timing_rows(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark full 7-direction AudioSet teacher inference")
    parser.add_argument("audio", nargs="?", default=None, type=Path, help="Optional WAV/MP3/etc. file. Omit for synthetic 8ch audio.")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL, choices=TEACHER_MODEL_CHOICES)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation, e.g. sdpa")
    parser.add_argument("--compile-model", action="store_true", help="Wrap the teacher model with torch.compile")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode when --compile-model is set")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Synthetic audio sample rate")
    parser.add_argument("--channels", type=int, default=8, help="Synthetic audio channel count")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-synchronize", action="store_true", help="Do not synchronize CUDA/MPS between profiled stages")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio, sample_rate, source = load_or_synthesize_audio(
        args.audio,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
        channel_count=args.channels,
        seed=args.seed,
    )
    teacher = create_audio_event_teacher(
        args.teacher_model,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        compile_model=args.compile_mode if args.compile_model else False,
    )
    result = benchmark_direction_teacher(
        teacher,
        audio,
        sample_rate,
        runs=args.runs,
        warmups=args.warmups,
        top_k=args.top_k,
        synchronize=not args.no_synchronize,
    )
    payload = {
        "source": source,
        "sample_rate": sample_rate,
        "audio_shape": list(audio.shape),
        "teacher_model": args.teacher_model,
        "model_id": getattr(teacher, "model_id", args.model_id),
        "resolved_device": str(getattr(teacher, "device", args.device)),
        "resolved_dtype": str(getattr(teacher, "dtype", args.dtype)),
        "attn_implementation": args.attn_implementation,
        "compile_model": args.compile_mode if args.compile_model else False,
        **result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
