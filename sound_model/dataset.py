"""Dataset helpers for V0 clip-level PUBG sound-event training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import itertools
import random

import numpy as np

from .audio_features import DEFAULT_CLASSES, write_wav


@dataclass(frozen=True)
class ClipRecord:
    path: Path
    labels: np.ndarray
    split: str


def parse_label_value(value: str | int | float | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "none"}:
        return 0.0
    return 1.0


def label_vector_from_row(row: dict[str, str], classes: tuple[str, ...] = DEFAULT_CLASSES) -> np.ndarray:
    """Parse either one column named ``labels`` or one binary column per class."""

    vector = np.zeros(len(classes), dtype=np.float32)
    if row.get("labels"):
        labels = {
            label.strip().lower()
            for part in row["labels"].replace(";", "|").replace(",", "|").split("|")
            for label in [part]
            if label.strip()
        }
        for idx, name in enumerate(classes):
            vector[idx] = 1.0 if name.lower() in labels else 0.0
        return vector

    found_any_class_column = False
    for idx, name in enumerate(classes):
        if name in row:
            found_any_class_column = True
            vector[idx] = parse_label_value(row.get(name))
    if not found_any_class_column:
        raise ValueError("manifest needs a labels column or one binary column per class")
    return vector


def load_manifest(
    manifest_path: str | Path,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    root: str | Path | None = None,
) -> list[ClipRecord]:
    """Load a CSV manifest.

    Required audio column: one of ``audio_path``, ``path``, ``file``, or ``wav``.
    Label format: ``labels`` pipe/semicolon/comma list or binary class columns.
    Optional split column: ``train``, ``valid``/``val``, or ``test``.
    """

    manifest_path = Path(manifest_path)
    base = Path(root) if root is not None else manifest_path.parent
    records: list[ClipRecord] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty manifest: {manifest_path}")
        for line_no, row in enumerate(reader, start=2):
            raw_path = row.get("audio_path") or row.get("path") or row.get("file") or row.get("wav")
            if not raw_path:
                raise ValueError(f"manifest line {line_no}: missing audio_path/path/file/wav column")
            path = Path(raw_path)
            if not path.is_absolute():
                path = base / path
            split = (row.get("split") or "train").strip().lower()
            if split == "val":
                split = "valid"
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"manifest line {line_no}: invalid split {split!r}")
            records.append(ClipRecord(path=path, labels=label_vector_from_row(row, classes), split=split))

    if not records:
        raise ValueError(f"manifest has no rows: {manifest_path}")
    return records


def split_records(
    records: list[ClipRecord], *, seed: int = 42, valid_ratio: float = 0.15, test_ratio: float = 0.15
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    """Return train/valid/test records, auto-splitting if the manifest has only train rows."""

    train = [record for record in records if record.split == "train"]
    valid = [record for record in records if record.split == "valid"]
    test = [record for record in records if record.split == "test"]
    if valid or test:
        return train, valid, test

    shuffled = train[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_valid = int(round(n_total * valid_ratio))
    test = shuffled[:n_test]
    valid = shuffled[n_test : n_test + n_valid]
    train = shuffled[n_test + n_valid :]
    if not valid and len(train) > 1:
        valid = [train.pop()]
    return train, valid, test


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    count_per_class: int = 16,
    sample_rate: int = 16_000,
    seed: int = 42,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> Path:
    """Generate a deterministic synthetic smoke dataset and return manifest path.

    The clips are intentionally simple and are **not** a substitute for real PUBG
    recordings.  They are used to verify that feature extraction, multilabel
    learning, checkpointing, and inference all work end-to-end.
    """

    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rng = np.random.default_rng(seed)

    rows: list[dict[str, str]] = []
    split_cycle = itertools.cycle(["train", "train", "train", "train", "valid", "test"])

    for event in classes:
        for idx in range(count_per_class):
            audio = _base_noise(sample_rate, rng)
            labels = {name: "0" for name in classes}
            if event == "background":
                labels["background"] = "1"
            else:
                _add_event(audio, event, sample_rate, rng)
                labels[event] = "1"
            filename = f"{event}_{idx:04d}.wav"
            write_wav(audio_dir / filename, audio, sample_rate)
            rows.append({"audio_path": f"audio/{filename}", "split": next(split_cycle), **labels})

    non_background = [name for name in classes if name != "background"]
    overlap_pairs = [("footstep", "gunshot"), ("footstep", "vehicle"), ("gunshot", "explosion")]
    overlap_pairs += [tuple(rng.choice(non_background, size=2, replace=False)) for _ in range(max(1, count_per_class // 2))]
    for idx, pair in enumerate(overlap_pairs):
        audio = _base_noise(sample_rate, rng)
        labels = {name: "0" for name in classes}
        for event in pair:
            _add_event(audio, event, sample_rate, rng)
            labels[event] = "1"
        filename = f"overlap_{idx:04d}_{pair[0]}_{pair[1]}.wav"
        write_wav(audio_dir / filename, audio, sample_rate)
        rows.append({"audio_path": f"audio/{filename}", "split": next(split_cycle), **labels})

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["audio_path", "split", *classes]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _base_noise(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = sample_rate
    noise = rng.normal(0.0, 0.008, size=(length, 2)).astype(np.float32)
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    ambience = 0.006 * np.sin(2 * np.pi * 45.0 * t + rng.uniform(0, 2 * np.pi))
    noise += ambience[:, None].astype(np.float32)
    return noise


def _add_event(audio: np.ndarray, event: str, sample_rate: int, rng: np.random.Generator) -> None:
    if event == "footstep":
        for center in np.linspace(0.18, 0.82, 4) + rng.normal(0, 0.015, size=4):
            _add_decay_burst(audio, sample_rate, float(center), 0.055, 0.09, rng, low_freq=120.0)
    elif event == "gunshot":
        _add_decay_burst(audio, sample_rate, float(rng.uniform(0.25, 0.75)), 0.045, 0.38, rng, low_freq=850.0)
    elif event == "vehicle":
        t = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        rumble = 0.12 * np.sin(2 * np.pi * 72.0 * t) + 0.05 * np.sin(2 * np.pi * 146.0 * t)
        wobble = 0.65 + 0.35 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi))
        pan = float(rng.uniform(-0.4, 0.4))
        audio += _pan((rumble * wobble).astype(np.float32), pan)
    elif event == "explosion":
        center = float(rng.uniform(0.18, 0.65))
        _add_decay_burst(audio, sample_rate, center, 0.22, 0.42, rng, low_freq=58.0)
    elif event == "background":
        return
    else:
        raise ValueError(f"unknown synthetic event {event!r}")
    np.clip(audio, -1.0, 1.0, out=audio)


def _add_decay_burst(
    audio: np.ndarray,
    sample_rate: int,
    center_sec: float,
    duration_sec: float,
    amplitude: float,
    rng: np.random.Generator,
    *,
    low_freq: float,
) -> None:
    start = max(0, int(center_sec * sample_rate))
    length = max(1, int(duration_sec * sample_rate))
    end = min(len(audio), start + length)
    if end <= start:
        return
    t = np.arange(end - start, dtype=np.float32) / sample_rate
    envelope = np.exp(-6.0 * t / max(duration_sec, 1e-3)).astype(np.float32)
    tone = np.sin(2 * np.pi * low_freq * t).astype(np.float32)
    burst_noise = rng.normal(0.0, 1.0, size=end - start).astype(np.float32)
    signal = amplitude * envelope * (0.45 * tone + 0.55 * burst_noise)
    pan = float(rng.uniform(-0.75, 0.75))
    audio[start:end] += _pan(signal, pan)


def _pan(signal: np.ndarray, pan: float) -> np.ndarray:
    left_gain = np.sqrt(max(0.0, (1.0 - pan) * 0.5))
    right_gain = np.sqrt(max(0.0, (1.0 + pan) * 0.5))
    return np.stack([signal * left_gain, signal * right_gain], axis=1).astype(np.float32)
