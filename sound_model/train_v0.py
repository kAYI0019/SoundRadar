"""Train the V0 PUBG accessibility sound-event model.

Examples
--------
Smoke-train a real checkpoint from deterministic synthetic clips::

    .venv/bin/python -m sound_model.train_v0 --generate-smoke-data --epochs 8

Train from a real manifest::

    .venv/bin/python -m sound_model.train_v0 --manifest sound_model/dataset_v0/labels_clip_v0.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np

try:  # Support both `python -m sound_model.train_v0` and direct script execution.
    from .audio_features import (
        DEFAULT_CLASSES,
        DEFAULT_MEL_BINS,
        DEFAULT_N_FFT,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_STFT_HOP,
        DEFAULT_WINDOW_SEC,
        extract_feature_vector,
        read_wav,
    )
    from .dataset import ClipRecord, generate_synthetic_dataset, load_manifest, split_records
    from .model import choose_thresholds, multilabel_metrics, save_checkpoint, sigmoid, train_mlp
except ImportError:  # pragma: no cover - direct script fallback
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sound_model.audio_features import (  # type: ignore
        DEFAULT_CLASSES,
        DEFAULT_MEL_BINS,
        DEFAULT_N_FFT,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_STFT_HOP,
        DEFAULT_WINDOW_SEC,
        extract_feature_vector,
        read_wav,
    )
    from sound_model.dataset import ClipRecord, generate_synthetic_dataset, load_manifest, split_records  # type: ignore
    from sound_model.model import choose_thresholds, multilabel_metrics, save_checkpoint, sigmoid, train_mlp  # type: ignore


def load_feature_matrix(records: list[ClipRecord], feature_config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for record in records:
        if not record.path.exists():
            raise FileNotFoundError(record.path)
        audio, sample_rate = read_wav(record.path)
        x_rows.append(extract_feature_vector(audio, sample_rate, **feature_config))
        y_rows.append(record.labels.astype(np.float32))
    if not x_rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, len(DEFAULT_CLASSES)), dtype=np.float32)
    return np.stack(x_rows, axis=0).astype(np.float32), np.stack(y_rows, axis=0).astype(np.float32)


def train_from_manifest(args: argparse.Namespace) -> dict[str, object]:
    classes = tuple(args.classes.split(",")) if args.classes else DEFAULT_CLASSES
    feature_config: dict[str, object] = {
        "target_rate": args.sample_rate,
        "window_sec": args.window_sec,
        "n_fft": args.n_fft,
        "stft_hop": args.stft_hop,
        "mel_bins": args.mel_bins,
    }

    manifest_path: Path
    if args.generate_smoke_data:
        manifest_path = generate_synthetic_dataset(
            args.smoke_data_dir,
            count_per_class=args.smoke_count_per_class,
            sample_rate=args.sample_rate,
            seed=args.seed,
            classes=classes,
        )
    elif args.manifest:
        manifest_path = Path(args.manifest)
    else:
        raise SystemExit("--manifest 또는 --generate-smoke-data 중 하나가 필요합니다.")

    records = load_manifest(manifest_path, classes=classes, root=args.dataset_root)
    train_records, valid_records, test_records = split_records(records, seed=args.seed)
    if not valid_records:
        raise ValueError("validation split is empty; provide valid rows or more data")

    x_train, y_train = load_feature_matrix(train_records, feature_config)
    x_valid, y_valid = load_feature_matrix(valid_records, feature_config)
    x_test, y_test = load_feature_matrix(test_records, feature_config) if test_records else (x_valid, y_valid)

    model, standardizer, pos_weight, history = train_mlp(
        x_train,
        y_train,
        x_valid,
        y_valid,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        l2=args.l2,
    )
    x_valid_n = standardizer.transform(x_valid)
    x_test_n = standardizer.transform(x_test)
    valid_prob = sigmoid(model.predict_logits(x_valid_n))
    thresholds = choose_thresholds(y_valid, valid_prob)
    test_prob = sigmoid(model.predict_logits(x_test_n))

    metrics = {
        "manifest": str(manifest_path),
        "splits": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
        },
        "classes": list(classes),
        "feature_config": feature_config,
        "history": history.rows,
        "valid": multilabel_metrics(y_valid, valid_prob, thresholds, classes),
        "test": multilabel_metrics(y_test, test_prob, thresholds, classes),
        "note": (
            "Synthetic smoke-data metrics only prove the training pipeline runs; "
            "real PUBG clips are required for gameplay-valid performance."
            if args.generate_smoke_data
            else "Metrics computed from the provided manifest."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.model_name
    metrics_path = args.output_dir / args.metrics_name
    save_checkpoint(
        checkpoint_path,
        model,
        standardizer,
        classes=classes,
        thresholds=thresholds,
        feature_config=feature_config,
        metrics=metrics,
        pos_weight=pos_weight,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["metrics_path"] = str(metrics_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V0 clip-level multilabel PUBG sound-event model")
    parser.add_argument("--manifest", type=Path, help="CSV manifest for real WAV clips")
    parser.add_argument("--dataset-root", type=Path, help="Base directory for relative audio paths")
    parser.add_argument("--classes", help="Comma-separated class list; default is V0 classes")
    parser.add_argument("--generate-smoke-data", action="store_true", help="Create deterministic synthetic clips and train on them")
    parser.add_argument("--smoke-data-dir", type=Path, default=Path("sound_model/generated/smoke_dataset"))
    parser.add_argument("--smoke-count-per-class", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=Path("sound_model/artifacts"))
    parser.add_argument("--model-name", default="model_mlp_v0_smoke.npz")
    parser.add_argument("--metrics-name", default="metrics_v0_smoke.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--stft-hop", type=int, default=DEFAULT_STFT_HOP)
    parser.add_argument("--mel-bins", type=int, default=DEFAULT_MEL_BINS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = train_from_manifest(args)
    print(json.dumps({
        "checkpoint": metrics["checkpoint"],
        "metrics_path": metrics["metrics_path"],
        "valid_macro_f1": metrics["valid"]["macro_f1"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "note": metrics["note"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
