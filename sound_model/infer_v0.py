"""Run V0 checkpoint inference on a WAV clip."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

try:
    from .model import load_checkpoint
except ImportError:  # pragma: no cover - direct script fallback
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sound_model.model import load_checkpoint  # type: ignore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict PUBG accessibility sound-event probabilities for one WAV")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("sound_model/artifacts/model_mlp_v0_smoke.npz"))
    args = parser.parse_args(argv)

    model = load_checkpoint(args.checkpoint)
    probabilities = model.predict_wav(args.audio)
    print(json.dumps({
        "audio": str(args.audio),
        "checkpoint": str(args.checkpoint),
        "probabilities": probabilities,
        "active_labels": model.active_labels(probabilities),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
