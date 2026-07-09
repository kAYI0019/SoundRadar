"""Re-evaluate tagged SoundRadar capture samples across threshold profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .capture_direction_sample_gui import (
    EVENT_SCORE_ORDER,
    SAMPLE_TAGS,
    available_threshold_profiles,
    compare_threshold_profiles,
)
from .direction_events import predict_direction_events_file


EVALUATION_FIELDS = (
    "audio_path",
    "tag",
    "notes",
    "profile",
    "teacher_model",
    "device",
    "top_k",
    "shown_events",
    "shown_event_count",
    "expected_detected",
    "gunshot_shown",
    "gunshot_suppressed",
    "max_gunshot",
    "max_vehicle",
    "max_footstep",
    "max_explosion",
    "analysis_path",
)


def read_sample_library(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def default_evaluation_path(library_path: str | Path) -> Path:
    return Path(library_path).with_suffix(".evaluation.csv")


def resolve_audio_path(library_path: str | Path, audio_path: str | Path) -> Path:
    path = Path(str(audio_path)).expanduser()
    if path.is_absolute():
        return path
    return Path(library_path).expanduser().parent / path


def _shown_events_csv(events) -> str:
    return ";".join(f"{event['event']}:{event['direction']}:{float(event['score']):.3f}" for event in events)


def _directions_csv(directions) -> str:
    return ";".join(str(direction) for direction in directions)


def _expected_detected(tag: str, shown_events) -> str:
    if tag not in ("gunshot", "vehicle", "footstep"):
        return ""
    return "yes" if any(event.get("event") == tag for event in shown_events) else "no"


def evaluation_rows_for_prediction(
    library_record: dict[str, str],
    prediction,
    *,
    profiles=None,
    teacher_model: str = "ast",
    device: str = "auto",
    top_k: int = 5,
) -> list[dict[str, str]]:
    profiles = tuple(profiles or available_threshold_profiles())
    comparisons = compare_threshold_profiles(prediction, profiles=profiles)
    rows = []
    tag = str(library_record.get("tag", ""))
    if tag and tag not in SAMPLE_TAGS:
        raise ValueError(f"unknown sample tag: {tag}")

    for summary in comparisons:
        shown_events = list(summary.get("shown_events", ()))
        gunshot_display = summary.get("gunshot_display", {}) or {}
        max_scores = summary.get("max_scores", {}) or {}
        rows.append(
            {
                "audio_path": str(library_record.get("audio_path", "")),
                "tag": tag,
                "notes": str(library_record.get("notes", "")),
                "profile": str(summary.get("profile", "")),
                "teacher_model": str(teacher_model),
                "device": str(device),
                "top_k": str(int(top_k)),
                "shown_events": _shown_events_csv(shown_events),
                "shown_event_count": str(int(summary.get("shown_event_count", len(shown_events)))),
                "expected_detected": _expected_detected(tag, shown_events),
                "gunshot_shown": _directions_csv(gunshot_display.get("shown_directions", ())),
                "gunshot_suppressed": _directions_csv(gunshot_display.get("spatially_suppressed_directions", ())),
                "max_gunshot": f"{float(max_scores.get('gunshot', 0.0)):.3f}",
                "max_vehicle": f"{float(max_scores.get('vehicle', 0.0)):.3f}",
                "max_footstep": f"{float(max_scores.get('footstep', 0.0)):.3f}",
                "max_explosion": f"{float(max_scores.get('explosion', 0.0)):.3f}",
                "analysis_path": str(library_record.get("analysis_path", "")),
            }
        )
    return rows


def write_evaluation_csv(path: str | Path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EVALUATION_FIELDS})
    return path


def evaluate_sample_library(
    library_path: str | Path,
    *,
    out_path: str | Path | None = None,
    profiles=None,
    teacher_model: str = "ast",
    device: str = "auto",
    top_k: int = 5,
    limit: int | None = None,
) -> Path:
    library_path = Path(library_path)
    out_path = Path(out_path) if out_path is not None else default_evaluation_path(library_path)
    records = read_sample_library(library_path)
    if limit is not None:
        records = records[: int(limit)]

    rows = []
    for record in records:
        audio_path = resolve_audio_path(library_path, record.get("audio_path", ""))
        prediction = predict_direction_events_file(
            audio_path,
            teacher_model=teacher_model,
            device=device,
            top_k=top_k,
        )
        row_source = dict(record)
        row_source["audio_path"] = str(audio_path)
        rows.extend(
            evaluation_rows_for_prediction(
                row_source,
                prediction,
                profiles=profiles,
                teacher_model=teacher_model,
                device=device,
                top_k=top_k,
            )
        )
    return write_evaluation_csv(out_path, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate a SoundRadar sample_library.csv across threshold profiles")
    parser.add_argument("library", type=Path, help="CSV produced by the capture GUI Tag row")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV; defaults to *.evaluation.csv")
    parser.add_argument("--profiles", nargs="+", default=None, help="Profiles to compare; defaults to all runtime profiles")
    parser.add_argument("--teacher-model", default="ast", choices=["ast", "efficientat-mn10", "efficientat-mn20"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N library rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = evaluate_sample_library(
        args.library,
        out_path=args.out,
        profiles=args.profiles,
        teacher_model=args.teacher_model,
        device=args.device,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
