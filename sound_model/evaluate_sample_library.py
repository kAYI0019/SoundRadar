"""Re-evaluate tagged SoundRadar capture samples across threshold profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .capture_direction_sample_gui import (
    EVENT_SCORE_ORDER,
    SAMPLE_TAGS,
    available_threshold_profiles,
    compare_threshold_profiles,
)
from .ast_teacher import create_audio_event_teacher, load_audio_channels
from .direction_events import score_direction_events
from .sample_library import read_library


EVALUATION_FIELDS = (
    "audio_path",
    "tag",
    "notes",
    "profile",
    "teacher_model",
    "device",
    "top_k",
    "label_score_semantics",
    "expected_label",
    "predicted_primary_label",
    "predicted_secondary_labels",
    "raw_gunshot_score",
    "raw_vehicle_score",
    "adjusted_gunshot_score",
    "adjusted_vehicle_score",
    "score_margin",
    "vehicle_shown",
    "classification_result",
    "resolver_label",
    "resolver_confidence",
    "resolver_reason",
    "inference_latency_ms",
    "confusion_prediction_label",
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
SUMMARY_FIELDS = (
    "profile",
    "rows",
    "target_rows",
    "target_detected",
    "target_missed",
    "unknown_or_bad_rows",
    "unknown_or_bad_with_icons",
    "multi_icon_rows",
    "gunshot_shown_rows",
    "gunshot_suppressed_rows",
    "avg_shown_event_count",
    "gunshot_precision",
    "gunshot_recall",
    "gunshot_f1",
    "vehicle_precision",
    "vehicle_recall",
    "vehicle_f1",
    "macro_f1",
    "gunshot_to_vehicle_count",
    "gunshot_to_vehicle_rate",
    "vehicle_to_gunshot_count",
    "vehicle_to_gunshot_rate",
    "both_shown_count",
    "both_shown_rate",
    "miss_rate",
    "false_positive_rate",
    "unknown_rate",
    "mean_displayed_event_count",
    "p50_inference_latency_ms",
    "p95_inference_latency_ms",
    "confusion_gunshot_gunshot",
    "confusion_gunshot_vehicle",
    "confusion_gunshot_unknown",
    "confusion_vehicle_gunshot",
    "confusion_vehicle_vehicle",
    "confusion_vehicle_unknown",
    "confusion_unknown_gunshot",
    "confusion_unknown_vehicle",
    "confusion_unknown_unknown",
)
TARGET_TAGS = frozenset(("gunshot", "vehicle", "footstep"))
UNKNOWN_TAGS = frozenset(("unknown", "bad sample"))


def read_sample_library(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def default_evaluation_path(library_path: str | Path) -> Path:
    return Path(library_path).with_suffix(".evaluation.csv")


def default_summary_path(evaluation_path: str | Path) -> Path:
    return Path(evaluation_path).with_suffix(".summary.csv")


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


def _expected_label(tag: str) -> str:
    return tag if tag in TARGET_TAGS else "unknown"


def _primary_prediction(shown_events) -> tuple[str, list[str], str | None]:
    ordered = sorted(
        shown_events,
        key=lambda event: float(event.get("score", 0.0)),
        reverse=True,
    )
    if not ordered:
        return "unknown", [], None
    primary = str(ordered[0].get("event", "unknown") or "unknown")
    secondary = []
    for event in ordered[1:]:
        label = str(event.get("event", "unknown") or "unknown")
        if label != primary and label not in secondary:
            secondary.append(label)
    return primary, secondary, str(ordered[0].get("direction", "")) or None


def classify_prediction(expected_label: str, primary_label: str, shown_event_types) -> str:
    shown_types = set(shown_event_types or ())
    if "gunshot" in shown_types and "vehicle" in shown_types:
        return "both_shown"
    if expected_label == "gunshot":
        if primary_label == "gunshot":
            return "correct"
        if primary_label == "vehicle":
            return "gunshot_to_vehicle"
        return "missed"
    if expected_label == "vehicle":
        if primary_label == "vehicle":
            return "correct"
        if primary_label == "gunshot":
            return "vehicle_to_gunshot"
        return "missed"
    if expected_label == "footstep":
        return "correct" if primary_label == "footstep" else "missed"
    return "false_positive" if primary_label != "unknown" else "unknown"


def _max_direction_score(prediction, event_name: str, *, raw: bool = False) -> float:
    field_name = "raw_direction_event_scores" if raw else "direction_event_scores"
    scores_by_direction = getattr(prediction, field_name, {}) or {}
    if raw and not scores_by_direction:
        scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    return max(
        (float((scores or {}).get(event_name, 0.0)) for scores in scores_by_direction.values()),
        default=0.0,
    )


def _resolver_direction(prediction, primary_direction: str | None) -> str | None:
    decisions = getattr(prediction, "vehicle_gun_decisions_by_direction", {}) or {}
    if primary_direction in decisions:
        return primary_direction
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    candidates = [direction for direction in scores_by_direction if direction in decisions]
    if not candidates:
        return None
    raw_scores_by_direction = getattr(prediction, "raw_direction_event_scores", {}) or {}
    return max(
        candidates,
        key=lambda direction: (
            max(
                float((scores_by_direction.get(direction, {}) or {}).get("gunshot", 0.0)),
                float((scores_by_direction.get(direction, {}) or {}).get("vehicle", 0.0)),
            ),
            max(
                float((raw_scores_by_direction.get(direction, {}) or {}).get("gunshot", 0.0)),
                float((raw_scores_by_direction.get(direction, {}) or {}).get("vehicle", 0.0)),
            ),
        ),
    )


def _object_field(value, name: str, default=""):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


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
        shown_types = {str(event.get("event", "")) for event in shown_events}
        primary_label, secondary_labels, primary_direction = _primary_prediction(shown_events)
        expected_label = _expected_label(tag)
        classification_result = classify_prediction(expected_label, primary_label, shown_types)
        resolver_direction = _resolver_direction(prediction, primary_direction)
        resolver_decision = (
            (getattr(prediction, "vehicle_gun_decisions_by_direction", {}) or {}).get(resolver_direction)
            if resolver_direction is not None
            else None
        )
        confusion_prediction = (
            "unknown"
            if classification_result in ("both_shown", "missed", "unknown")
            or primary_label not in ("gunshot", "vehicle")
            else primary_label
        )
        gunshot_display = summary.get("gunshot_display", {}) or {}
        max_scores = summary.get("max_scores", {}) or {}
        vehicle_directions = [
            str(event.get("direction", ""))
            for event in shown_events
            if event.get("event") == "vehicle"
        ]
        adjusted_gunshot = _max_direction_score(prediction, "gunshot")
        adjusted_vehicle = _max_direction_score(prediction, "vehicle")
        rows.append(
            {
                "audio_path": str(library_record.get("audio_path", "")),
                "tag": tag,
                "notes": str(library_record.get("notes", "")),
                "profile": str(summary.get("profile", "")),
                "teacher_model": str(teacher_model),
                "device": str(device),
                "top_k": str(int(top_k)),
                "label_score_semantics": str(
                    (getattr(prediction, "label_score_semantics_by_direction", {}) or {}).get(
                        resolver_direction,
                        "unknown",
                    )
                ),
                "expected_label": expected_label,
                "predicted_primary_label": primary_label,
                "predicted_secondary_labels": ";".join(secondary_labels),
                "raw_gunshot_score": f"{_max_direction_score(prediction, 'gunshot', raw=True):.3f}",
                "raw_vehicle_score": f"{_max_direction_score(prediction, 'vehicle', raw=True):.3f}",
                "adjusted_gunshot_score": f"{adjusted_gunshot:.3f}",
                "adjusted_vehicle_score": f"{adjusted_vehicle:.3f}",
                "score_margin": f"{adjusted_gunshot - adjusted_vehicle:.3f}",
                "shown_events": _shown_events_csv(shown_events),
                "shown_event_count": str(int(summary.get("shown_event_count", len(shown_events)))),
                "expected_detected": _expected_detected(tag, shown_events),
                "gunshot_shown": _directions_csv(gunshot_display.get("shown_directions", ())),
                "vehicle_shown": _directions_csv(vehicle_directions),
                "classification_result": classification_result,
                "resolver_label": str(_object_field(resolver_decision, "label", "unknown")),
                "resolver_confidence": f"{float(_object_field(resolver_decision, 'confidence', 0.0)):.3f}",
                "resolver_reason": str(_object_field(resolver_decision, "reason", "resolver unavailable")),
                "inference_latency_ms": (
                    f"{float(getattr(prediction, 'inference_latency_ms')):.3f}"
                    if getattr(prediction, "inference_latency_ms", None) is not None
                    else ""
                ),
                "confusion_prediction_label": confusion_prediction,
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


def _int_field(row, name: str) -> int:
    try:
        return int(row.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _float_field(row, name: str) -> float | None:
    try:
        value = row.get(name, "")
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_matrix_for_rows(rows) -> dict[str, dict[str, int]]:
    """Return the 3x3 matrix; ``both_shown`` rows use the unknown column."""

    labels = ("gunshot", "vehicle", "unknown")
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        actual = str(row.get("expected_label", row.get("tag", "unknown")))
        if actual not in matrix:
            actual = "unknown"
        predicted = str(row.get("confusion_prediction_label", row.get("predicted_primary_label", "unknown")))
        if predicted not in labels:
            predicted = "unknown"
        matrix[actual][predicted] += 1
    return matrix


def _class_metrics(matrix, label: str) -> tuple[float, float, float]:
    true_positive = int(matrix[label][label])
    false_positive = sum(int(matrix[actual][label]) for actual in matrix if actual != label)
    false_negative = sum(int(matrix[label][predicted]) for predicted in matrix[label] if predicted != label)
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    f1 = _safe_rate(2.0 * precision * recall, precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_evaluation_rows(rows) -> list[dict[str, str]]:
    """Aggregate profile metrics with explicit, stable rate denominators.

    Cross-confusion rates use the corresponding actual class count, miss rate
    uses tagged target rows, false-positive rate uses unknown/bad rows, and
    both/unknown rates use all rows in the profile.
    """

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get("profile", "")), []).append(row)

    summaries = []
    for profile, profile_rows in sorted(grouped.items()):
        shown_counts = [_int_field(row, "shown_event_count") for row in profile_rows]
        target_rows = [row for row in profile_rows if row.get("tag") in TARGET_TAGS]
        unknown_or_bad_rows = [row for row in profile_rows if row.get("tag") in UNKNOWN_TAGS]
        target_detected = [
            row
            for row in target_rows
            if (
                row.get("classification_result") == "correct"
                if row.get("classification_result")
                else row.get("expected_detected") == "yes"
            )
        ]
        matrix = confusion_matrix_for_rows(profile_rows)
        gunshot_precision, gunshot_recall, gunshot_f1 = _class_metrics(matrix, "gunshot")
        vehicle_precision, vehicle_recall, vehicle_f1 = _class_metrics(matrix, "vehicle")
        actual_gunshot = sum(matrix["gunshot"].values())
        actual_vehicle = sum(matrix["vehicle"].values())
        gunshot_to_vehicle_count = sum(
            1 for row in profile_rows if row.get("classification_result") == "gunshot_to_vehicle"
        )
        vehicle_to_gunshot_count = sum(
            1 for row in profile_rows if row.get("classification_result") == "vehicle_to_gunshot"
        )
        both_shown_count = sum(1 for row in profile_rows if row.get("classification_result") == "both_shown")
        missed_count = sum(1 for row in profile_rows if row.get("classification_result") == "missed")
        false_positive_count = sum(
            1 for row in unknown_or_bad_rows if _int_field(row, "shown_event_count") > 0
        )
        unknown_count = sum(1 for row in profile_rows if row.get("classification_result") == "unknown")
        latencies = [
            latency
            for latency in (_float_field(row, "inference_latency_ms") for row in profile_rows)
            if latency is not None
        ]
        row_count = len(profile_rows)
        summary = {
            "profile": profile,
            "rows": str(len(profile_rows)),
            "target_rows": str(len(target_rows)),
            "target_detected": str(len(target_detected)),
            "target_missed": str(len(target_rows) - len(target_detected)),
            "unknown_or_bad_rows": str(len(unknown_or_bad_rows)),
            "unknown_or_bad_with_icons": str(sum(1 for row in unknown_or_bad_rows if _int_field(row, "shown_event_count") > 0)),
            "multi_icon_rows": str(sum(1 for count in shown_counts if count > 1)),
            "gunshot_shown_rows": str(sum(1 for row in profile_rows if row.get("gunshot_shown"))),
            "gunshot_suppressed_rows": str(sum(1 for row in profile_rows if row.get("gunshot_suppressed"))),
            "avg_shown_event_count": f"{(sum(shown_counts) / len(shown_counts)) if shown_counts else 0.0:.3f}",
            "gunshot_precision": f"{gunshot_precision:.3f}",
            "gunshot_recall": f"{gunshot_recall:.3f}",
            "gunshot_f1": f"{gunshot_f1:.3f}",
            "vehicle_precision": f"{vehicle_precision:.3f}",
            "vehicle_recall": f"{vehicle_recall:.3f}",
            "vehicle_f1": f"{vehicle_f1:.3f}",
            "macro_f1": f"{(gunshot_f1 + vehicle_f1) / 2.0:.3f}",
            "gunshot_to_vehicle_count": str(gunshot_to_vehicle_count),
            "gunshot_to_vehicle_rate": f"{_safe_rate(gunshot_to_vehicle_count, actual_gunshot):.3f}",
            "vehicle_to_gunshot_count": str(vehicle_to_gunshot_count),
            "vehicle_to_gunshot_rate": f"{_safe_rate(vehicle_to_gunshot_count, actual_vehicle):.3f}",
            "both_shown_count": str(both_shown_count),
            "both_shown_rate": f"{_safe_rate(both_shown_count, row_count):.3f}",
            "miss_rate": f"{_safe_rate(missed_count, len(target_rows)):.3f}",
            "false_positive_rate": f"{_safe_rate(false_positive_count, len(unknown_or_bad_rows)):.3f}",
            "unknown_rate": f"{_safe_rate(unknown_count, row_count):.3f}",
            "mean_displayed_event_count": f"{_safe_rate(sum(shown_counts), row_count):.3f}",
            "p50_inference_latency_ms": f"{float(np.percentile(latencies, 50)):.3f}" if latencies else "",
            "p95_inference_latency_ms": f"{float(np.percentile(latencies, 95)):.3f}" if latencies else "",
        }
        for actual in ("gunshot", "vehicle", "unknown"):
            for predicted in ("gunshot", "vehicle", "unknown"):
                summary[f"confusion_{actual}_{predicted}"] = str(matrix[actual][predicted])
        summaries.append(summary)
    return summaries


def format_evaluation_summary(summaries) -> str:
    lines = []
    for summary in summaries:
        lines.append(
            f"profile {summary.get('profile', '')}: gunshot F1={summary.get('gunshot_f1', '0')} "
            f"vehicle F1={summary.get('vehicle_f1', '0')} macro F1={summary.get('macro_f1', '0')}"
        )
        lines.append(
            "actual \\ predicted  gunshot  vehicle  unknown"
        )
        for actual in ("gunshot", "vehicle", "unknown"):
            values = [summary.get(f"confusion_{actual}_{predicted}", "0") for predicted in ("gunshot", "vehicle", "unknown")]
            lines.append(f"{actual:<19} {values[0]:>7} {values[1]:>8} {values[2]:>8}")
        lines.append(
            f"cross errors: gunshot->vehicle={summary.get('gunshot_to_vehicle_count', '0')} "
            f"vehicle->gunshot={summary.get('vehicle_to_gunshot_count', '0')} "
            f"both_shown={summary.get('both_shown_count', '0')} missed={summary.get('target_missed', '0')}"
        )
    return "\n".join(lines)


def write_summary_csv(path: str | Path, summaries) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in SUMMARY_FIELDS})
    return path


def evaluate_sample_library_rows(
    library_path: str | Path,
    *,
    profiles=None,
    teacher_model: str = "ast",
    device: str = "auto",
    top_k: int = 5,
    limit: int | None = None,
) -> list[dict[str, str]]:
    library_path = Path(library_path)
    records = [record for record in read_library(library_path) if record.get("review_status") == "reviewed"]
    if limit is not None:
        records = records[: int(limit)]
    if not records:
        return []

    teacher = create_audio_event_teacher(teacher_model, device=device)
    rows = []
    for record in records:
        audio_path = resolve_audio_path(library_path, record.get("audio_path", ""))
        audio, sample_rate = load_audio_channels(audio_path)
        prediction = score_direction_events(
            audio,
            sample_rate,
            teacher,
            top_k=top_k,
            source_path=str(audio_path),
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
    return rows


def evaluate_sample_library(
    library_path: str | Path,
    *,
    out_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    profiles=None,
    teacher_model: str = "ast",
    device: str = "auto",
    top_k: int = 5,
    limit: int | None = None,
) -> Path:
    library_path = Path(library_path)
    out_path = Path(out_path) if out_path is not None else default_evaluation_path(library_path)
    rows = evaluate_sample_library_rows(
        library_path,
        profiles=profiles,
        teacher_model=teacher_model,
        device=device,
        top_k=top_k,
        limit=limit,
    )
    evaluation_path = write_evaluation_csv(out_path, rows)
    if summary_path is not None:
        write_summary_csv(summary_path, summarize_evaluation_rows(rows))
    return evaluation_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate a SoundRadar sample_library.csv across threshold profiles")
    parser.add_argument("library", type=Path, help="CSV produced by the capture GUI Tag row")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV; defaults to *.evaluation.csv")
    parser.add_argument("--summary-out", type=Path, default=None, help="Profile summary CSV; defaults to evaluation *.summary.csv")
    parser.add_argument("--no-summary", action="store_true", help="Do not write the profile summary CSV")
    parser.add_argument("--profiles", nargs="+", default=None, help="Profiles to compare; defaults to all runtime profiles")
    parser.add_argument("--teacher-model", default="ast", choices=["ast", "efficientat-mn10", "efficientat-mn20"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N library rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_path = args.out or default_evaluation_path(args.library)
    summary_path = None if args.no_summary else (args.summary_out or default_summary_path(evaluation_path))
    out_path = evaluate_sample_library(
        args.library,
        out_path=evaluation_path,
        summary_path=summary_path,
        profiles=args.profiles,
        teacher_model=args.teacher_model,
        device=args.device,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(f"wrote {out_path}")
    if summary_path is not None:
        print(f"wrote {summary_path}")
        print(format_evaluation_summary(summarize_evaluation_rows(read_sample_library(out_path))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
