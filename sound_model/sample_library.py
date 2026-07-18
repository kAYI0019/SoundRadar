"""Backward-compatible, update-in-place CSV sample library helpers."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid


REVIEW_LABELS = ("gunshot", "vehicle", "footstep", "explosion", "other", "unknown", "bad_sample")
REVIEW_STATUSES = ("pending", "reviewed", "rejected")
LEGACY_FIELDS = ("created_at", "audio_path", "tag", "notes", "analysis_path", "peak_summary")
LIBRARY_FIELDS = LEGACY_FIELDS + (
    "sample_id",
    "audio_hash",
    "session_id",
    "capture_source",
    "capture_seconds",
    "provisional_tag",
    "reviewed_labels",
    "review_status",
    "label_confidence",
    "metadata_path",
)


def audio_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_labels(labels) -> tuple[str, ...]:
    if isinstance(labels, str):
        try:
            decoded = json.loads(labels)
            values = decoded if isinstance(decoded, list) else [labels]
        except json.JSONDecodeError:
            values = [value.strip() for value in labels.replace(";", ",").split(",") if value.strip()]
    else:
        values = list(labels or ())
    normalized = []
    for value in values:
        label = "bad_sample" if str(value) == "bad sample" else str(value)
        if label not in REVIEW_LABELS:
            raise ValueError(f"unknown sample label: {value}")
        if label not in normalized:
            normalized.append(label)
    return tuple(normalized)


def normalize_library_record(record: dict[str, str]) -> dict[str, str]:
    result = {field: str(record.get(field, "") or "") for field in LIBRARY_FIELDS}
    legacy_tag = result["tag"]
    labels = normalize_labels(result["reviewed_labels"]) if result["reviewed_labels"] else ()
    if not result["provisional_tag"] and legacy_tag:
        result["provisional_tag"] = legacy_tag
    if not labels and legacy_tag:
        labels = normalize_labels(legacy_tag)
    if not result["review_status"]:
        result["review_status"] = "reviewed" if labels else "pending"
    result["reviewed_labels"] = json.dumps(list(labels), ensure_ascii=False)
    if not result["tag"] and labels:
        result["tag"] = "bad sample" if labels[0] == "bad_sample" else labels[0]
    return result


def read_library(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return [normalize_library_record(dict(row)) for row in csv.DictReader(source)]


def build_sample_record(
    *,
    audio_path: str | Path,
    provisional_tag: str | None = None,
    reviewed_labels=(),
    review_status: str | None = None,
    notes: str = "",
    analysis_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    peak_summary: str | None = None,
    capture_source: str = "manual",
    capture_seconds: float | None = None,
    sample_id: str | None = None,
    session_id: str = "",
    created_at: datetime | None = None,
) -> dict[str, str]:
    audio_path = Path(audio_path)
    labels = normalize_labels(reviewed_labels)
    if provisional_tag == "bad sample":
        provisional_tag = "bad_sample"
    if provisional_tag is not None and provisional_tag not in REVIEW_LABELS:
        raise ValueError(f"unknown provisional tag: {provisional_tag}")
    status = review_status or ("reviewed" if labels else "pending")
    if status not in REVIEW_STATUSES:
        raise ValueError(f"unknown review status: {status}")
    created_at = created_at or datetime.now()
    digest = audio_sha256(audio_path) if audio_path.exists() else ""
    first_label = labels[0] if labels else provisional_tag or ""
    return normalize_library_record(
        {
            "created_at": created_at.isoformat(timespec="seconds"),
            "audio_path": str(audio_path),
            "tag": "bad sample" if first_label == "bad_sample" else first_label,
            "notes": str(notes),
            "analysis_path": "" if analysis_path is None else str(Path(analysis_path)),
            "peak_summary": "" if peak_summary is None else str(peak_summary),
            "sample_id": sample_id or f"{created_at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "audio_hash": digest,
            "session_id": str(session_id),
            "capture_source": str(capture_source),
            "capture_seconds": "" if capture_seconds is None else f"{float(capture_seconds):g}",
            "provisional_tag": provisional_tag or "",
            "reviewed_labels": json.dumps(list(labels), ensure_ascii=False),
            "review_status": status,
            "metadata_path": "" if metadata_path is None else str(Path(metadata_path)),
        }
    )


def upsert_library_record(path: str | Path, record: dict[str, str]) -> tuple[Path, bool]:
    """Write one logical record. Returns ``(path, created)``."""

    path = Path(path)
    normalized = normalize_library_record(record)
    rows = read_library(path)
    match_index = None
    for index, row in enumerate(rows):
        same_id = normalized["sample_id"] and row.get("sample_id") == normalized["sample_id"]
        same_hash = normalized["audio_hash"] and row.get("audio_hash") == normalized["audio_hash"]
        same_path = Path(row.get("audio_path", "")) == Path(normalized["audio_path"])
        if same_id or same_hash or same_path:
            match_index = index
            break
    created = match_index is None
    if created:
        rows.append(normalized)
    else:
        merged = dict(rows[match_index])
        merged.update({key: value for key, value in normalized.items() if value != ""})
        if rows[match_index].get("sample_id"):
            merged["sample_id"] = rows[match_index]["sample_id"]
        if rows[match_index].get("created_at"):
            merged["created_at"] = rows[match_index]["created_at"]
        rows[match_index] = normalize_library_record(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=LIBRARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LIBRARY_FIELDS})
    temporary.replace(path)
    return path, created
