"""Qt-free presentation model for SoundRadar capture analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


EVENT_NAMES_KO = {
    "gunshot": "총성",
    "vehicle": "차량",
    "footstep": "발소리",
    "explosion": "폭발",
    "unknown": "감지 없음",
}
DIRECTION_NAMES_KO = {
    "front_left": "왼쪽 앞",
    "front": "앞",
    "front_right": "오른쪽 앞",
    "left": "왼쪽",
    "right": "오른쪽",
    "rear_left": "왼쪽 뒤",
    "rear_right": "오른쪽 뒤",
    "unknown": "방향 없음",
}
PROFILE_NAMES_KO = {
    "default": "기본",
    "quiet": "조용함",
    "aggressive": "적극적",
    "debug": "디버그",
}


@dataclass(frozen=True)
class DirectionRow:
    direction: str
    displayed_events: tuple[str, ...]
    scores: dict[str, float]
    top_labels: tuple[str, ...] = ()
    suppressed: bool = False
    cooldown_blocked: bool = False
    noteworthy: bool = False


@dataclass(frozen=True)
class ProfileRow:
    profile: str
    shown: str
    suppressed: str
    description: str


@dataclass(frozen=True)
class AnalysisSummary:
    audio_path: str
    requested_model: str
    loaded_model: str
    display_model: str
    analysis_device: str
    threshold_profile: str
    inference_latency_ms: float | None
    analyzed_at: str
    sample_rate: int | None
    channel_count: int | None
    duration_seconds: float | None
    primary_event: str
    primary_direction: str
    max_scores: dict[str, float]
    score_margin: float
    warnings: tuple[str, ...] = ()
    explanation: str = ""
    direction_rows: tuple[DirectionRow, ...] = field(default_factory=tuple)
    profile_rows: tuple[ProfileRow, ...] = field(default_factory=tuple)

    @property
    def model_consistent(self) -> bool:
        return bool(self.requested_model) and self.requested_model == self.loaded_model == self.display_model

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_consistent"] = self.model_consistent
        return payload


def _display_debug_sets(display_debug) -> tuple[set[str], set[str]]:
    if display_debug is None:
        return set(), set()
    suppressed = set(getattr(display_debug, "spatially_suppressed_directions", ()) or ())
    cooldown = set(getattr(display_debug, "cooldown_blocked_directions", ()) or ())
    if isinstance(display_debug, dict):
        suppressed = set(display_debug.get("spatially_suppressed_directions", ()) or ())
        cooldown = set(display_debug.get("cooldown_blocked_directions", ()) or ())
    return suppressed, cooldown


def build_direction_rows(prediction, displayed_events=(), display_debug=None, *, margin_threshold: float = 0.15):
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    top_by_direction = getattr(prediction, "top_labels_by_direction", {}) or {}
    events_by_direction: dict[str, list[str]] = {}
    for event in displayed_events or ():
        events_by_direction.setdefault(str(event.get("direction", "")), []).append(str(event.get("event", "")))
    suppressed, cooldown = _display_debug_sets(display_debug)
    rows = []
    for direction, scores_value in scores_by_direction.items():
        scores = {name: float((scores_value or {}).get(name, 0.0)) for name in EVENT_NAMES_KO if name != "unknown"}
        gun_vehicle_close = abs(scores.get("gunshot", 0.0) - scores.get("vehicle", 0.0)) <= margin_threshold and max(
            scores.get("gunshot", 0.0), scores.get("vehicle", 0.0)
        ) > 0.0
        active = tuple(events_by_direction.get(direction, ()))
        noteworthy = bool(active or gun_vehicle_close or direction in suppressed or direction in cooldown)
        labels = tuple(
            str(item.get("label", ""))
            for item in (top_by_direction.get(direction, ()) or ())[:3]
            if isinstance(item, dict) and item.get("label")
        )
        rows.append(
            DirectionRow(
                direction=str(direction),
                displayed_events=active,
                scores=scores,
                top_labels=labels,
                suppressed=direction in suppressed,
                cooldown_blocked=direction in cooldown,
                noteworthy=noteworthy,
            )
        )
    return tuple(rows)


def build_profile_rows(comparisons: Iterable[dict[str, object]]) -> tuple[ProfileRow, ...]:
    rows = []
    for comparison in comparisons or ():
        shown_events = comparison.get("shown_events", ()) or ()
        shown = ", ".join(
            f"{DIRECTION_NAMES_KO.get(str(event.get('direction')), str(event.get('direction')))} "
            f"{EVENT_NAMES_KO.get(str(event.get('event')), str(event.get('event')))}"
            for event in shown_events
        ) or "없음"
        gunshot_display = comparison.get("gunshot_display", {}) or {}
        suppressed_directions = gunshot_display.get("spatially_suppressed_directions", ()) or ()
        suppressed = ", ".join(DIRECTION_NAMES_KO.get(str(value), str(value)) for value in suppressed_directions) or "없음"
        profile = str(comparison.get("profile", "default"))
        description = {
            "default": "현재 기본 설정",
            "quiet": "오탐과 중복 표시 감소",
            "aggressive": "더 많은 이벤트 표시",
            "debug": "임계값 검증용",
        }.get(profile, "사용자 설정")
        rows.append(ProfileRow(profile=profile, shown=shown, suppressed=suppressed, description=description))
    return tuple(rows)


def build_analysis_warnings(
    *,
    requested_model: str,
    loaded_model: str,
    direction_rows: Iterable[DirectionRow],
    inference_latency_ms: float | None,
    channel_count: int | None = None,
    active_channel_count: int | None = None,
    peak: float | None = None,
    margin_threshold: float = 0.15,
) -> tuple[str, ...]:
    rows = tuple(direction_rows)
    warnings = []
    if requested_model != loaded_model:
        warnings.append("model_mismatch")
    max_gunshot = max((row.scores.get("gunshot", 0.0) for row in rows), default=0.0)
    max_vehicle = max((row.scores.get("vehicle", 0.0) for row in rows), default=0.0)
    max_explosion = max((row.scores.get("explosion", 0.0) for row in rows), default=0.0)
    displayed = {event for row in rows for event in row.displayed_events}
    if max(max_gunshot, max_vehicle) > 0.0 and abs(max_gunshot - max_vehicle) <= margin_threshold:
        warnings.append("gunshot_vehicle_ambiguous")
    if {"gunshot", "vehicle"}.issubset(displayed):
        warnings.append("gunshot_vehicle_overlap")
    if max_gunshot >= 0.7 and max_explosion >= 0.7:
        warnings.append("gunshot_explosion_overlap")
    if any(
        len(rows) > 1
        and len({round(row.scores.get(event, 0.0), 6) for row in rows}) == 1
        and max((row.scores.get(event, 0.0) for row in rows), default=0.0) > 0.0
        for event in ("gunshot", "vehicle", "footstep", "explosion")
    ):
        warnings.append("uniform_direction_scores")
    if any(any(value > 0 for value in row.scores.values()) for row in rows) and not displayed:
        warnings.append("prediction_without_display")
    if channel_count is not None and channel_count >= 8 and active_channel_count is not None and active_channel_count <= 2:
        warnings.append("few_active_channels")
    if peak is not None and peak < 0.001:
        warnings.append("recording_too_quiet")
    if peak is not None and peak >= 0.999:
        warnings.append("recording_clipping")
    if inference_latency_ms is not None and inference_latency_ms > 1000.0:
        warnings.append("slow_inference")
    return tuple(dict.fromkeys(warnings))


def build_human_readable_explanation(primary_event: str, primary_direction: str, warnings: Iterable[str]) -> str:
    warnings = set(warnings)
    if primary_event == "unknown":
        return "표시 임계값을 넘은 이벤트가 없습니다."
    message = (
        f"{DIRECTION_NAMES_KO.get(primary_direction, primary_direction)}에서 "
        f"{EVENT_NAMES_KO.get(primary_event, primary_event)}이(가) 가장 강하게 감지되었습니다."
    )
    if "gunshot_vehicle_ambiguous" in warnings:
        message += " 총성 점수와 차량 점수의 차이가 작아 혼동 샘플로 검수하는 것이 좋습니다."
    if "gunshot_explosion_overlap" in warnings:
        message += " 총성과 폭발이 함께 높아 복합음 또는 오분류 가능성을 확인하세요."
    return message


def build_analysis_summary(
    prediction,
    *,
    audio_path: str | Path,
    requested_model: str,
    loaded_model: str,
    device: str,
    threshold_profile: str,
    analyzed_at: str,
    displayed_events=(),
    display_debug=None,
    profile_comparisons=(),
    channel_count: int | None = None,
    duration_seconds: float | None = None,
    active_channel_count: int | None = None,
    peak: float | None = None,
) -> AnalysisSummary:
    rows = build_direction_rows(prediction, displayed_events, display_debug)
    ordered = sorted(
        displayed_events or (),
        key=lambda event: float(event.get("score", 0.0)),
        reverse=True,
    )
    primary_event = str(ordered[0].get("event", "unknown")) if ordered else "unknown"
    primary_direction = str(ordered[0].get("direction", "unknown")) if ordered else "unknown"
    max_scores = {
        event: max((row.scores.get(event, 0.0) for row in rows), default=0.0)
        for event in ("gunshot", "vehicle", "footstep", "explosion")
    }
    warnings = build_analysis_warnings(
        requested_model=requested_model,
        loaded_model=loaded_model,
        direction_rows=rows,
        inference_latency_ms=getattr(prediction, "inference_latency_ms", None),
        channel_count=channel_count,
        active_channel_count=active_channel_count,
        peak=peak,
    )
    return AnalysisSummary(
        audio_path=str(Path(audio_path)),
        requested_model=str(requested_model),
        loaded_model=str(loaded_model),
        display_model=str(loaded_model),
        analysis_device=str(device),
        threshold_profile=str(threshold_profile),
        inference_latency_ms=getattr(prediction, "inference_latency_ms", None),
        analyzed_at=str(analyzed_at),
        sample_rate=int(getattr(prediction, "sample_rate", 0) or 0) or None,
        channel_count=channel_count,
        duration_seconds=duration_seconds,
        primary_event=primary_event,
        primary_direction=primary_direction,
        max_scores=max_scores,
        score_margin=abs(max_scores["gunshot"] - max_scores["vehicle"]),
        warnings=warnings,
        explanation=build_human_readable_explanation(primary_event, primary_direction, warnings),
        direction_rows=rows,
        profile_rows=build_profile_rows(profile_comparisons),
    )


def summary_from_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return v2 summary fields while accepting legacy analysis JSON."""

    teacher_model = str(payload.get("teacher_model", "unknown") or "unknown")
    summary = dict(payload.get("analysis_summary", {}) or {})
    summary.setdefault("audio_path", str(payload.get("audio_path", "")))
    summary.setdefault("requested_model", str(payload.get("requested_model", teacher_model)))
    summary.setdefault("loaded_model", str(payload.get("loaded_model", teacher_model)))
    summary.setdefault("display_model", str(payload.get("display_model", summary["loaded_model"])))
    summary.setdefault("analysis_device", str(payload.get("analysis_device", payload.get("device", "unknown"))))
    summary.setdefault("threshold_profile", str(payload.get("threshold_profile", "default")))
    summary.setdefault("warnings", list(payload.get("warnings", ())))
    return summary
