"""Qt-free formatting helpers for the direction-event debug HUD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from sound_model.event_detection import (
    DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS,
    DEFAULT_GUNSHOT_BIAS_MARGIN,
    DEFAULT_GUNSHOT_MAX_DIRECTIONS,
    DEFAULT_MAX_EVENTS_PER_DIRECTION,
    DIRECTION_EVENT_ORDER,
    DIRECTION_EVENT_ORDER_INDEX,
    GunshotDisplayDecision,
    displayed_active_events,
    gunshot_display_decision,
)


EVENT_DEBUG_LABELS = {
    "explosion": "EXP",
    "gunshot": "GUN",
    "vehicle": "VEH",
    "footstep": "FOOT",
}
DIRECTION_DEBUG_LABELS = {
    "front_left": "FL",
    "front": "F",
    "front_right": "FR",
    "left": "L",
    "right": "R",
    "rear_left": "RL",
    "rear_right": "RR",
}


@dataclass(frozen=True)
class DirectionEventDebugConfig:
    event_thresholds: Mapping[str, float]
    secondary_thresholds: Mapping[str, float]
    max_events: int
    gunshot_bias_margin: float
    gunshot_max_directions: int


def default_debug_config():
    return DirectionEventDebugConfig(
        event_thresholds=DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS,
        secondary_thresholds=DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS,
        max_events=DEFAULT_MAX_EVENTS_PER_DIRECTION,
        gunshot_bias_margin=DEFAULT_GUNSHOT_BIAS_MARGIN,
        gunshot_max_directions=DEFAULT_GUNSHOT_MAX_DIRECTIONS,
    )


def compact_score(score):
    return f"{max(0.0, min(1.0, float(score))):.2f}".lstrip("0")


def _debug_direction_label(direction):
    return DIRECTION_DEBUG_LABELS.get(direction, str(direction))


def _format_debug_direction_list(directions):
    ordered = sorted(directions or (), key=lambda direction: DIRECTION_EVENT_ORDER_INDEX.get(direction, len(DIRECTION_EVENT_ORDER)))
    if not ordered:
        return "--"
    return "/".join(_debug_direction_label(direction) for direction in ordered)


def _format_debug_direction_scores(direction_scores, max_items=7):
    items = list(direction_scores or ())
    if not items:
        return "--"
    return ",".join(f"{_debug_direction_label(direction)} {compact_score(score)}" for direction, score in items[: int(max_items)])


def _format_gunshot_cooldown_debug(display_debug):
    if display_debug is None:
        return "--"
    parts = []
    if display_debug.gunshot_global_cooldown_blocked_directions:
        parts.append(f"G:{_format_debug_direction_list(display_debug.gunshot_global_cooldown_blocked_directions)}")
    if display_debug.gunshot_sector_cooldown_blocked_directions:
        parts.append(f"S:{_format_debug_direction_list(display_debug.gunshot_sector_cooldown_blocked_directions)}")
    return ",".join(parts) if parts else "--"


def direction_event_device_label(runtime=None, requested_device=None, resolved_device=None, resolved_dtype=None):
    if runtime is not None:
        requested_device = getattr(runtime, "device", requested_device)
        resolved_device = getattr(runtime, "resolved_device", resolved_device)
        resolved_dtype = getattr(runtime, "resolved_dtype", resolved_dtype)
    requested = str(requested_device or "?")
    if resolved_device:
        resolved = str(resolved_device)
        label = resolved if requested == resolved else f"{requested}→{resolved}"
    else:
        label = f"{requested}→?" if requested == "auto" else requested
    return f"{label}/{resolved_dtype}" if resolved_dtype else label


def direction_event_debug_header(
    status=None,
    requested_device=None,
    resolved_device=None,
    resolved_dtype=None,
    teacher_model=None,
    *,
    default_teacher_model=None,
):
    teacher_label = str(teacher_model or default_teacher_model or "teacher")
    if status is None and requested_device is None and resolved_device is None and resolved_dtype is None:
        return f"{teacher_label} events"
    device_label = direction_event_device_label(
        requested_device=requested_device,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
    )
    return f"{teacher_label} {status or 'idle'} {device_label}"


def format_latency_ms(value):
    if value is None:
        return "--"
    return f"{max(0.0, float(value)):.0f}ms"


def direction_latency_debug_line(radar_latency_ms=None, ast_latency_ms=None):
    if radar_latency_ms is None or ast_latency_ms is None:
        delta = "--"
    else:
        delta_value = float(ast_latency_ms) - float(radar_latency_ms)
        sign = "+" if delta_value >= 0 else ""
        delta = f"{sign}{delta_value:.0f}ms"
    return f"lag radar {format_latency_ms(radar_latency_ms)} model {format_latency_ms(ast_latency_ms)} Δ{delta}"


def direction_event_debug_cell(
    direction,
    scores_by_direction,
    active_by_direction,
    threshold,
    *,
    config=None,
):
    config = default_debug_config() if config is None else config
    label = DIRECTION_DEBUG_LABELS[direction]
    scores = scores_by_direction.get(direction, {}) or {}
    events = displayed_active_events(
        scores,
        active_by_direction.get(direction, ()),
        threshold,
        config.event_thresholds,
        config.secondary_thresholds,
        config.max_events,
        config.gunshot_bias_margin,
    )
    if not events:
        return f"{label}: --"
    event_labels = "/".join(EVENT_DEBUG_LABELS[event_name] for event_name, _ in events)
    scores_label = "/".join(compact_score(score) for _, score in events)
    return f"{label}: {event_labels} {scores_label}"


def direction_event_gunshot_debug_line(
    prediction,
    display_debug=None,
    threshold=0.1,
    *,
    config=None,
):
    config = default_debug_config() if config is None else config
    if display_debug is not None:
        decision = display_debug.gunshot_decision
        shown_directions = display_debug.gunshot_emitted_directions
    elif prediction is not None:
        scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
        active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
        decision = gunshot_display_decision(
            scores_by_direction,
            active_by_direction,
            threshold,
            config.event_thresholds,
            config.gunshot_max_directions,
        )
        shown_directions = tuple(decision.allowed_directions)
    else:
        decision = GunshotDisplayDecision(tuple(), frozenset(), frozenset())
        shown_directions = tuple()

    return (
        f"gun cand {_format_debug_direction_scores(decision.candidate_scores)} "
        f"show {_format_debug_direction_list(shown_directions)} "
        f"sup {_format_debug_direction_list(decision.spatially_suppressed_directions)} "
        f"cd {_format_gunshot_cooldown_debug(display_debug)}"
    )


def direction_event_debug_lines(
    prediction,
    threshold=0.1,
    max_lines=7,
    status=None,
    requested_device=None,
    resolved_device=None,
    resolved_dtype=None,
    teacher_model=None,
    radar_latency_ms=None,
    ast_latency_ms=None,
    display_debug=None,
    *,
    config=None,
    default_teacher_model=None,
):
    config = default_debug_config() if config is None else config
    header = direction_event_debug_header(
        status,
        requested_device,
        resolved_device,
        resolved_dtype,
        teacher_model,
        default_teacher_model=default_teacher_model,
    )
    if prediction is None:
        lines = [
            header,
            direction_latency_debug_line(radar_latency_ms, ast_latency_ms),
            direction_event_gunshot_debug_line(None, display_debug, threshold, config=config),
            "waiting for audio/model...",
        ]
        return lines[: int(max_lines)] if max_lines else lines

    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}

    def cell(direction):
        return direction_event_debug_cell(
            direction,
            scores_by_direction,
            active_by_direction,
            threshold,
            config=config,
        )

    lines = [
        header,
        direction_latency_debug_line(radar_latency_ms, ast_latency_ms),
        direction_event_gunshot_debug_line(prediction, display_debug, threshold, config=config),
        f"        {cell('front')}",
        f"{cell('front_left')}    {cell('front_right')}",
        f"{cell('left')}    {cell('right')}",
        f"{cell('rear_left')}    {cell('rear_right')}",
    ]
    return lines[: int(max_lines)] if max_lines else lines
