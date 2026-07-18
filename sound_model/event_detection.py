"""Pure event selection, smoothing, and pulse helpers for SoundRadar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIRECTION_EVENT_SECTORS = {
    "front_left": 11,
    "front": 0,
    "front_right": 1,
    "left": 9,
    "right": 3,
    "rear_left": 7,
    "rear_right": 5,
}
DIRECTION_EVENT_ORDER = tuple(DIRECTION_EVENT_SECTORS)
DIRECTION_EVENT_ORDER_INDEX = {direction: index for index, direction in enumerate(DIRECTION_EVENT_ORDER)}
DIRECTION_EVENT_PRIORITY = ("explosion", "gunshot", "vehicle", "footstep")
CLASSIFIED_EVENT_KINDS = frozenset(DIRECTION_EVENT_PRIORITY)
GUNSHOT_DIRECTION_NEIGHBORS = {
    "front_left": frozenset(("front", "left")),
    "front": frozenset(("front_left", "front_right")),
    "front_right": frozenset(("front", "right")),
    "left": frozenset(("front_left", "rear_left")),
    "right": frozenset(("front_right", "rear_right")),
    "rear_left": frozenset(("left", "rear_right")),
    "rear_right": frozenset(("right", "rear_left")),
}

DEFAULT_RIPPLE_THRESHOLD = 0.03
DEFAULT_RIPPLE_COOLDOWN = 0.18
DEFAULT_RIPPLE_DURATION = 0.65
DEFAULT_DIRECTION_EVENT_THRESHOLD = 0.10
DEFAULT_DIRECTION_EVENT_COOLDOWN = 0.55
DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS = {
    "gunshot": 0.10,
    "explosion": 0.85,
    "vehicle": 0.60,
    "footstep": 0.85,
}
DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS = {
    "gunshot": 0.35,
    "explosion": 0.85,
    "vehicle": 0.60,
    "footstep": 0.85,
}
DEFAULT_GUNSHOT_BIAS_MARGIN = 0.25
DEFAULT_MAX_EVENTS_PER_DIRECTION = 2
DEFAULT_GUNSHOT_MAX_DIRECTIONS = 2
DEFAULT_GUNSHOT_GLOBAL_COOLDOWN = 0.18
DEFAULT_EVENT_ICON_DURATION = 3.5
DEFAULT_SMOOTHING_WINDOW = 3


@dataclass
class SoundPulse:
    sector: int
    strength: float
    created_at: float
    duration: float = DEFAULT_RIPPLE_DURATION
    kind: str = "unknown"
    lane_index: int = 0
    lane_count: int = 1


@dataclass(frozen=True)
class GunshotDisplayDecision:
    candidate_scores: tuple
    allowed_directions: frozenset
    spatially_suppressed_directions: frozenset


@dataclass(frozen=True)
class DirectionEventPulseDebug:
    gunshot_decision: GunshotDisplayDecision
    gunshot_emitted_directions: tuple
    gunshot_global_cooldown_blocked_directions: tuple
    gunshot_sector_cooldown_blocked_directions: tuple


@dataclass
class SmoothedDirectionEventPrediction:
    sample_rate: int
    direction_event_scores: dict
    active_events_by_direction: dict
    top_labels_by_direction: dict
    source_path: str | None = None
    mode: str = "smoothed direction-event inference"

    def to_jsonable(self):
        return {
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "mode": self.mode,
            "directions": list(DIRECTION_EVENT_SECTORS),
            "classes": sorted({event for scores in self.direction_event_scores.values() for event in scores}),
            "direction_event_scores": self.direction_event_scores,
            "active_events_by_direction": self.active_events_by_direction,
            "top_labels_by_direction": self.top_labels_by_direction,
        }


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def classify_basic_sound_event(strength, previous_strength=0.0):
    strength = clamp(float(strength))
    previous_strength = clamp(float(previous_strength))
    if strength >= 0.85:
        return "impact"
    if strength - previous_strength >= 0.25:
        return "sharp"
    return "unknown"


def should_emit_pulse(last_time, now, cooldown=DEFAULT_RIPPLE_COOLDOWN):
    return now - last_time >= cooldown


def create_pulses_from_levels(
    levels,
    now,
    threshold=DEFAULT_RIPPLE_THRESHOLD,
    cooldown=DEFAULT_RIPPLE_COOLDOWN,
    last_pulse_times=None,
    previous_levels=None,
    pulse_duration=DEFAULT_RIPPLE_DURATION,
):
    pulses = []
    levels = np.asarray(levels)
    if previous_levels is None:
        previous_levels = np.zeros_like(levels)
    for sector, strength in enumerate(levels):
        strength = float(strength)
        if strength < threshold:
            continue
        if last_pulse_times is not None and not should_emit_pulse(last_pulse_times[sector], now, cooldown):
            continue
        previous_strength = float(previous_levels[sector]) if sector < len(previous_levels) else 0.0
        pulses.append(
            SoundPulse(
                sector=sector,
                strength=clamp(strength),
                created_at=now,
                duration=pulse_duration,
                kind=classify_basic_sound_event(strength, previous_strength),
            )
        )
        if last_pulse_times is not None:
            last_pulse_times[sector] = now
    return pulses


def event_display_threshold(event_name, threshold, event_thresholds):
    return max(float(threshold), float(event_thresholds.get(event_name, threshold)))


def dominant_active_event(
    scores,
    active_events,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
    gunshot_bias_margin=DEFAULT_GUNSHOT_BIAS_MARGIN,
):
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    active = set(active_events or [])
    best_event = None
    best_score = 0.0
    gunshot_score = clamp(float(scores.get("gunshot", 0.0)))
    gunshot_threshold = event_display_threshold("gunshot", threshold, event_thresholds)
    for event_name in DIRECTION_EVENT_PRIORITY:
        score = clamp(float(scores.get(event_name, 0.0)))
        required_score = event_display_threshold(event_name, threshold, event_thresholds)
        if score >= required_score and (not active or event_name in active):
            if best_event is None or score > best_score:
                best_event = event_name
                best_score = score
    if (
        best_event in ("explosion", "footstep")
        and gunshot_score >= gunshot_threshold
        and (not active or "gunshot" in active)
        and best_score - gunshot_score <= float(gunshot_bias_margin)
    ):
        return "gunshot", gunshot_score
    return best_event, best_score


def active_event_candidates(
    scores,
    active_events,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
):
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    active = set(active_events or [])
    candidates = []
    for event_name in DIRECTION_EVENT_PRIORITY:
        score = clamp(float(scores.get(event_name, 0.0)))
        if score >= event_display_threshold(event_name, threshold, event_thresholds) and (not active or event_name in active):
            candidates.append((event_name, score))
    return candidates


def displayed_active_events(
    scores,
    active_events,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
    secondary_thresholds=None,
    max_events=DEFAULT_MAX_EVENTS_PER_DIRECTION,
    gunshot_bias_margin=DEFAULT_GUNSHOT_BIAS_MARGIN,
):
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    if secondary_thresholds is None:
        secondary_thresholds = DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS
    candidates = active_event_candidates(scores, active_events, threshold, event_thresholds)
    if not candidates:
        return []

    primary_event, primary_score = dominant_active_event(
        scores,
        active_events,
        threshold,
        event_thresholds,
        gunshot_bias_margin,
    )
    ordered = []
    if primary_event is not None:
        ordered.append((primary_event, primary_score))

    for event_name, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if event_name == primary_event:
            continue
        if primary_event == "vehicle" and event_name == "gunshot":
            continue
        if score >= event_display_threshold(event_name, threshold, secondary_thresholds):
            ordered.append((event_name, score))
        if len(ordered) >= int(max_events):
            break
    return ordered[: int(max_events)]


def _direction_event_is_active(event_name, active_events):
    active = set(active_events or ())
    return not active or event_name in active


def _event_score_for_direction(scores_by_direction, active_by_direction, direction, event_name, threshold, event_thresholds):
    scores = scores_by_direction.get(direction, {}) or {}
    score = clamp(float(scores.get(event_name, 0.0)))
    if score < event_display_threshold(event_name, threshold, event_thresholds):
        return None
    if not _direction_event_is_active(event_name, active_by_direction.get(direction, ())):
        return None
    return score


def _direction_clusters(directions, neighbors):
    remaining = set(directions)
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster = {start}
        stack = [start]
        while stack:
            direction = stack.pop()
            for neighbor in neighbors.get(direction, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    stack.append(neighbor)
        clusters.append(cluster)
    return clusters


def _sorted_direction_scores(scores_by_direction):
    return tuple(
        sorted(
            scores_by_direction.items(),
            key=lambda item: (-item[1], DIRECTION_EVENT_ORDER_INDEX.get(item[0], len(DIRECTION_EVENT_ORDER))),
        )
    )


def gunshot_candidate_scores(
    scores_by_direction,
    active_by_direction,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
):
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    candidate_scores = {}
    for direction in DIRECTION_EVENT_SECTORS:
        score = _event_score_for_direction(
            scores_by_direction,
            active_by_direction,
            direction,
            "gunshot",
            threshold,
            event_thresholds,
        )
        if score is not None:
            candidate_scores[direction] = score
    return candidate_scores


def gunshot_display_decision(
    scores_by_direction,
    active_by_direction,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
    max_directions=DEFAULT_GUNSHOT_MAX_DIRECTIONS,
):
    """Return gunshot candidates and spatial suppression decisions."""
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    candidate_scores = gunshot_candidate_scores(
        scores_by_direction,
        active_by_direction,
        threshold,
        event_thresholds,
    )
    if len(candidate_scores) <= 1:
        return GunshotDisplayDecision(
            candidate_scores=_sorted_direction_scores(candidate_scores),
            allowed_directions=frozenset(candidate_scores),
            spatially_suppressed_directions=frozenset(),
        )

    allowed = []
    for cluster in _direction_clusters(candidate_scores, GUNSHOT_DIRECTION_NEIGHBORS):
        winner = max(
            cluster,
            key=lambda direction: (
                candidate_scores[direction],
                -DIRECTION_EVENT_ORDER_INDEX.get(direction, len(DIRECTION_EVENT_ORDER)),
            ),
        )
        allowed.append((winner, candidate_scores[winner]))

    allowed.sort(key=lambda item: (-item[1], DIRECTION_EVENT_ORDER_INDEX.get(item[0], len(DIRECTION_EVENT_ORDER))))
    allowed_directions = {direction for direction, _ in allowed[: max(1, int(max_directions))]}
    return GunshotDisplayDecision(
        candidate_scores=_sorted_direction_scores(candidate_scores),
        allowed_directions=frozenset(allowed_directions),
        spatially_suppressed_directions=frozenset(set(candidate_scores) - allowed_directions),
    )


def spatially_allowed_gunshot_directions(
    scores_by_direction,
    active_by_direction,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=None,
    max_directions=DEFAULT_GUNSHOT_MAX_DIRECTIONS,
):
    return set(
        gunshot_display_decision(
            scores_by_direction,
            active_by_direction,
            threshold,
            event_thresholds,
            max_directions,
        ).allowed_directions
    )


def suppress_displayed_events_for_direction(direction, events, allowed_gunshot_directions):
    if direction in allowed_gunshot_directions:
        return events
    return [(event_name, score) for event_name, score in events if event_name != "gunshot"]


def _prediction_event_names(predictions):
    names = set(("background", *DIRECTION_EVENT_PRIORITY))
    for prediction in predictions:
        scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
        for scores in scores_by_direction.values():
            names.update((scores or {}).keys())
    return tuple(sorted(names))


def smooth_direction_event_predictions(predictions, *, window=DEFAULT_SMOOTHING_WINDOW):
    valid = [prediction for prediction in predictions if prediction is not None]
    if not valid:
        return None
    window = max(1, int(window))
    valid = valid[-window:]
    if len(valid) == 1 or window <= 1:
        return valid[-1]

    latest = valid[-1]
    event_names = _prediction_event_names(valid)
    weights = np.arange(1, len(valid) + 1, dtype=np.float32)
    weight_sum = float(np.sum(weights))
    smoothed_scores = {}
    smoothed_active = {}

    for direction in DIRECTION_EVENT_SECTORS:
        direction_scores = {}
        active_union = set()
        for prediction in valid:
            active_union.update((getattr(prediction, "active_events_by_direction", {}) or {}).get(direction, ()) or ())
        for event_name in event_names:
            weighted_score = 0.0
            for weight, prediction in zip(weights, valid):
                scores = (getattr(prediction, "direction_event_scores", {}) or {}).get(direction, {}) or {}
                weighted_score += float(weight) * clamp(float(scores.get(event_name, 0.0)))
            direction_scores[event_name] = weighted_score / weight_sum
        smoothed_scores[direction] = direction_scores
        smoothed_active[direction] = [event_name for event_name in DIRECTION_EVENT_PRIORITY if event_name in active_union]

    return SmoothedDirectionEventPrediction(
        sample_rate=int(getattr(latest, "sample_rate", 0) or 0),
        direction_event_scores=smoothed_scores,
        active_events_by_direction=smoothed_active,
        top_labels_by_direction=dict(getattr(latest, "top_labels_by_direction", {}) or {}),
        source_path=getattr(latest, "source_path", None),
    )


def create_pulses_from_direction_events(
    prediction,
    now,
    threshold=DEFAULT_DIRECTION_EVENT_THRESHOLD,
    cooldown=DEFAULT_DIRECTION_EVENT_COOLDOWN,
    last_pulse_times=None,
    last_global_event_times=None,
    display_debug=None,
    event_thresholds=None,
    secondary_thresholds=None,
    max_events=DEFAULT_MAX_EVENTS_PER_DIRECTION,
    gunshot_bias_margin=DEFAULT_GUNSHOT_BIAS_MARGIN,
    gunshot_max_directions=DEFAULT_GUNSHOT_MAX_DIRECTIONS,
    gunshot_global_cooldown=DEFAULT_GUNSHOT_GLOBAL_COOLDOWN,
    event_icon_duration=DEFAULT_EVENT_ICON_DURATION,
):
    if event_thresholds is None:
        event_thresholds = DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS
    if secondary_thresholds is None:
        secondary_thresholds = DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS
    pulses = []
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
    gunshot_decision = gunshot_display_decision(
        scores_by_direction,
        active_by_direction,
        threshold,
        event_thresholds,
        gunshot_max_directions,
    )
    allowed_gunshot_directions = set(gunshot_decision.allowed_directions)
    emitted_gunshot_directions = []
    global_cooldown_blocked_gunshot_directions = []
    sector_cooldown_blocked_gunshot_directions = []

    for direction, sector in DIRECTION_EVENT_SECTORS.items():
        scores = scores_by_direction.get(direction, {})
        events = displayed_active_events(
            scores,
            active_by_direction.get(direction, ()),
            threshold,
            event_thresholds,
            secondary_thresholds,
            max_events,
            gunshot_bias_margin,
        )
        events = suppress_displayed_events_for_direction(direction, events, allowed_gunshot_directions)
        if not events:
            continue
        if (
            last_global_event_times is not None
            and any(event_name == "gunshot" for event_name, _ in events)
            and not should_emit_pulse(
                last_global_event_times.get("gunshot", -float("inf")),
                now,
                gunshot_global_cooldown,
            )
        ):
            global_cooldown_blocked_gunshot_directions.append(direction)
            events = [(event_name, score) for event_name, score in events if event_name != "gunshot"]
            if not events:
                continue
        if last_pulse_times is not None and not should_emit_pulse(last_pulse_times[sector], now, cooldown):
            if any(event_name == "gunshot" for event_name, _ in events):
                sector_cooldown_blocked_gunshot_directions.append(direction)
            continue
        lane_count = len(events)
        for lane_index, (event_name, score) in enumerate(events):
            pulses.append(
                SoundPulse(
                    sector=sector,
                    strength=score,
                    created_at=now,
                    duration=event_icon_duration,
                    kind=event_name,
                    lane_index=lane_index,
                    lane_count=lane_count,
                )
            )
        if last_pulse_times is not None:
            last_pulse_times[sector] = now
        if last_global_event_times is not None and any(event_name == "gunshot" for event_name, _ in events):
            last_global_event_times["gunshot"] = now
        if any(event_name == "gunshot" for event_name, _ in events):
            emitted_gunshot_directions.append(direction)
    if display_debug is not None:
        display_debug["debug"] = DirectionEventPulseDebug(
            gunshot_decision=gunshot_decision,
            gunshot_emitted_directions=tuple(emitted_gunshot_directions),
            gunshot_global_cooldown_blocked_directions=tuple(global_cooldown_blocked_gunshot_directions),
            gunshot_sector_cooldown_blocked_directions=tuple(sector_cooldown_blocked_gunshot_directions),
        )
    return pulses
