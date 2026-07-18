"""Class-specific temporal policies for live gunshot and vehicle events.

This state operates after teacher inference and the per-window resolver.  It
does not replace display cooldown or spatial gunshot suppression: those remain
responsible for icon duplication and direction selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .vehicle_gun_resolver import (
    VehicleGunDecision,
    apply_vehicle_gun_decision,
    resolve_vehicle_gun,
)


@dataclass(frozen=True)
class EventTemporalConfig:
    """Hysteresis thresholds over normalized resolver/teacher scores."""

    gunshot_on_threshold: float = 0.10
    gunshot_decay: float = 0.08
    vehicle_on_threshold: float = 0.60
    vehicle_keep_threshold: float = 0.40
    vehicle_required_frames: int = 2
    vehicle_release_frames: int = 2
    vehicle_hold_decay: float = 0.85
    vehicle_release_decay: float = 0.60


@dataclass
class _DirectionTemporalState:
    gunshot_score: float = 0.0
    vehicle_score: float = 0.0
    vehicle_qualifying_frames: int = 0
    vehicle_missing_frames: int = 0
    vehicle_active: bool = False
    vehicle_persistence: float = 0.0


class EventTemporalState:
    """Maintain independent fast gunshot and persistent vehicle state per direction."""

    def __init__(self, config: EventTemporalConfig = EventTemporalConfig()):
        self.config = config
        self._directions: dict[str, _DirectionTemporalState] = {}

    def reset(self) -> None:
        self._directions.clear()

    def _direction_state(self, direction: str) -> _DirectionTemporalState:
        return self._directions.setdefault(direction, _DirectionTemporalState())

    def _update_vehicle_state(self, state: _DirectionTemporalState, signal: float) -> None:
        config = self.config
        if state.vehicle_active:
            if signal >= float(config.vehicle_keep_threshold):
                state.vehicle_missing_frames = 0
                state.vehicle_persistence = 1.0
            else:
                state.vehicle_missing_frames += 1
                state.vehicle_persistence = max(
                    0.0,
                    1.0 - state.vehicle_missing_frames / max(1, int(config.vehicle_release_frames)),
                )
                if state.vehicle_missing_frames >= max(1, int(config.vehicle_release_frames)):
                    state.vehicle_active = False
                    state.vehicle_qualifying_frames = 0
            return

        state.vehicle_missing_frames = 0
        if signal >= float(config.vehicle_on_threshold):
            state.vehicle_qualifying_frames += 1
        else:
            state.vehicle_qualifying_frames = 0
        required = max(1, int(config.vehicle_required_frames))
        state.vehicle_persistence = min(1.0, state.vehicle_qualifying_frames / required)
        if state.vehicle_qualifying_frames >= required:
            state.vehicle_active = True
            state.vehicle_persistence = 1.0

    @staticmethod
    def _vehicle_signal(scores, evidence, decision) -> float:
        values = [float((scores or {}).get("vehicle", 0.0))]
        if evidence is not None:
            values.extend(
                (
                    float(getattr(evidence, "vehicle_teacher_score", 0.0)),
                    float(getattr(evidence, "road_vehicle_label_score", 0.0)),
                )
            )
        if decision is not None:
            values.append(float(getattr(decision, "vehicle_evidence", 0.0)))
        return max(0.0, min(1.0, max(values, default=0.0)))

    def apply(self, prediction):
        """Attach resolver and temporal score layers, returning ``prediction``.

        ``raw_direction_event_scores`` remains the teacher layer,
        ``resolver_direction_event_scores`` stores the persistence-aware
        resolver layer, and both ``temporal_direction_event_scores`` and
        ``direction_event_scores`` store the final display input.
        """

        scores_by_direction = dict(getattr(prediction, "direction_event_scores", {}) or {})
        raw_by_direction = dict(getattr(prediction, "raw_direction_event_scores", {}) or {})
        evidence_by_direction = dict(getattr(prediction, "vehicle_gun_evidence_by_direction", {}) or {})
        decisions_by_direction = dict(getattr(prediction, "vehicle_gun_decisions_by_direction", {}) or {})
        active_by_direction = dict(getattr(prediction, "active_events_by_direction", {}) or {})
        directions = tuple(dict.fromkeys((*scores_by_direction, *raw_by_direction, *evidence_by_direction)))
        resolver_scores_by_direction = {}
        temporal_scores_by_direction = {}
        temporal_active_by_direction = {}

        for direction in directions:
            current_scores = dict(scores_by_direction.get(direction, {}) or {})
            raw_scores = dict(raw_by_direction.get(direction, current_scores) or {})
            evidence = evidence_by_direction.get(direction)
            decision = decisions_by_direction.get(direction)
            state = self._direction_state(direction)
            signal = self._vehicle_signal(current_scores, evidence, decision)
            self._update_vehicle_state(state, signal)

            if evidence is not None and float(raw_scores.get("explosion", 0.0)) < 0.25:
                evidence = replace(evidence, vehicle_persistence=state.vehicle_persistence)
                decision = resolve_vehicle_gun(evidence)
                current_scores = apply_vehicle_gun_decision(raw_scores, evidence, decision)
                evidence_by_direction[direction] = evidence
                decisions_by_direction[direction] = decision
            resolver_scores_by_direction[direction] = dict(current_scores)

            current_gunshot = max(0.0, min(1.0, float(current_scores.get("gunshot", 0.0))))
            state.gunshot_score = max(current_gunshot, state.gunshot_score * float(self.config.gunshot_decay))

            current_vehicle = max(0.0, min(1.0, float(current_scores.get("vehicle", 0.0))))
            if state.vehicle_active:
                decay = (
                    self.config.vehicle_hold_decay
                    if signal >= float(self.config.vehicle_keep_threshold)
                    else self.config.vehicle_release_decay
                )
                state.vehicle_score = max(
                    current_vehicle,
                    state.vehicle_score * float(decay),
                    float(self.config.vehicle_keep_threshold),
                )
            else:
                state.vehicle_score = 0.0

            temporal_scores = dict(current_scores)
            temporal_scores["gunshot"] = state.gunshot_score
            temporal_scores["vehicle"] = state.vehicle_score
            temporal_scores_by_direction[direction] = temporal_scores

            active_set = set(active_by_direction.get(direction, ()) or ())
            active_set.difference_update(("gunshot", "vehicle"))
            if state.gunshot_score >= float(self.config.gunshot_on_threshold):
                active_set.add("gunshot")
            if state.vehicle_active and state.vehicle_score >= float(self.config.vehicle_keep_threshold):
                active_set.add("vehicle")
            temporal_active_by_direction[direction] = [
                event_name
                for event_name in ("explosion", "gunshot", "vehicle", "footstep")
                if event_name in active_set
            ]

        prediction.resolver_direction_event_scores = resolver_scores_by_direction
        prediction.temporal_direction_event_scores = temporal_scores_by_direction
        prediction.direction_event_scores = temporal_scores_by_direction
        prediction.active_events_by_direction = temporal_active_by_direction
        prediction.vehicle_gun_evidence_by_direction = evidence_by_direction
        prediction.vehicle_gun_decisions_by_direction = decisions_by_direction
        return prediction
