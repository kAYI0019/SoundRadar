"""Explainable vehicle-versus-gunshot resolution for SoundRadar.

The resolver intentionally uses only existing teacher scores and time-domain
waveform evidence.  It does not add spectral features, temporal state, or a
trained classifier, so its decisions remain deterministic and easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _linear_score(value: float, start: float, full: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


@dataclass(frozen=True)
class VehicleGunEvidence:
    """Teacher and waveform evidence kept separate for later analysis."""

    gunshot_teacher_score: float = 0.0
    vehicle_teacher_score: float = 0.0
    gunshot_label_score: float = 0.0
    road_vehicle_label_score: float = 0.0
    transient_score: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    crest_factor: float = 0.0


@dataclass(frozen=True)
class VehicleGunDecision:
    label: str
    confidence: float
    gunshot_evidence: float
    vehicle_evidence: float
    reason: str


@dataclass(frozen=True)
class VehicleGunResolverConfig:
    """Initial hand-tuned policy; these values require real-sample tuning."""

    minimum_vehicle_label_score: float = 0.30
    vehicle_margin: float = 0.10
    decision_margin: float = 0.15
    minimum_teacher_score: float = 0.08
    strong_transient_score: float = 0.65


DEFAULT_VEHICLE_GUN_RESOLVER_CONFIG = VehicleGunResolverConfig()


def strong_road_vehicle_evidence(
    road_vehicle_score: float,
    gunshot_label_score: float,
    *,
    config: VehicleGunResolverConfig = DEFAULT_VEHICLE_GUN_RESOLVER_CONFIG,
) -> bool:
    """Apply the score-and-margin gate used for strong road-vehicle evidence."""

    road_vehicle_score = _clamp(road_vehicle_score)
    gunshot_label_score = _clamp(gunshot_label_score)
    return (
        road_vehicle_score >= float(config.minimum_vehicle_label_score)
        and road_vehicle_score >= gunshot_label_score + float(config.vehicle_margin)
    )


def resolve_vehicle_gun(
    evidence: VehicleGunEvidence,
    *,
    config: VehicleGunResolverConfig = DEFAULT_VEHICLE_GUN_RESOLVER_CONFIG,
) -> VehicleGunDecision:
    """Resolve gunshot, vehicle, or unknown from normalized evidence.

    Teacher scores carry most of the weight.  A strong short transient can
    correct a generic/non-road vehicle teacher error, while a road-vehicle
    label must pass both an absolute threshold and a margin over the gunshot
    label before it can suppress that correction.
    """

    gun_teacher = _clamp(evidence.gunshot_teacher_score)
    vehicle_teacher = _clamp(evidence.vehicle_teacher_score)
    gun_label = _clamp(evidence.gunshot_label_score)
    road_vehicle = _clamp(evidence.road_vehicle_label_score)
    transient = _clamp(evidence.transient_score)
    peak_score = _linear_score(evidence.peak, 0.04, 0.35)
    rms_presence = _linear_score(evidence.rms, 0.005, 0.12)
    crest_score = _linear_score(evidence.crest_factor, 6.0, 20.0)
    strong_vehicle = strong_road_vehicle_evidence(
        road_vehicle,
        gun_label,
        config=config,
    )

    gunshot_score = (
        0.60 * gun_teacher
        + 0.15 * gun_label
        + 0.20 * transient
        + 0.03 * crest_score
        + 0.02 * peak_score
    )
    vehicle_score = (
        0.60 * vehicle_teacher
        + 0.25 * road_vehicle
        + 0.15 * (1.0 - transient) * rms_presence
    )

    if strong_vehicle:
        vehicle_score += 0.10 * road_vehicle
        gunshot_score *= 0.85
    elif transient >= 0.35:
        # Correct sharp generic-mechanical/vehicle errors only when actual
        # road-vehicle evidence did not pass the score-and-margin gate.
        gunshot_score += 0.25 * transient * (1.0 - road_vehicle)
        vehicle_score *= 1.0 - 0.75 * transient

    gunshot_score = _clamp(gunshot_score)
    vehicle_score = _clamp(vehicle_score)
    margin = gunshot_score - vehicle_score
    strongest_teacher = max(gun_teacher, vehicle_teacher, gun_label, road_vehicle)
    strong_waveform_teacher_conflict = (
        transient >= float(config.strong_transient_score)
        and strong_vehicle
        and vehicle_teacher >= gun_teacher + float(config.vehicle_margin)
        and abs(margin) < 2.0 * float(config.decision_margin)
    )

    if strong_waveform_teacher_conflict:
        return VehicleGunDecision(
            label="unknown",
            confidence=_clamp(1.0 - abs(margin) / max(2.0 * config.decision_margin, 1.0e-9)),
            gunshot_evidence=gunshot_score,
            vehicle_evidence=vehicle_score,
            reason=(
                f"waveform/teacher conflict; margin={margin:.3f}; transient={transient:.3f}; "
                f"vehicle_teacher={vehicle_teacher:.3f}"
            ),
        )

    if strongest_teacher < float(config.minimum_teacher_score) and transient < float(config.strong_transient_score):
        return VehicleGunDecision(
            label="unknown",
            confidence=_clamp(1.0 - strongest_teacher / max(config.minimum_teacher_score, 1.0e-9)),
            gunshot_evidence=gunshot_score,
            vehicle_evidence=vehicle_score,
            reason=(
                f"low evidence; teacher={strongest_teacher:.3f}; transient={transient:.3f}; "
                f"margin={margin:.3f}"
            ),
        )

    decision_margin = max(float(config.decision_margin), 1.0e-9)
    if margin >= decision_margin:
        label = "gunshot"
    elif margin <= -decision_margin:
        label = "vehicle"
    else:
        label = "unknown"

    if label == "unknown":
        confidence = _clamp(1.0 - abs(margin) / decision_margin)
        reason_prefix = "ambiguous evidence"
    else:
        confidence = _clamp(0.5 + abs(margin) / 2.0)
        reason_prefix = f"{label} margin"

    return VehicleGunDecision(
        label=label,
        confidence=confidence,
        gunshot_evidence=gunshot_score,
        vehicle_evidence=vehicle_score,
        reason=(
            f"{reason_prefix}={margin:.3f}; gun={gunshot_score:.3f}; vehicle={vehicle_score:.3f}; "
            f"transient={transient:.3f}; road_vehicle_strong={'yes' if strong_vehicle else 'no'}"
        ),
    )
