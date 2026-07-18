"""Explainable vehicle-versus-gunshot resolution for SoundRadar."""

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


def _short_score(value: float, full: float, zero: float) -> float:
    """Return one at or below ``full`` and zero at or above ``zero``."""

    value = float(value)
    if not math.isfinite(value) or value >= zero:
        return 0.0
    if value <= full:
        return 1.0
    return (zero - value) / (zero - full)


@dataclass(frozen=True)
class VehicleGunEvidence:
    """Normalized teacher/waveform evidence and timing values in milliseconds.

    New fields default to zero so older callers remain valid.  Acoustic feature
    weighting is enabled only when at least one spectral/envelope field carries
    signal, preventing a legacy all-default construction from looking like an
    unrealistically instantaneous event.
    """

    gunshot_teacher_score: float = 0.0
    vehicle_teacher_score: float = 0.0
    gunshot_label_score: float = 0.0
    road_vehicle_label_score: float = 0.0
    transient_score: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    crest_factor: float = 0.0
    spectral_flux: float = 0.0
    low_frequency_ratio: float = 0.0
    mid_frequency_ratio: float = 0.0
    high_frequency_ratio: float = 0.0
    attack_time_ms: float = 0.0
    peak_hold_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    onset_duration_ms: float = 0.0
    energy_concentration: float = 0.0
    vehicle_persistence: float = 0.0


@dataclass(frozen=True)
class VehicleGunDecision:
    label: str
    confidence: float
    gunshot_evidence: float
    vehicle_evidence: float
    reason: str


@dataclass(frozen=True)
class VehicleGunResolverConfig:
    """Central thresholds and weights; values require real-sample calibration."""

    minimum_vehicle_label_score: float = 0.30
    vehicle_margin: float = 0.10
    decision_margin: float = 0.15
    minimum_teacher_score: float = 0.08
    strong_transient_score: float = 0.65
    transient_correction_threshold: float = 0.35
    peak_start: float = 0.04
    peak_full: float = 0.35
    rms_start: float = 0.005
    rms_full: float = 0.12
    spectral_flux_start: float = 0.20
    spectral_flux_full: float = 0.70
    crest_factor_start: float = 6.0
    crest_factor_full: float = 20.0
    fast_attack_full_ms: float = 15.0
    fast_attack_zero_ms: float = 120.0
    short_onset_full_ms: float = 80.0
    short_onset_zero_ms: float = 450.0
    concentration_start: float = 0.15
    concentration_full: float = 0.70
    low_frequency_start: float = 0.35
    low_frequency_full: float = 0.80
    sustained_onset_start_ms: float = 120.0
    sustained_onset_full_ms: float = 700.0
    persistence_start: float = 0.40
    persistence_full: float = 1.0
    gun_acoustic_weight: float = 0.18
    vehicle_acoustic_weight: float = 0.14
    vehicle_persistence_weight: float = 0.12
    persistence_gun_suppression: float = 0.12
    gun_teacher_weight: float = 0.60
    gun_label_weight: float = 0.15
    gun_transient_weight: float = 0.20
    gun_crest_weight: float = 0.03
    gun_peak_weight: float = 0.02
    vehicle_teacher_weight: float = 0.60
    vehicle_label_weight: float = 0.25
    vehicle_sustained_rms_weight: float = 0.15
    acoustic_flux_weight: float = 0.30
    acoustic_crest_weight: float = 0.20
    acoustic_fast_attack_weight: float = 0.15
    acoustic_short_onset_weight: float = 0.15
    acoustic_concentration_weight: float = 0.20
    acoustic_low_frequency_weight: float = 0.45
    acoustic_sustained_onset_weight: float = 0.25
    acoustic_low_flux_weight: float = 0.15
    acoustic_low_concentration_weight: float = 0.15
    strong_vehicle_bonus_weight: float = 0.10
    strong_vehicle_gun_multiplier: float = 0.85
    transient_correction_weight: float = 0.25
    transient_vehicle_suppression: float = 0.75


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
    peak_score = _linear_score(evidence.peak, config.peak_start, config.peak_full)
    rms_presence = _linear_score(evidence.rms, config.rms_start, config.rms_full)
    crest_score = _linear_score(
        evidence.crest_factor,
        config.crest_factor_start,
        config.crest_factor_full,
    )
    spectral_flux = _clamp(evidence.spectral_flux)
    low_frequency_ratio = _clamp(evidence.low_frequency_ratio)
    vehicle_persistence = _clamp(evidence.vehicle_persistence)
    acoustic_features_available = any(
        float(value) > 0.0
        for value in (
            evidence.spectral_flux,
            evidence.low_frequency_ratio,
            evidence.mid_frequency_ratio,
            evidence.high_frequency_ratio,
            evidence.attack_time_ms,
            evidence.peak_hold_time_ms,
            evidence.decay_time_ms,
            evidence.onset_duration_ms,
            evidence.energy_concentration,
        )
    )
    flux_score = _linear_score(spectral_flux, config.spectral_flux_start, config.spectral_flux_full)
    fast_attack_score = _short_score(
        evidence.attack_time_ms,
        config.fast_attack_full_ms,
        config.fast_attack_zero_ms,
    )
    short_onset_score = _short_score(
        evidence.onset_duration_ms,
        config.short_onset_full_ms,
        config.short_onset_zero_ms,
    )
    concentration_score = _linear_score(
        evidence.energy_concentration,
        config.concentration_start,
        config.concentration_full,
    )
    low_frequency_score = _linear_score(
        low_frequency_ratio,
        config.low_frequency_start,
        config.low_frequency_full,
    )
    sustained_onset_score = _linear_score(
        evidence.onset_duration_ms,
        config.sustained_onset_start_ms,
        config.sustained_onset_full_ms,
    )
    persistence_score = _linear_score(
        vehicle_persistence,
        config.persistence_start,
        config.persistence_full,
    )
    gun_acoustic_score = (
        config.acoustic_flux_weight * flux_score
        + config.acoustic_crest_weight * crest_score
        + config.acoustic_fast_attack_weight * fast_attack_score
        + config.acoustic_short_onset_weight * short_onset_score
        + config.acoustic_concentration_weight * concentration_score
    ) if acoustic_features_available else 0.0
    vehicle_acoustic_score = (
        config.acoustic_low_frequency_weight * low_frequency_score
        + config.acoustic_sustained_onset_weight * sustained_onset_score
        + config.acoustic_low_flux_weight * (1.0 - flux_score)
        + config.acoustic_low_concentration_weight * (1.0 - concentration_score)
    ) if acoustic_features_available else 0.0
    strong_vehicle = strong_road_vehicle_evidence(
        road_vehicle,
        gun_label,
        config=config,
    )

    gunshot_score = (
        config.gun_teacher_weight * gun_teacher
        + config.gun_label_weight * gun_label
        + config.gun_transient_weight * transient
        + config.gun_crest_weight * crest_score
        + config.gun_peak_weight * peak_score
    )
    vehicle_score = (
        config.vehicle_teacher_weight * vehicle_teacher
        + config.vehicle_label_weight * road_vehicle
        + config.vehicle_sustained_rms_weight * (1.0 - transient) * rms_presence
    )

    if acoustic_features_available:
        gunshot_score += float(config.gun_acoustic_weight) * gun_acoustic_score * (1.0 - road_vehicle)
        vehicle_score += float(config.vehicle_acoustic_weight) * vehicle_acoustic_score * (1.0 - transient)
        # Persistence is corroboration rather than a standalone vehicle rule.
        vehicle_score += (
            float(config.vehicle_persistence_weight)
            * persistence_score
            * max(vehicle_teacher, road_vehicle)
        )
        gunshot_score *= 1.0 - config.persistence_gun_suppression * persistence_score

    if strong_vehicle:
        vehicle_score += config.strong_vehicle_bonus_weight * road_vehicle
        gunshot_score *= config.strong_vehicle_gun_multiplier
    elif transient >= config.transient_correction_threshold:
        # Correct sharp generic-mechanical/vehicle errors only when actual
        # road-vehicle evidence did not pass the score-and-margin gate.
        gunshot_score += config.transient_correction_weight * transient * (1.0 - road_vehicle)
        vehicle_score *= 1.0 - config.transient_vehicle_suppression * transient

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
                f"flux={spectral_flux:.3f}; persistence={vehicle_persistence:.3f}; "
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
                f"flux={spectral_flux:.3f}; persistence={vehicle_persistence:.3f}; margin={margin:.3f}"
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
            f"transient={transient:.3f}; flux={spectral_flux:.3f}; low={low_frequency_ratio:.3f}; "
            f"persistence={vehicle_persistence:.3f}; "
            f"road_vehicle_strong={'yes' if strong_vehicle else 'no'}"
        ),
    )


def apply_vehicle_gun_decision(
    raw_scores: dict[str, float],
    evidence: VehicleGunEvidence,
    decision: VehicleGunDecision,
) -> dict[str, float]:
    """Apply a resolver decision while preserving unrelated event scores."""

    adjusted = dict(raw_scores)
    if decision.label == "gunshot":
        adjusted["gunshot"] = max(
            float(raw_scores.get("gunshot", 0.0)),
            float(decision.gunshot_evidence),
            float(evidence.transient_score),
        )
        adjusted["vehicle"] = 0.0
    elif decision.label == "vehicle":
        adjusted["gunshot"] = 0.0
        adjusted["vehicle"] = max(
            float(raw_scores.get("vehicle", 0.0)),
            float(decision.vehicle_evidence),
        )
    else:
        adjusted["gunshot"] = 0.0
        adjusted["vehicle"] = 0.0
    return adjusted
