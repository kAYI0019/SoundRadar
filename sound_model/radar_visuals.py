"""Qt-free pulse lifecycle and radar visual geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from sound_model.radar_directions import arc_start_deg_for_position


@dataclass(frozen=True)
class WatercolorBlob:
    angle_deg: float
    distance: float
    radius: float
    opacity: float
    stretch: float = 1.0
    rotation_deg: float = 0.0
    flow_deg: float = 0.0


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def pulse_age_ratio(pulse, now):
    if pulse.duration <= 0:
        return 1.0
    return clamp((now - pulse.created_at) / pulse.duration)


def pulse_opacity(pulse, now):
    age = pulse_age_ratio(pulse, now)
    if age >= 1.0:
        return 0.0
    return clamp(pulse.strength) * ((1.0 - age) ** 1.5)


def pulse_expired(pulse, now):
    return pulse_age_ratio(pulse, now) >= 1.0


def normalize_degrees(angle):
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def sector_mid_angle_deg(sector):
    return normalize_degrees(arc_start_deg_for_position(sector) + 15)


def event_icon_center_xy(
    sector,
    min_side,
    center_x,
    center_y,
    *,
    distance_ratio,
    lane_index=0,
    lane_count=1,
    lane_spacing_ratio,
):
    angle_rad = math.radians(sector_mid_angle_deg(sector))
    distance = float(min_side) * float(distance_ratio)
    lane_count = max(1, int(lane_count))
    lane_index = clamp(float(lane_index), 0.0, float(lane_count - 1))
    lane_offset = (lane_index - (lane_count - 1) * 0.5) * float(min_side) * float(lane_spacing_ratio)
    return (
        float(center_x) + math.cos(angle_rad) * distance - math.sin(angle_rad) * lane_offset,
        float(center_y) - math.sin(angle_rad) * distance - math.cos(angle_rad) * lane_offset,
    )


def event_icon_size(
    pulse,
    now,
    min_side,
    *,
    min_size_ratio,
    max_size_ratio,
    pop_age_ratio,
    pop_scale,
    size_scale,
):
    strength = clamp(float(getattr(pulse, "strength", 0.0)))
    base_ratio = float(min_size_ratio) + (float(max_size_ratio) - float(min_size_ratio)) * math.sqrt(strength)
    age = pulse_age_ratio(pulse, now)
    pop = 1.0 + float(pop_scale) * max(0.0, 1.0 - age / float(pop_age_ratio))
    return float(min_side) * base_ratio * pop * float(size_scale)


def event_icon_opacity(pulse, now, *, hold_age_ratio):
    age = pulse_age_ratio(pulse, now)
    if age >= 1.0:
        return 0.0
    visibility = 0.68 + 0.32 * clamp(float(getattr(pulse, "strength", 0.0)))
    if age <= float(hold_age_ratio):
        return visibility
    fade = (1.0 - age) / max(1e-6, 1.0 - float(hold_age_ratio))
    return visibility * (fade ** 1.25)


def pulse_ripple_radius(pulse, now, min_side):
    age = pulse_age_ratio(pulse, now)
    return float(min_side) * (0.38 + 0.16 * age)


def watercolor_pulse_seed(pulse):
    strength_key = int(round(clamp(float(pulse.strength)) * 1000))
    time_key = int(round(float(pulse.created_at) * 1000))
    return ((int(pulse.sector) + 1) * 1_000_003 + time_key * 9_176 + strength_key * 37) & 0xFFFFFFFF


def watercolor_blob_specs(
    pulse,
    now,
    min_side,
    *,
    blob_count,
    angle_spread,
    inner_safe_ratio,
    flow_drift_deg,
):
    age = pulse_age_ratio(pulse, now)
    strength = clamp(float(pulse.strength))
    opacity = pulse_opacity(pulse, now)
    base_distance = max(
        pulse_ripple_radius(pulse, now, min_side),
        float(min_side) * float(inner_safe_ratio),
    )
    center_angle = sector_mid_angle_deg(pulse.sector)
    rng = random.Random(watercolor_pulse_seed(pulse))
    specs = []
    for index in range(int(blob_count)):
        angle_offset = rng.uniform(-float(angle_spread), float(angle_spread))
        flow_deg = rng.uniform(-float(flow_drift_deg), float(flow_drift_deg))
        distance = base_distance + float(min_side) * (0.006 * index + rng.uniform(0.0, 0.026))
        radius = float(min_side) * rng.uniform(0.038, 0.078) * (0.82 + 0.58 * strength) * (0.9 + 0.28 * age)
        blob_opacity = clamp(opacity * rng.uniform(0.52, 0.95))
        specs.append(
            WatercolorBlob(
                angle_deg=normalize_degrees(center_angle + angle_offset + flow_deg * (age ** 1.2)),
                distance=distance,
                radius=radius,
                opacity=blob_opacity,
                stretch=rng.uniform(1.25, 2.15),
                rotation_deg=normalize_degrees(center_angle + 90 + flow_deg * 0.65 + rng.uniform(-22, 22)),
                flow_deg=flow_deg,
            )
        )
    return specs


def watercolor_color_level(strength):
    return clamp(float(strength)) ** 2
