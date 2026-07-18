"""Pure radar direction and channel-mapping helpers for SoundRadar."""

from __future__ import annotations

from typing import Mapping

import numpy as np

RADAR_SECTORS = 12

FRONT_LEFT = "avg"
FRONT_RIGHT = "avd"
CENTER = "c"
LEFT = "g"
RIGHT = "d"
REAR_LEFT = "arg"
REAR_RIGHT = "ard"

DEFAULT_DIRECTIONAL_MIN_RATIO = 0.01


def arc_start_deg_for_position(position: int) -> int:
    """Return QPainter.drawArc start degrees for clock-like radar positions.

    QPainter uses 0 degrees at 3 o'clock and positive angles counter-clockwise.
    Radar positions are clock-like: 0=front/top, 3=right, 6=rear/bottom, 9=left.
    """
    angle = 75 - position * 30
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def build_channel_mapping(
    channel_count: int,
    output_channel_count: int | None = None,
) -> tuple[dict[str, int | None], str]:
    if output_channel_count is not None and output_channel_count < 8 and channel_count >= 2:
        return {
            FRONT_LEFT: 0,
            FRONT_RIGHT: 1,
            CENTER: None,
            RIGHT: 1,
            LEFT: 0,
            REAR_LEFT: 0,
            REAR_RIGHT: 1,
        }, "stereo fallback (2-channel output)"
    if channel_count >= 8:
        return {
            FRONT_LEFT: 0,
            FRONT_RIGHT: 1,
            CENTER: 2,
            RIGHT: 5,
            LEFT: 4,
            REAR_LEFT: 6,
            REAR_RIGHT: 7,
        }, "7.1 surround"
    if channel_count >= 2:
        return {
            FRONT_LEFT: 0,
            FRONT_RIGHT: 1,
            CENTER: None,
            RIGHT: 1,
            LEFT: 0,
            REAR_LEFT: 0,
            REAR_RIGHT: 1,
        }, "stereo fallback"
    if channel_count == 1:
        return {
            FRONT_LEFT: 0,
            FRONT_RIGHT: 0,
            CENTER: 0,
            RIGHT: 0,
            LEFT: 0,
            REAR_LEFT: 0,
            REAR_RIGHT: 0,
        }, "mono fallback"
    raise ValueError("Selected device has no input channels.")


def positive_difference(primary: float, secondary: float) -> float:
    return max(0.0, float(primary) - float(secondary))


def directional_difference(
    primary: float,
    secondary: float,
    min_ratio: float = DEFAULT_DIRECTIONAL_MIN_RATIO,
) -> float:
    diff = positive_difference(primary, secondary)
    if diff <= 0:
        return 0.0
    baseline = min(float(primary), float(secondary))
    if baseline <= 0:
        return diff
    return diff if diff / baseline > float(min_ratio) else 0.0


def centered_pair_strength(first: float, second: float, balance_tolerance: float = 0.25) -> float:
    strongest = max(float(first), float(second))
    if strongest <= 0:
        return 0.0
    if abs(float(first) - float(second)) / strongest > float(balance_tolerance):
        return 0.0
    return (float(first) + float(second)) / 2


def mapped_channel_value(values, channel_map: Mapping[str, int | None], key: str) -> float:
    index = channel_map.get(key)
    if index is None:
        return 0.0
    return float(values[index])


def compute_direction_levels(
    max_values,
    channel_map: Mapping[str, int | None],
    min_ratio: float = DEFAULT_DIRECTIONAL_MIN_RATIO,
) -> np.ndarray:
    """Map channel peak values to the 12 visual radar sectors."""
    values = np.asarray(max_values)
    front_left = mapped_channel_value(values, channel_map, FRONT_LEFT)
    front_right = mapped_channel_value(values, channel_map, FRONT_RIGHT)
    center = mapped_channel_value(values, channel_map, CENTER)
    left = mapped_channel_value(values, channel_map, LEFT)
    right = mapped_channel_value(values, channel_map, RIGHT)
    rear_left = mapped_channel_value(values, channel_map, REAR_LEFT)
    rear_right = mapped_channel_value(values, channel_map, REAR_RIGHT)

    levels = np.zeros(RADAR_SECTORS, dtype=float)
    levels[0] = max(center, centered_pair_strength(front_left, front_right))
    levels[1] = directional_difference(front_right, front_left, min_ratio)
    levels[2] = centered_pair_strength(front_right, right)
    levels[3] = right
    levels[4] = centered_pair_strength(right, rear_right)
    levels[5] = directional_difference(rear_right, rear_left, min_ratio)
    levels[6] = centered_pair_strength(rear_left, rear_right)
    levels[7] = directional_difference(rear_left, rear_right, min_ratio)
    levels[8] = centered_pair_strength(rear_left, left)
    levels[9] = left
    levels[10] = centered_pair_strength(left, front_left)
    levels[11] = directional_difference(front_left, front_right, min_ratio)
    return levels
