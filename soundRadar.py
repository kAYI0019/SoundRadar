import ctypes
import ctypes.util
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets

from sound_model.event_debug import (
    DIRECTION_DEBUG_LABELS,
    EVENT_DEBUG_LABELS,
    DirectionEventDebugConfig,
    compact_score as _compact_score,
    direction_event_debug_cell as _direction_event_debug_cell,
    direction_event_debug_header as _direction_event_debug_header,
    direction_event_debug_lines as _direction_event_debug_lines,
    direction_event_device_label as _direction_event_device_label,
    direction_event_gunshot_debug_line as _direction_event_gunshot_debug_line,
    direction_latency_debug_line as _direction_latency_debug_line,
    format_latency_ms as _format_latency_ms,
)
from sound_model.event_detection import (
    CLASSIFIED_EVENT_KINDS,
    DEFAULT_DIRECTION_EVENT_COOLDOWN,
    DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS,
    DEFAULT_DIRECTION_EVENT_THRESHOLD,
    DEFAULT_EVENT_ICON_DURATION,
    DEFAULT_GUNSHOT_BIAS_MARGIN,
    DEFAULT_GUNSHOT_GLOBAL_COOLDOWN,
    DEFAULT_GUNSHOT_MAX_DIRECTIONS,
    DEFAULT_MAX_EVENTS_PER_DIRECTION,
    DEFAULT_RIPPLE_COOLDOWN,
    DEFAULT_RIPPLE_DURATION,
    DEFAULT_RIPPLE_THRESHOLD,
    DEFAULT_SMOOTHING_WINDOW,
    DIRECTION_EVENT_ORDER,
    DIRECTION_EVENT_ORDER_INDEX,
    DIRECTION_EVENT_PRIORITY,
    DIRECTION_EVENT_SECTORS,
    GUNSHOT_DIRECTION_NEIGHBORS,
    DirectionEventPulseDebug,
    GunshotDisplayDecision,
    SmoothedDirectionEventPrediction,
    SoundPulse,
    active_event_candidates as _active_event_candidates,
    classify_basic_sound_event,
    create_pulses_from_direction_events as _create_pulses_from_direction_events,
    create_pulses_from_levels as _create_pulses_from_levels,
    displayed_active_events as _displayed_active_events,
    dominant_active_event as _dominant_active_event,
    event_display_threshold,
    gunshot_candidate_scores as _gunshot_candidate_scores,
    gunshot_display_decision as _gunshot_display_decision,
    should_emit_pulse,
    smooth_direction_event_predictions as _smooth_direction_event_predictions,
    spatially_allowed_gunshot_directions as _spatially_allowed_gunshot_directions,
    suppress_displayed_events_for_direction,
)
from sound_model.direction_runtime import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DirectionEventRuntime as _DirectionEventRuntime,
    normalize_analysis_timing,
)
from sound_model.radar_directions import (
    CENTER,
    DEFAULT_DIRECTIONAL_MIN_RATIO,
    FRONT_LEFT,
    FRONT_RIGHT,
    LEFT,
    RADAR_SECTORS,
    REAR_LEFT,
    REAR_RIGHT,
    RIGHT,
    arc_start_deg_for_position,
    build_channel_mapping,
    centered_pair_strength,
    compute_direction_levels as _compute_direction_levels,
    directional_difference as _directional_difference,
    mapped_channel_value,
    positive_difference,
)
from sound_model.radar_visuals import (
    WatercolorBlob,
    event_icon_center_xy as _event_icon_center_xy,
    event_icon_opacity as _event_icon_opacity,
    event_icon_size as _event_icon_size,
    normalize_degrees,
    pulse_age_ratio,
    pulse_expired,
    pulse_opacity,
    pulse_ripple_radius,
    sector_mid_angle_deg,
    watercolor_blob_specs as _watercolor_blob_specs,
    watercolor_color_level,
    watercolor_pulse_seed,
)
from sound_model.rolling_capture import (
    RollingAudioCapture,
    RollingCaptureSnapshot,
    TimedAudioBlock,
    consume_rolling_capture_trigger as _consume_rolling_capture_trigger,
    rolling_capture_metadata_path as _rolling_capture_metadata_path,
    rolling_capture_metadata_payload as _rolling_capture_metadata_payload,
    rolling_capture_output_path as _rolling_capture_output_path,
    write_rolling_capture_snapshot as _write_rolling_capture_snapshot,
    write_rolling_capture_trigger as _write_rolling_capture_trigger,
)

# GLOBAL PARAMETERS (kept as simple module settings for easy tuning)
n_chans = 8
n_channel = n_chans
maxSoundValue = 1.0  # float32 streams are normalized to -1.0..1.0

STRENGTH_MODE = 2
minTFU = 0.5  # minimum Time needed for First Update (upper sound value)
minTBU = 0.1  # minimum Time needed Between Update (lower sound value)
maxdifmain = DEFAULT_DIRECTIONAL_MIN_RATIO  # min ratio difference between paired directional channels
maxColorRange = 255
minThreshold = 0.005
prevmax = np.zeros(RADAR_SECTORS)
refreshtime = 0.1
fade_decay_rate = 2.0

# Visualization settings
size_multiplier = 15.0
opacity_multiplier = 0.7
ARC_OPACITY_MULTIPLIER = 0.28
ARC_MIN_VISIBLE_STRENGTH = 0.02
SHOW_ARCS = True
SHOW_RIPPLES = True
RIPPLE_STYLE = "watercolor"
RIPPLE_THRESHOLD = DEFAULT_RIPPLE_THRESHOLD
RIPPLE_COOLDOWN = DEFAULT_RIPPLE_COOLDOWN
RIPPLE_DURATION = DEFAULT_RIPPLE_DURATION
MAX_ACTIVE_PULSES = 72
WATERCOLOR_BLOBS = 7
WATERCOLOR_ANGLE_SPREAD = 26.0
WATERCOLOR_INNER_SAFE_RATIO = 0.44
WATERCOLOR_ALPHA_MULTIPLIER = 0.78
WATERCOLOR_SOFT_SCALE = 1.85
WATERCOLOR_PIGMENT_SCALE = 1.05
WATERCOLOR_FLOW_TRAILS = 3
WATERCOLOR_FLOW_DRIFT_DEG = 9.0
WATERCOLOR_TRAIL_GAP_RATIO = 0.024
DEBUG = False

# Optional no-training AudioSet teacher bridge. The model loads lazily in a
# background worker so the overlay can keep painting while inference runs.
ENABLE_AST_DIRECTION_EVENTS = True
AST_DIRECTION_EVENT_DEVICE = "auto"
AST_DIRECTION_EVENT_DTYPE = "auto"
AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION = None
AST_DIRECTION_EVENT_COMPILE = False
AST_DIRECTION_EVENT_TEACHER_MODEL = "efficientat-mn20"
# efficientat-mn10 efficientat-mn20 ast
AST_DIRECTION_EVENT_MODEL_ID = None
AST_DIRECTION_EVENT_TOP_K = 5
AST_DIRECTION_EVENT_WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS
AST_DIRECTION_EVENT_INTERVAL = DEFAULT_INTERVAL_SECONDS
AST_DIRECTION_EVENT_THRESHOLD = DEFAULT_DIRECTION_EVENT_THRESHOLD
AST_DIRECTION_EVENT_COOLDOWN = DEFAULT_DIRECTION_EVENT_COOLDOWN
AST_DIRECTION_EVENT_WARMUP = True
SHOW_EVENT_DEBUG_TEXT = True
EVENT_DEBUG_MAX_LINES = 7
DIRECTION_EVENT_DISPLAY_THRESHOLDS = dict(DEFAULT_DIRECTION_EVENT_DISPLAY_THRESHOLDS)
DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS = dict(DEFAULT_DIRECTION_EVENT_SECONDARY_THRESHOLDS)
DIRECTION_EVENT_GUNSHOT_DISPLAY_THRESHOLD = DIRECTION_EVENT_DISPLAY_THRESHOLDS["gunshot"]
DIRECTION_EVENT_GUNSHOT_SECONDARY_THRESHOLD = DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS["gunshot"]
DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN = DEFAULT_GUNSHOT_BIAS_MARGIN
DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION = DEFAULT_MAX_EVENTS_PER_DIRECTION
DIRECTION_EVENT_SMOOTHING_ENABLED = True
DIRECTION_EVENT_SMOOTHING_WINDOW = DEFAULT_SMOOTHING_WINDOW

# Gunshot direction tuning: keep detection permissive, then suppress spatial
# bleed and rapid repeats at display time so distant shots are still visible.
GUNSHOT_SPATIAL_MAX_DIRECTIONS = DEFAULT_GUNSHOT_MAX_DIRECTIONS
GUNSHOT_GLOBAL_COOLDOWN = DEFAULT_GUNSHOT_GLOBAL_COOLDOWN

ROLLING_CAPTURE_ENABLED = True
ROLLING_CAPTURE_SECONDS = 5.0
ROLLING_CAPTURE_DIR = str(Path.home() / "SoundRadarSamples" / "rolling")
ROLLING_CAPTURE_TRIGGER_PATH = "/tmp/soundradar-save-rolling"

RUNTIME_CONFIG_FILENAME = "soundradar.local.json"
RUNTIME_CONFIG_ENV = "SOUNDRADAR_CONFIG"
THRESHOLD_PROFILE_ENV = "SOUNDRADAR_THRESHOLD_PROFILE"
RUNTIME_CONFIG_KEYS = frozenset(
    (
        "schema_version",
        "enable_direction_events",
        "teacher_model",
        "model_id",
        "device",
        "dtype",
        "attn_implementation",
        "compile_model",
        "top_k",
        "window_seconds",
        "interval_seconds",
        "warmup",
        "threshold_profile",
        "rolling_capture_enabled",
        "rolling_capture_seconds",
        "rolling_capture_dir",
        "rolling_capture_trigger_path",
        "event_icon_labels",
        "event_icon_scale",
        "event_icon_opacity",
        "event_smoothing_enabled",
        "event_smoothing_window",
    )
)


def default_runtime_config_path():
    return Path(__file__).with_name(RUNTIME_CONFIG_FILENAME)


def runtime_config_path(environ=None):
    environ = os.environ if environ is None else environ
    path = environ.get(RUNTIME_CONFIG_ENV)
    return Path(path).expanduser() if path else default_runtime_config_path()


def load_runtime_config(path=None, environ=None):
    path = runtime_config_path(environ) if path is None else Path(path).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    unknown = sorted(set(data) - RUNTIME_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown runtime config keys: {', '.join(unknown)}")
    return dict(data)


def _config_bool(value, key):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"{key} must be a boolean")


def _config_int(value, key):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _config_float(value, key):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def _config_positive_float(value, key):
    parsed = _config_float(value, key)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _config_positive_int(value, key):
    parsed = _config_int(value, key)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _config_optional_str(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _config_compile_model(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off", ""):
            return False
        return value.strip()
    raise ValueError("compile_model must be a boolean or torch.compile mode string")


def apply_runtime_config(config):
    global ENABLE_AST_DIRECTION_EVENTS, AST_DIRECTION_EVENT_DEVICE, AST_DIRECTION_EVENT_DTYPE
    global AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION, AST_DIRECTION_EVENT_COMPILE
    global AST_DIRECTION_EVENT_TEACHER_MODEL, AST_DIRECTION_EVENT_MODEL_ID, AST_DIRECTION_EVENT_TOP_K
    global AST_DIRECTION_EVENT_WINDOW_SECONDS, AST_DIRECTION_EVENT_INTERVAL, AST_DIRECTION_EVENT_WARMUP
    global ROLLING_CAPTURE_ENABLED, ROLLING_CAPTURE_SECONDS, ROLLING_CAPTURE_DIR, ROLLING_CAPTURE_TRIGGER_PATH
    global EVENT_ICON_SHOW_LABELS, EVENT_ICON_SIZE_SCALE, EVENT_ICON_ALPHA_SCALE
    global DIRECTION_EVENT_SMOOTHING_ENABLED, DIRECTION_EVENT_SMOOTHING_WINDOW

    config = dict(config or {})
    unknown = sorted(set(config) - RUNTIME_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown runtime config keys: {', '.join(unknown)}")

    if "enable_direction_events" in config:
        ENABLE_AST_DIRECTION_EVENTS = _config_bool(config["enable_direction_events"], "enable_direction_events")
    if "teacher_model" in config:
        AST_DIRECTION_EVENT_TEACHER_MODEL = str(config["teacher_model"])
    if "model_id" in config:
        AST_DIRECTION_EVENT_MODEL_ID = _config_optional_str(config["model_id"])
    if "device" in config:
        AST_DIRECTION_EVENT_DEVICE = str(config["device"])
    if "dtype" in config:
        AST_DIRECTION_EVENT_DTYPE = str(config["dtype"])
    if "attn_implementation" in config:
        AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION = _config_optional_str(config["attn_implementation"])
    if "compile_model" in config:
        AST_DIRECTION_EVENT_COMPILE = _config_compile_model(config["compile_model"])
    if "top_k" in config:
        AST_DIRECTION_EVENT_TOP_K = _config_int(config["top_k"], "top_k")
    if "window_seconds" in config:
        AST_DIRECTION_EVENT_WINDOW_SECONDS = _config_float(config["window_seconds"], "window_seconds")
    if "interval_seconds" in config:
        AST_DIRECTION_EVENT_INTERVAL = _config_float(config["interval_seconds"], "interval_seconds")
    AST_DIRECTION_EVENT_WINDOW_SECONDS, AST_DIRECTION_EVENT_INTERVAL = normalize_analysis_timing(
        AST_DIRECTION_EVENT_WINDOW_SECONDS,
        AST_DIRECTION_EVENT_INTERVAL,
    )
    if "warmup" in config:
        AST_DIRECTION_EVENT_WARMUP = _config_bool(config["warmup"], "warmup")
    if "rolling_capture_enabled" in config:
        ROLLING_CAPTURE_ENABLED = _config_bool(config["rolling_capture_enabled"], "rolling_capture_enabled")
    if "rolling_capture_seconds" in config:
        ROLLING_CAPTURE_SECONDS = _config_float(config["rolling_capture_seconds"], "rolling_capture_seconds")
    if "rolling_capture_dir" in config:
        ROLLING_CAPTURE_DIR = str(Path(str(config["rolling_capture_dir"])).expanduser())
    if "rolling_capture_trigger_path" in config:
        ROLLING_CAPTURE_TRIGGER_PATH = str(Path(str(config["rolling_capture_trigger_path"])).expanduser())
    if "event_icon_labels" in config:
        EVENT_ICON_SHOW_LABELS = _config_bool(config["event_icon_labels"], "event_icon_labels")
    if "event_icon_scale" in config:
        EVENT_ICON_SIZE_SCALE = _config_positive_float(config["event_icon_scale"], "event_icon_scale")
    if "event_icon_opacity" in config:
        EVENT_ICON_ALPHA_SCALE = _config_positive_float(config["event_icon_opacity"], "event_icon_opacity")
    if "event_smoothing_enabled" in config:
        DIRECTION_EVENT_SMOOTHING_ENABLED = _config_bool(config["event_smoothing_enabled"], "event_smoothing_enabled")
    if "event_smoothing_window" in config:
        DIRECTION_EVENT_SMOOTHING_WINDOW = _config_positive_int(config["event_smoothing_window"], "event_smoothing_window")
    return config


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    ripple_threshold: float
    ripple_cooldown: float
    direction_event_threshold: float
    direction_event_cooldown: float
    gunshot_global_cooldown: float
    gunshot_spatial_max_directions: int
    direction_event_display_thresholds: dict
    direction_event_secondary_thresholds: dict
    show_event_debug_text: bool


def _threshold_dict(**overrides):
    values = dict(DIRECTION_EVENT_DISPLAY_THRESHOLDS)
    values.update(overrides)
    return values


def _secondary_threshold_dict(**overrides):
    values = dict(DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS)
    values.update(overrides)
    return values


THRESHOLD_PROFILES = {
    "default": ThresholdProfile(
        name="default",
        ripple_threshold=RIPPLE_THRESHOLD,
        ripple_cooldown=RIPPLE_COOLDOWN,
        direction_event_threshold=AST_DIRECTION_EVENT_THRESHOLD,
        direction_event_cooldown=AST_DIRECTION_EVENT_COOLDOWN,
        gunshot_global_cooldown=GUNSHOT_GLOBAL_COOLDOWN,
        gunshot_spatial_max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
        direction_event_display_thresholds=dict(DIRECTION_EVENT_DISPLAY_THRESHOLDS),
        direction_event_secondary_thresholds=dict(DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS),
        show_event_debug_text=SHOW_EVENT_DEBUG_TEXT,
    ),
    "quiet": ThresholdProfile(
        name="quiet",
        ripple_threshold=0.045,
        ripple_cooldown=0.26,
        direction_event_threshold=0.16,
        direction_event_cooldown=0.75,
        gunshot_global_cooldown=0.30,
        gunshot_spatial_max_directions=1,
        direction_event_display_thresholds=_threshold_dict(gunshot=0.16, explosion=0.90, vehicle=0.70, footstep=0.90),
        direction_event_secondary_thresholds=_secondary_threshold_dict(gunshot=0.48, explosion=0.90, vehicle=0.70, footstep=0.90),
        show_event_debug_text=SHOW_EVENT_DEBUG_TEXT,
    ),
    "aggressive": ThresholdProfile(
        name="aggressive",
        ripple_threshold=0.020,
        ripple_cooldown=0.12,
        direction_event_threshold=0.07,
        direction_event_cooldown=0.38,
        gunshot_global_cooldown=0.12,
        gunshot_spatial_max_directions=2,
        direction_event_display_thresholds=_threshold_dict(gunshot=0.07, explosion=0.72, vehicle=0.48, footstep=0.72),
        direction_event_secondary_thresholds=_secondary_threshold_dict(gunshot=0.25, explosion=0.72, vehicle=0.48, footstep=0.72),
        show_event_debug_text=SHOW_EVENT_DEBUG_TEXT,
    ),
    "debug": ThresholdProfile(
        name="debug",
        ripple_threshold=0.015,
        ripple_cooldown=0.08,
        direction_event_threshold=0.05,
        direction_event_cooldown=0.20,
        gunshot_global_cooldown=0.05,
        gunshot_spatial_max_directions=3,
        direction_event_display_thresholds=_threshold_dict(gunshot=0.05, explosion=0.55, vehicle=0.35, footstep=0.55),
        direction_event_secondary_thresholds=_secondary_threshold_dict(gunshot=0.18, explosion=0.55, vehicle=0.35, footstep=0.55),
        show_event_debug_text=True,
    ),
}


def threshold_profile_names():
    return tuple(THRESHOLD_PROFILES)


def threshold_profile(name=None):
    profile_name = (name or "default").strip().lower()
    try:
        return THRESHOLD_PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(threshold_profile_names())
        raise ValueError(f"unknown threshold profile: {name}; choose one of: {choices}") from exc


def parse_threshold_profile_args(argv, environ=None, runtime_config=None):
    environ = os.environ if environ is None else environ
    runtime_config = runtime_config or {}
    profile_name = runtime_config.get("threshold_profile", "default")
    profile_name = environ.get(THRESHOLD_PROFILE_ENV, profile_name)
    cleaned = []
    args = list(argv)
    index = 0
    while index < len(args):
        value = args[index]
        if value == "--threshold-profile":
            if index + 1 >= len(args):
                raise ValueError("--threshold-profile requires a value")
            profile_name = args[index + 1]
            index += 2
            continue
        if value.startswith("--threshold-profile="):
            profile_name = value.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned, threshold_profile(profile_name).name


def apply_threshold_profile(name=None):
    global RIPPLE_THRESHOLD, RIPPLE_COOLDOWN
    global AST_DIRECTION_EVENT_THRESHOLD, AST_DIRECTION_EVENT_COOLDOWN
    global DIRECTION_EVENT_GUNSHOT_DISPLAY_THRESHOLD, DIRECTION_EVENT_GUNSHOT_SECONDARY_THRESHOLD
    global GUNSHOT_GLOBAL_COOLDOWN, GUNSHOT_SPATIAL_MAX_DIRECTIONS, SHOW_EVENT_DEBUG_TEXT

    profile = threshold_profile(name)
    RIPPLE_THRESHOLD = profile.ripple_threshold
    RIPPLE_COOLDOWN = profile.ripple_cooldown
    AST_DIRECTION_EVENT_THRESHOLD = profile.direction_event_threshold
    AST_DIRECTION_EVENT_COOLDOWN = profile.direction_event_cooldown
    GUNSHOT_GLOBAL_COOLDOWN = profile.gunshot_global_cooldown
    GUNSHOT_SPATIAL_MAX_DIRECTIONS = profile.gunshot_spatial_max_directions
    DIRECTION_EVENT_DISPLAY_THRESHOLDS.clear()
    DIRECTION_EVENT_DISPLAY_THRESHOLDS.update(profile.direction_event_display_thresholds)
    DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS.clear()
    DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS.update(profile.direction_event_secondary_thresholds)
    DIRECTION_EVENT_GUNSHOT_DISPLAY_THRESHOLD = DIRECTION_EVENT_DISPLAY_THRESHOLDS["gunshot"]
    DIRECTION_EVENT_GUNSHOT_SECONDARY_THRESHOLD = DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS["gunshot"]
    SHOW_EVENT_DEBUG_TEXT = profile.show_event_debug_text
    return profile


EVENT_ICON_DURATION = DEFAULT_EVENT_ICON_DURATION
EVENT_ICON_HOLD_AGE_RATIO = 0.68
EVENT_ICON_DISTANCE_RATIO = 0.40
EVENT_ICON_STACK_SPACING_RATIO = 0.052
EVENT_ICON_MIN_SIZE_RATIO = 0.026
EVENT_ICON_MAX_SIZE_RATIO = 0.044
EVENT_ICON_POP_AGE_RATIO = 0.28
EVENT_ICON_POP_SCALE = 0.08
EVENT_ICON_SIZE_SCALE = 1.0
EVENT_ICON_ALPHA_SCALE = 1.0
EVENT_ICON_SHOW_LABELS = False
EVENT_ICON_RGBA = (238, 236, 220, 205)
EVENT_ICON_SHADOW_RGBA = (0, 0, 0, 120)
EVENT_ICON_BADGE_RGBA = (4, 8, 10, 96)
EVENT_KIND_RGBA = {
    "footstep": (78, 226, 118, 125),
    "gunshot": (255, 150, 70, 170),
    "vehicle": (76, 166, 255, 135),
    "explosion": (255, 90, 74, 180),
}
EVENT_ICON_LABELS = {
    "explosion": "BOOM",
    "gunshot": "GUN",
    "vehicle": "CAR",
    "footstep": "STEP",
}
q = queue.Queue()


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def create_pulses_from_levels(
    levels,
    now,
    threshold=RIPPLE_THRESHOLD,
    cooldown=RIPPLE_COOLDOWN,
    last_pulse_times=None,
    previous_levels=None,
):
    return _create_pulses_from_levels(
        levels,
        now,
        threshold,
        cooldown,
        last_pulse_times,
        previous_levels,
        pulse_duration=RIPPLE_DURATION,
    )


def dominant_active_event(
    scores,
    active_events,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
    return _dominant_active_event(
        scores,
        active_events,
        threshold,
        event_thresholds,
        gunshot_bias_margin=DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN,
    )


def active_event_candidates(
    scores,
    active_events,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
    return _active_event_candidates(scores, active_events, threshold, event_thresholds)


def displayed_active_events(
    scores,
    active_events,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    secondary_thresholds=DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS,
    max_events=DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION,
):
    return _displayed_active_events(
        scores,
        active_events,
        threshold,
        event_thresholds,
        secondary_thresholds,
        max_events,
        gunshot_bias_margin=DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN,
    )


def gunshot_candidate_scores(
    scores_by_direction,
    active_by_direction,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
    return _gunshot_candidate_scores(scores_by_direction, active_by_direction, threshold, event_thresholds)


def gunshot_display_decision(
    scores_by_direction,
    active_by_direction,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
):
    return _gunshot_display_decision(
        scores_by_direction,
        active_by_direction,
        threshold,
        event_thresholds,
        max_directions,
    )


def spatially_allowed_gunshot_directions(
    scores_by_direction,
    active_by_direction,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
):
    return _spatially_allowed_gunshot_directions(
        scores_by_direction,
        active_by_direction,
        threshold,
        event_thresholds,
        max_directions,
    )


def smooth_direction_event_predictions(predictions, *, window=DIRECTION_EVENT_SMOOTHING_WINDOW):
    return _smooth_direction_event_predictions(predictions, window=window)


def create_pulses_from_direction_events(
    prediction,
    now,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    cooldown=AST_DIRECTION_EVENT_COOLDOWN,
    last_pulse_times=None,
    last_global_event_times=None,
    display_debug=None,
):
    return _create_pulses_from_direction_events(
        prediction,
        now,
        threshold,
        cooldown,
        last_pulse_times,
        last_global_event_times,
        display_debug,
        event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
        secondary_thresholds=DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS,
        max_events=DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION,
        gunshot_bias_margin=DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN,
        gunshot_max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
        gunshot_global_cooldown=GUNSHOT_GLOBAL_COOLDOWN,
        event_icon_duration=EVENT_ICON_DURATION,
    )


def compact_score(score):
    return _compact_score(score)


def _direction_event_debug_config():
    return DirectionEventDebugConfig(
        event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
        secondary_thresholds=DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS,
        max_events=DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION,
        gunshot_bias_margin=DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN,
        gunshot_max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
    )


def direction_event_device_label(runtime=None, requested_device=None, resolved_device=None, resolved_dtype=None):
    return _direction_event_device_label(runtime, requested_device, resolved_device, resolved_dtype)


def direction_event_debug_header(
    status=None,
    requested_device=None,
    resolved_device=None,
    resolved_dtype=None,
    teacher_model=None,
):
    return _direction_event_debug_header(
        status,
        requested_device,
        resolved_device,
        resolved_dtype,
        teacher_model,
        default_teacher_model=AST_DIRECTION_EVENT_TEACHER_MODEL,
    )


def format_latency_ms(value):
    return _format_latency_ms(value)


def direction_latency_debug_line(radar_latency_ms=None, ast_latency_ms=None):
    return _direction_latency_debug_line(radar_latency_ms, ast_latency_ms)


def direction_event_debug_cell(direction, scores_by_direction, active_by_direction, threshold=AST_DIRECTION_EVENT_THRESHOLD):
    return _direction_event_debug_cell(
        direction,
        scores_by_direction,
        active_by_direction,
        threshold,
        config=_direction_event_debug_config(),
    )


def direction_event_gunshot_debug_line(
    prediction,
    display_debug=None,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
):
    return _direction_event_gunshot_debug_line(
        prediction,
        display_debug,
        threshold,
        config=_direction_event_debug_config(),
    )


def direction_event_debug_lines(
    prediction,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    max_lines=EVENT_DEBUG_MAX_LINES,
    status=None,
    requested_device=None,
    resolved_device=None,
    resolved_dtype=None,
    teacher_model=None,
    radar_latency_ms=None,
    ast_latency_ms=None,
    display_debug=None,
):
    return _direction_event_debug_lines(
        prediction,
        threshold,
        max_lines,
        status,
        requested_device,
        resolved_device,
        resolved_dtype,
        teacher_model,
        radar_latency_ms,
        ast_latency_ms,
        display_debug,
        config=_direction_event_debug_config(),
        default_teacher_model=AST_DIRECTION_EVENT_TEACHER_MODEL,
    )


def is_classified_event_kind(kind):
    return kind in CLASSIFIED_EVENT_KINDS


def event_icon_center(
    sector,
    min_side,
    center_x,
    center_y,
    distance_ratio=EVENT_ICON_DISTANCE_RATIO,
    lane_index=0,
    lane_count=1,
    lane_spacing_ratio=EVENT_ICON_STACK_SPACING_RATIO,
):
    x, y = _event_icon_center_xy(
        sector,
        min_side,
        center_x,
        center_y,
        distance_ratio=distance_ratio,
        lane_index=lane_index,
        lane_count=lane_count,
        lane_spacing_ratio=lane_spacing_ratio,
    )
    return QtCore.QPointF(x, y)


def event_icon_size(pulse, now, min_side):
    return _event_icon_size(
        pulse,
        now,
        min_side,
        min_size_ratio=EVENT_ICON_MIN_SIZE_RATIO,
        max_size_ratio=EVENT_ICON_MAX_SIZE_RATIO,
        pop_age_ratio=EVENT_ICON_POP_AGE_RATIO,
        pop_scale=EVENT_ICON_POP_SCALE,
        size_scale=EVENT_ICON_SIZE_SCALE,
    )


def event_icon_opacity(pulse, now):
    return _event_icon_opacity(pulse, now, hold_age_ratio=EVENT_ICON_HOLD_AGE_RATIO)


def event_icon_color(pulse, now):
    red, green, blue, alpha = EVENT_ICON_RGBA
    return QtGui.QColor(red, green, blue, int(clamp(alpha * event_icon_opacity(pulse, now) * EVENT_ICON_ALPHA_SCALE, 0, 255)))


def event_icon_shadow_color(pulse, now):
    red, green, blue, alpha = EVENT_ICON_SHADOW_RGBA
    return QtGui.QColor(red, green, blue, int(clamp(alpha * event_icon_opacity(pulse, now) * EVENT_ICON_ALPHA_SCALE, 0, 255)))


def event_icon_accent_color(pulse, now):
    red, green, blue, alpha = color_rgba_for_kind(pulse.kind, EVENT_ICON_RGBA)
    return QtGui.QColor(red, green, blue, int(clamp(alpha * event_icon_opacity(pulse, now) * EVENT_ICON_ALPHA_SCALE, 0, 255)))


def event_icon_badge_color(pulse, now):
    red, green, blue, alpha = EVENT_ICON_BADGE_RGBA
    return QtGui.QColor(red, green, blue, int(clamp(alpha * event_icon_opacity(pulse, now) * EVENT_ICON_ALPHA_SCALE, 0, 255)))


def event_icon_label(kind):
    return EVENT_ICON_LABELS.get(kind, str(kind or "?").upper()[:4])


def event_star_polygon(point_count, inner_radius, outer_radius, rotation_deg=90.0):
    points = []
    for index in range(point_count * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = math.radians(rotation_deg + index * 180.0 / point_count)
        points.append(QtCore.QPointF(math.cos(angle) * radius, -math.sin(angle) * radius))
    return QtGui.QPolygonF(points)


def watercolor_blob_specs(pulse, now, min_side, blob_count=WATERCOLOR_BLOBS):
    return _watercolor_blob_specs(
        pulse,
        now,
        min_side,
        blob_count=blob_count,
        angle_spread=WATERCOLOR_ANGLE_SPREAD,
        inner_safe_ratio=WATERCOLOR_INNER_SAFE_RATIO,
        flow_drift_deg=WATERCOLOR_FLOW_DRIFT_DEG,
    )


def color_rgba_for_kind(kind, fallback_rgba):
    rgba = EVENT_KIND_RGBA.get(kind)
    return rgba if rgba is not None else fallback_rgba


def watercolor_color(strength, opacity, kind="unknown"):
    _ = kind
    color_level = watercolor_color_level(strength)
    if color_level < 0.25:
        rgba = (72, 210, 116, 112)
    elif color_level < 0.4:
        rgba = (78, 226, 118, 128)
    elif color_level < 0.75:
        rgba = (238, 198, 68, 158)
    else:
        rgba = (238, 118, 48, 200)
    red, green, blue, alpha = rgba
    return QtGui.QColor(red, green, blue, int(clamp(alpha * opacity * WATERCOLOR_ALPHA_MULTIPLIER, 0, 255)))


def pulse_color(strength, opacity, kind="unknown"):
    _ = kind
    strength = clamp(float(strength))
    if strength < 0.25:
        rgba = (60, 200, 60, 70)
    elif strength < 0.4:
        rgba = (40, 255, 80, 110)
    elif strength < 0.75:
        rgba = (255, 220, 60, 160)
    else:
        rgba = (255, 120, 40, 230)
    red, green, blue, alpha = rgba
    return QtGui.QColor(red, green, blue, int(clamp(alpha * opacity * opacity_multiplier, 0, 255)))


def centered_top_left(screen_geometry, window_size):
    return QtCore.QPoint(
        screen_geometry.x() + (screen_geometry.width() - window_size.width()) // 2,
        screen_geometry.y() + (screen_geometry.height() - window_size.height()) // 2,
    )


def active_screen():
    screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
    return screen or QtWidgets.QApplication.primaryScreen()


def fit_square_size_to_screen(desired_size, screen_geometry, max_screen_ratio=0.9):
    max_size = int(min(screen_geometry.width(), screen_geometry.height()) * max_screen_ratio)
    return min(desired_size, max_size)


def desired_window_size(base_size=500):
    return int(base_size * (1.0 + max(0, (size_multiplier - 5.0) * 0.1)))


def fitted_window_size(screen=None):
    screen = screen or active_screen()
    desired_size = desired_window_size()
    if screen is None:
        return desired_size
    return fit_square_size_to_screen(desired_size, screen.geometry())


def overlay_window_flags(platform_name=None):
    """Qt-level flags shared by macOS and Windows overlays.

    Native platform hooks below strengthen always-on-top behavior, but these flags
    keep the window frameless, topmost, tool-like, click-through, and non-activating.
    """
    _ = platform_name or sys.platform  # currently identical, retained for tests/clarity
    flags = (
        QtCore.Qt.FramelessWindowHint
        | QtCore.Qt.WindowStaysOnTopHint
        | QtCore.Qt.Tool
        | QtCore.Qt.WindowDoesNotAcceptFocus
        | QtCore.Qt.NoDropShadowWindowHint
    )
    if hasattr(QtCore.Qt, "WindowTransparentForInput"):
        flags |= QtCore.Qt.WindowTransparentForInput
    return flags


def configure_overlay_widget(widget, platform_name=None):
    widget.setWindowFlags(overlay_window_flags(platform_name))
    widget.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
    widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
    widget.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
    widget.setFocusPolicy(QtCore.Qt.NoFocus)


def _objc_msg_send(receiver, selector_name, restype=None, argtypes=(), *args):
    objc_path = ctypes.util.find_library("objc")
    if not objc_path:
        raise RuntimeError("libobjc not found")

    objc = ctypes.CDLL(objc_path)
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    selector = objc.sel_registerName(selector_name.encode("ascii"))

    msg_send = objc.objc_msgSend
    msg_send.restype = restype
    msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(selector), *args)


def _objc_responds_to(receiver, selector_name):
    objc_path = ctypes.util.find_library("objc")
    if not objc_path:
        return False

    objc = ctypes.CDLL(objc_path)
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    selector = objc.sel_registerName(selector_name.encode("ascii"))
    responds_to_selector = objc.sel_registerName(b"respondsToSelector:")

    msg_send = objc.objc_msgSend
    msg_send.restype = ctypes.c_bool
    msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return bool(msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(responds_to_selector), ctypes.c_void_p(selector)))


def macos_overlay_window_level():
    if sys.platform != "darwin":
        return 0

    try:
        core_graphics_path = ctypes.util.find_library("CoreGraphics")
        if not core_graphics_path:
            return 1000
        core_graphics = ctypes.CDLL(core_graphics_path)
        core_graphics.CGWindowLevelForKey.restype = ctypes.c_int32
        core_graphics.CGWindowLevelForKey.argtypes = [ctypes.c_int32]
        return int(core_graphics.CGWindowLevelForKey(13))  # kCGScreenSaverWindowLevelKey
    except Exception:
        return 1000


def apply_macos_overlay_level(widget):
    if sys.platform != "darwin":
        return False

    try:
        widget.winId()  # ensure Qt has created the native NSView
        ns_view = int(widget.winId())
        ns_window = _objc_msg_send(ns_view, "window", restype=ctypes.c_void_p)
        if not ns_window:
            return False

        _objc_msg_send(
            ns_window,
            "setLevel:",
            None,
            (ctypes.c_long,),
            ctypes.c_long(macos_overlay_window_level()),
        )
        # CanJoinAllSpaces | FullScreenAuxiliary: visible across Spaces/full-screen
        # without activating the foreground app.
        collection_behavior = (1 << 0) | (1 << 8)
        _objc_msg_send(
            ns_window,
            "setCollectionBehavior:",
            None,
            (ctypes.c_ulong,),
            ctypes.c_ulong(collection_behavior),
        )
        _objc_msg_send(
            ns_window,
            "setIgnoresMouseEvents:",
            None,
            (ctypes.c_bool,),
            ctypes.c_bool(True),
        )
        if _objc_responds_to(ns_window, "setHidesOnDeactivate:"):
            _objc_msg_send(
                ns_window,
                "setHidesOnDeactivate:",
                None,
                (ctypes.c_bool,),
                ctypes.c_bool(False),
            )
        if _objc_responds_to(ns_window, "setCanHide:"):
            _objc_msg_send(
                ns_window,
                "setCanHide:",
                None,
                (ctypes.c_bool,),
                ctypes.c_bool(False),
            )
        _objc_msg_send(ns_window, "orderFrontRegardless", None)
        return True
    except Exception:
        return False


def apply_windows_overlay_level(widget):
    if sys.platform != "win32":
        return False

    try:
        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32

        gwl_exstyle = -20
        ws_ex_transparent = 0x00000020
        ws_ex_toolwindow = 0x00000080
        ws_ex_layered = 0x00080000
        ws_ex_noactivate = 0x08000000

        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_long.restype = ctypes.c_longlong
        set_long.restype = ctypes.c_longlong

        exstyle = int(get_long(hwnd, gwl_exstyle))
        exstyle |= ws_ex_transparent | ws_ex_toolwindow | ws_ex_layered | ws_ex_noactivate
        set_long(hwnd, gwl_exstyle, exstyle)

        hwnd_topmost = -1
        swp_nosize = 0x0001
        swp_nomove = 0x0002
        swp_noactivate = 0x0010
        swp_showwindow = 0x0040
        swp_noownerzorder = 0x0200
        flags = swp_nosize | swp_nomove | swp_noactivate | swp_showwindow | swp_noownerzorder
        return bool(user32.SetWindowPos(hwnd, hwnd_topmost, 0, 0, 0, 0, flags))
    except Exception:
        return False


def apply_native_overlay_level(widget, platform_name=None):
    platform_name = platform_name or sys.platform
    if platform_name == "darwin":
        return apply_macos_overlay_level(widget)
    if platform_name.startswith("win"):
        return apply_windows_overlay_level(widget)
    return False


class TranslucentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, position=0):
        super().__init__(parent)
        self.position = position
        self.strength = 0.0
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def _arc_color(self, strength):
        if strength < 0.25:
            rgba = (60, 200, 60, 40)
        elif strength < 0.4:
            rgba = (40, 255, 80, 90)
        elif strength < 0.75:
            rgba = (255, 220, 60, 150)
        else:
            rgba = (255, 120, 40, 220)
        r, g, b, alpha = rgba
        return QtGui.QColor(r, g, b, int(clamp(alpha * ARC_OPACITY_MULTIPLIER, 0, 255)))

    def _arc_radius(self, width, height, strength, pen_width):
        if size_multiplier <= 1.0:
            max_radius_ratio = 0.48
        elif size_multiplier <= 5.0:
            max_radius_ratio = 0.48 + (0.95 - 0.48) * ((size_multiplier - 1.0) / 4.0)
        else:
            max_radius_ratio = 0.95 + (0.98 - 0.95) * min((size_multiplier - 5.0) / 5.0, 1.0)

        max_radius = (min(width, height) / 2) * max_radius_ratio - pen_width
        max_min_ratio = 0.6
        desired_min_radius = min(width, height) * 0.18 * size_multiplier
        actual_min_radius = min(desired_min_radius, max_radius * max_min_ratio)
        min_r = actual_min_radius / size_multiplier
        max_r = max_radius / size_multiplier
        if min_r >= max_r:
            min_r = max_r * max_min_ratio
        return min((min_r + (max_r - min_r) * strength) * size_multiplier, max_radius)

    def paintEvent(self, event):
        _ = event
        if not SHOW_ARCS:
            return
        width, height = self.width(), self.height()
        center_x, center_y = width / 2, height / 2
        strength = clamp(float(getattr(self, "strength", 0.0)))
        if strength < ARC_MIN_VISIBLE_STRENGTH:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        pen_width = 2 + 10 * strength
        painter.setPen(QtGui.QPen(self._arc_color(strength), pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.setBrush(QtCore.Qt.NoBrush)

        radius = self._arc_radius(width, height, strength, pen_width)
        rect = QtCore.QRectF(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
        painter.drawArc(rect, int(arc_start_deg_for_position(self.position) * 16), int(30 * 16))
        painter.end()


class RippleOverlayWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def _paint_arc_ripple(self, painter, pulse, now, min_side, center_x, center_y):
        radius = pulse_ripple_radius(pulse, now, min_side)
        rect = QtCore.QRectF(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
        span = 18 + 18 * clamp(pulse.strength)
        start = sector_mid_angle_deg(pulse.sector) - span / 2
        opacity = pulse_opacity(pulse, now)
        line_width = 2 + 8 * clamp(pulse.strength) * (1.0 - pulse_age_ratio(pulse, now) * 0.55)
        painter.setPen(QtGui.QPen(pulse_color(pulse.strength, opacity, pulse.kind), line_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawArc(rect, int(start * 16), int(span * 16))

        inner_radius = radius - min_side * 0.045
        if inner_radius > min_side * 0.34:
            inner_rect = QtCore.QRectF(center_x - inner_radius, center_y - inner_radius, 2 * inner_radius, 2 * inner_radius)
            painter.setPen(QtGui.QPen(pulse_color(pulse.strength, opacity * 0.45, pulse.kind), max(1.0, line_width * 0.55), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawArc(inner_rect, int(start * 16), int(span * 16))

    def _draw_watercolor_ellipse(self, painter, center, radius, stretch, rotation_deg, color, stops):
        painter.save()
        painter.translate(center)
        painter.rotate(-rotation_deg)
        painter.scale(stretch, 1.0)
        gradient = QtGui.QRadialGradient(QtCore.QPointF(0, 0), radius)
        for stop, alpha_scale in stops:
            stop_color = QtGui.QColor(color)
            stop_color.setAlpha(int(color.alpha() * alpha_scale))
            gradient.setColorAt(stop, stop_color)
        transparent = QtGui.QColor(color)
        transparent.setAlpha(0)
        gradient.setColorAt(1.0, transparent)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawEllipse(QtCore.QRectF(-radius, -radius, radius * 2, radius * 2))
        painter.restore()

    def _watercolor_center(self, spec, center_x, center_y, distance=None, angle_deg=None):
        angle_rad = math.radians(spec.angle_deg if angle_deg is None else angle_deg)
        distance = spec.distance if distance is None else distance
        return QtCore.QPointF(
            center_x + math.cos(angle_rad) * distance,
            center_y - math.sin(angle_rad) * distance,
        )

    def _event_icon_pen_width(self, size):
        return max(2.0, size * 0.08)

    def _draw_event_icon_badge(self, painter, size, badge_color, accent_color):
        if EVENT_ICON_SHOW_LABELS:
            rect = QtCore.QRectF(-size * 0.66, -size * 0.60, size * 1.32, size * 1.36)
        else:
            rect = QtCore.QRectF(-size * 0.54, -size * 0.54, size * 1.08, size * 1.08)
        painter.setBrush(QtGui.QBrush(badge_color))
        painter.setPen(QtGui.QPen(accent_color, max(2.5, size * 0.055), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawRoundedRect(rect, size * 0.16, size * 0.16)

    def _draw_event_icon_label(self, painter, label, size, color, shadow_color):
        font = QtGui.QFont("Arial")
        font.setBold(True)
        font.setPixelSize(max(10, int(size * 0.22)))
        rect = QtCore.QRectF(-size * 0.58, size * 0.30, size * 1.16, size * 0.34)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(shadow_color, max(2.0, size * 0.04)))
        painter.drawText(rect.translated(0, size * 0.025), QtCore.Qt.AlignCenter, label)
        painter.setPen(color)
        painter.drawText(rect, QtCore.Qt.AlignCenter, label)

    def _draw_event_icon_gunshot(self, painter, size, color, accent_color, shadow_color):
        pen_width = self._event_icon_pen_width(size)
        painter.setBrush(QtGui.QBrush(accent_color))
        painter.setPen(QtGui.QPen(shadow_color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawPolygon(event_star_polygon(4, size * 0.14, size * 0.42, rotation_deg=45.0))
        painter.setPen(QtGui.QPen(color, pen_width * 0.75, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(QtCore.QPointF(-size * 0.44, size * 0.24), QtCore.QPointF(size * 0.18, -size * 0.18))
        painter.drawEllipse(QtCore.QPointF(size * 0.24, -size * 0.22), size * 0.06, size * 0.06)

    def _draw_event_icon_explosion(self, painter, size, color, accent_color, shadow_color):
        painter.setBrush(QtGui.QBrush(accent_color))
        painter.setPen(QtGui.QPen(shadow_color, self._event_icon_pen_width(size), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawPolygon(event_star_polygon(8, size * 0.22, size * 0.48, rotation_deg=90.0))
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(0, 0), size * 0.13, size * 0.13)

    def _draw_event_icon_vehicle(self, painter, size, color, accent_color, shadow_color):
        pen_width = self._event_icon_pen_width(size)
        painter.setPen(QtGui.QPen(shadow_color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.setBrush(QtGui.QBrush(accent_color))
        body = QtCore.QRectF(-size * 0.42, -size * 0.18, size * 0.84, size * 0.34)
        cabin = QtCore.QRectF(-size * 0.20, -size * 0.34, size * 0.40, size * 0.22)
        painter.drawRoundedRect(body, size * 0.08, size * 0.08)
        painter.drawRoundedRect(cabin, size * 0.06, size * 0.06)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(QtCore.QRectF(-size * 0.13, -size * 0.29, size * 0.10, size * 0.12))
        painter.drawRect(QtCore.QRectF(size * 0.03, -size * 0.29, size * 0.10, size * 0.12))
        painter.setBrush(QtGui.QBrush(shadow_color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(-size * 0.25, size * 0.20), size * 0.09, size * 0.09)
        painter.drawEllipse(QtCore.QPointF(size * 0.25, size * 0.20), size * 0.09, size * 0.09)

    def _draw_event_icon_footstep(self, painter, size, color, accent_color, shadow_color):
        pen_width = self._event_icon_pen_width(size)
        painter.setPen(QtGui.QPen(shadow_color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.setBrush(QtGui.QBrush(accent_color))
        for x_offset, y_offset, rotation in ((-0.16, 0.10, -20), (0.16, -0.12, -20)):
            painter.save()
            painter.translate(size * x_offset, size * y_offset)
            painter.rotate(rotation)
            painter.drawEllipse(QtCore.QRectF(-size * 0.13, -size * 0.23, size * 0.26, size * 0.46))
            painter.setBrush(QtGui.QBrush(color))
            for toe_x in (-0.09, 0.0, 0.09):
                painter.drawEllipse(QtCore.QPointF(size * toe_x, -size * 0.26), size * 0.035, size * 0.035)
            painter.setBrush(QtGui.QBrush(accent_color))
            painter.restore()

    def _paint_event_icon(self, painter, pulse, now, min_side, center_x, center_y):
        if not is_classified_event_kind(pulse.kind):
            return
        color = event_icon_color(pulse, now)
        if color.alpha() <= 0:
            return
        accent_color = event_icon_accent_color(pulse, now)
        badge_color = event_icon_badge_color(pulse, now)
        shadow_color = event_icon_shadow_color(pulse, now)
        size = event_icon_size(pulse, now, min_side)
        center = event_icon_center(
            pulse.sector,
            min_side,
            center_x,
            center_y,
            lane_index=getattr(pulse, "lane_index", 0),
            lane_count=getattr(pulse, "lane_count", 1),
        )
        painter.save()
        painter.translate(center)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        self._draw_event_icon_badge(painter, size, badge_color, accent_color)
        painter.save()
        if EVENT_ICON_SHOW_LABELS:
            painter.translate(0, -size * 0.11)
        glyph_size = size * (0.88 if EVENT_ICON_SHOW_LABELS else 0.78)
        if pulse.kind == "gunshot":
            self._draw_event_icon_gunshot(painter, glyph_size, color, accent_color, shadow_color)
        elif pulse.kind == "explosion":
            self._draw_event_icon_explosion(painter, glyph_size, color, accent_color, shadow_color)
        elif pulse.kind == "vehicle":
            self._draw_event_icon_vehicle(painter, glyph_size, color, accent_color, shadow_color)
        elif pulse.kind == "footstep":
            self._draw_event_icon_footstep(painter, glyph_size, color, accent_color, shadow_color)
        painter.restore()
        if EVENT_ICON_SHOW_LABELS:
            self._draw_event_icon_label(painter, event_icon_label(pulse.kind), size, color, shadow_color)
        painter.restore()

    def _paint_watercolor_pulse(self, painter, pulse, now, min_side, center_x, center_y):
        specs = watercolor_blob_specs(pulse, now, min_side)
        age = pulse_age_ratio(pulse, now)
        safe_distance = min_side * WATERCOLOR_INNER_SAFE_RATIO
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        for spec in specs:
            if spec.opacity <= 0:
                continue
            for trail_index in range(WATERCOLOR_FLOW_TRAILS, 0, -1):
                trail_ratio = trail_index / WATERCOLOR_FLOW_TRAILS
                trail_distance = max(
                    safe_distance,
                    spec.distance - min_side * WATERCOLOR_TRAIL_GAP_RATIO * trail_ratio * (1.0 + 0.45 * age),
                )
                trail_angle = normalize_degrees(spec.angle_deg - spec.flow_deg * 0.22 * trail_ratio)
                center = self._watercolor_center(spec, center_x, center_y, trail_distance, trail_angle)
                color = watercolor_color(pulse.strength, spec.opacity * (0.82 - 0.16 * trail_ratio), pulse.kind)
                if color.alpha() <= 0:
                    continue
                self._draw_watercolor_ellipse(
                    painter,
                    center,
                    spec.radius * (WATERCOLOR_SOFT_SCALE + 0.25 * trail_ratio),
                    spec.stretch * (1.16 + 0.22 * trail_ratio),
                    spec.rotation_deg - spec.flow_deg * 0.35 * trail_ratio,
                    color,
                    [(0.0, 0.16), (0.44, 0.105), (0.82, 0.035)],
                )

        for spec in specs:
            if spec.opacity <= 0:
                continue
            center = self._watercolor_center(spec, center_x, center_y)
            color = watercolor_color(pulse.strength, spec.opacity, pulse.kind)
            if color.alpha() <= 0:
                continue
            self._draw_watercolor_ellipse(
                painter,
                center,
                spec.radius * WATERCOLOR_SOFT_SCALE,
                spec.stretch * 1.12,
                spec.rotation_deg,
                color,
                [(0.0, 0.2), (0.45, 0.13), (0.82, 0.04)],
            )

        for spec in specs:
            if spec.opacity <= 0:
                continue
            center = self._watercolor_center(spec, center_x, center_y)
            color = watercolor_color(pulse.strength, spec.opacity, pulse.kind)
            if color.alpha() <= 0:
                continue
            self._draw_watercolor_ellipse(
                painter,
                center,
                spec.radius * WATERCOLOR_PIGMENT_SCALE,
                spec.stretch,
                spec.rotation_deg,
                color,
                [(0.0, 0.64), (0.36, 0.45), (0.72, 0.15)],
            )

    def _event_debug_lines(self, parent):
        runtime = getattr(parent, "direction_event_runtime", None)
        if runtime is None:
            return direction_event_debug_lines(
                getattr(parent, "latest_direction_event_prediction", None),
                status="off",
                requested_device="none",
            )

        requested_device = getattr(runtime, "device", AST_DIRECTION_EVENT_DEVICE)
        resolved_device = getattr(runtime, "resolved_device", None)
        resolved_dtype = getattr(runtime, "resolved_dtype", None)
        teacher_model = getattr(runtime, "teacher_model", AST_DIRECTION_EVENT_TEACHER_MODEL)
        if getattr(runtime, "disabled_reason", None):
            return [
                direction_event_debug_header("disabled", requested_device, resolved_device, resolved_dtype, teacher_model),
                f"{runtime.disabled_reason[:64]}",
            ]
        status = "running" if getattr(runtime, "_future", None) is not None else "idle"
        return direction_event_debug_lines(
            getattr(parent, "latest_direction_event_prediction", None),
            status=status,
            requested_device=requested_device,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            teacher_model=teacher_model,
            radar_latency_ms=getattr(parent, "radar_latency_ms", None),
            ast_latency_ms=getattr(parent, "ast_latency_ms", None),
            display_debug=getattr(parent, "latest_direction_event_display_debug", None),
        )

    def _paint_event_debug_text(self, painter, parent, min_side):
        if not SHOW_EVENT_DEBUG_TEXT:
            return
        lines = self._event_debug_lines(parent)
        if not lines:
            return

        font = QtGui.QFont("Menlo")
        font.setPointSize(max(10, int(min_side * 0.018)))
        painter.save()
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        padding = 8
        line_gap = 3
        line_height = metrics.height()
        text_width = max(metrics.horizontalAdvance(line) for line in lines)
        box_width = text_width + padding * 2
        box_height = line_height * len(lines) + line_gap * max(0, len(lines) - 1) + padding * 2
        rect = QtCore.QRectF(14, 14, box_width, box_height)

        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0), 0))
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.drawRoundedRect(rect, 8, 8)

        y = rect.top() + padding + metrics.ascent()
        for index, line in enumerate(lines):
            painter.setPen(QtGui.QColor(230, 255, 235, 235) if index == 0 else QtGui.QColor(220, 235, 255, 220))
            painter.drawText(QtCore.QPointF(rect.left() + padding, y), line)
            y += line_height + line_gap
        painter.restore()

    def paintEvent(self, event):
        _ = event
        if not SHOW_RIPPLES:
            return
        parent = self.parent()
        if parent is None:
            return

        now = time.time()
        width, height = self.width(), self.height()
        min_side = min(width, height)
        center_x, center_y = width / 2, height / 2

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for pulse in list(getattr(parent, "pulses", [])):
            if pulse_opacity(pulse, now) <= 0:
                continue
            if is_classified_event_kind(pulse.kind):
                self._paint_event_icon(painter, pulse, now, min_side, center_x, center_y)
            elif RIPPLE_STYLE == "watercolor":
                self._paint_watercolor_pulse(painter, pulse, now, min_side, center_x, center_y)
            else:
                self._paint_arc_ripple(painter, pulse, now, min_side, center_x, center_y)
        self._paint_event_debug_text(painter, parent, min_side)
        painter.end()


class ParentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        configure_overlay_widget(self)
        self.popframes = {}
        self._popflag = False
        self.global_peak = 0.1
        self.pulses = []
        self._last_pulse_times = np.zeros(RADAR_SECTORS)
        self._last_event_pulse_times = np.zeros(RADAR_SECTORS)
        self._last_global_event_times = {}
        self._previous_direction_levels = np.zeros(RADAR_SECTORS)
        self.direction_event_runtime = None
        self._direction_event_prediction_history = []
        self.rolling_capture = None
        self.threshold_profile_name = "default"
        self.latest_direction_event_prediction = None
        self.latest_direction_event_display_debug = None
        self.radar_latency_ms = None
        self.ast_latency_ms = None
        self._create_sectors()
        self.ripple_overlay = RippleOverlayWidget(self)
        self.ripple_overlay.move(0, 0)
        self.ripple_overlay.resize(self.width(), self.height())
        self.ripple_overlay.show()
        self.ripple_overlay.raise_()
        self.setBackgroundcolor()
        self._native_top_timer = QtCore.QTimer(self)
        self._native_top_timer.timeout.connect(self.ensure_on_top)
        self._native_top_timer.start(750)

    def _create_sectors(self):
        for position in range(RADAR_SECTORS):
            self.create_shape(position)

    def center_on_screen(self):
        screen = active_screen()
        if screen is not None:
            self.move(centered_top_left(screen.geometry(), self.size()))

    def ensure_on_top(self):
        if not self.windowFlags() & QtCore.Qt.WindowStaysOnTopHint:
            self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        apply_native_overlay_level(self)

    def showEvent(self, event):
        super().showEvent(event)
        apply_native_overlay_level(self)

    def resizeEvent(self, event):
        _ = event
        if self._popflag:
            for frame in self.popframes.values():
                shape = frame["shape"]
                shape.move(0, 0)
                shape.resize(self.width(), self.height())
        if hasattr(self, "ripple_overlay"):
            self.ripple_overlay.move(0, 0)
            self.ripple_overlay.resize(self.width(), self.height())
            self.ripple_overlay.raise_()

    def create_shape(self, position=0):
        shape = TranslucentWidget(self, position)
        shape.move(0, 0)
        shape.resize(self.width(), self.height())
        shape.show()
        self._popflag = True
        self.popframes[position] = {"shape": shape, "tupdate": 0.0, "fistFlag": False}

    def display_strength(self, raw):
        raw = clamp(float(raw))
        if STRENGTH_MODE == 1:
            self.global_peak = max(self.global_peak * 0.9, raw, 1e-3)
            ratio = raw / (self.global_peak + 1e-6)
            return 0.0 if ratio < 0.6 else clamp((ratio - 0.6) / 0.4)
        return raw * raw

    def updateBrush(self, color, position):
        _ = color  # color is kept for compatibility with the old call site.
        try:
            self.popframes[position]["shape"].strength = self.display_strength(prevmax[position])
        except Exception:
            self.popframes[position]["shape"].strength = 0.0
        self.update()

    def add_pulses(self, pulses):
        if not pulses:
            return
        self.pulses.extend(pulses)
        if len(self.pulses) > MAX_ACTIVE_PULSES:
            self.pulses = self.pulses[-MAX_ACTIVE_PULSES:]

    def prune_pulses(self, now):
        self.pulses = [pulse for pulse in self.pulses if not pulse_expired(pulse, now)]

    def setBackgroundcolor(self):
        palette = QtWidgets.QWidget.palette(self)
        palette.setColor(self.backgroundRole(), QtGui.QColor(0, 0, 0, 0))
        self.setPalette(palette)


def audio_callback(indata, frames, callback_time, status):
    _ = frames, callback_time
    if status:
        print(status, file=sys.stderr)
    q.put(TimedAudioBlock(indata.copy(), captured_at=time.perf_counter()))


def rolling_capture_output_path(directory=None, now=None):
    return _rolling_capture_output_path(directory or ROLLING_CAPTURE_DIR, now=now)


def rolling_capture_metadata_path(audio_path):
    return _rolling_capture_metadata_path(audio_path)


def rolling_capture_metadata_payload(
    snapshot,
    *,
    audio_path,
    saved_at=None,
    prediction=None,
    display_debug=None,
    threshold_profile_name=None,
):
    hud_summary_lines = None
    if prediction is not None:
        hud_summary_lines = direction_event_debug_lines(
            prediction,
            threshold=AST_DIRECTION_EVENT_THRESHOLD,
            display_debug=display_debug,
        )
    return _rolling_capture_metadata_payload(
        snapshot,
        audio_path=audio_path,
        saved_at=saved_at,
        prediction=prediction,
        hud_summary_lines=hud_summary_lines,
        threshold_profile_name=threshold_profile_name,
    )


def write_rolling_capture_snapshot(
    snapshot,
    *,
    directory=None,
    now=None,
    prediction=None,
    display_debug=None,
    threshold_profile_name=None,
):
    hud_summary_lines = None
    if prediction is not None:
        hud_summary_lines = direction_event_debug_lines(
            prediction,
            threshold=AST_DIRECTION_EVENT_THRESHOLD,
            display_debug=display_debug,
        )
    return _write_rolling_capture_snapshot(
        snapshot,
        directory=directory or ROLLING_CAPTURE_DIR,
        now=now,
        prediction=prediction,
        hud_summary_lines=hud_summary_lines,
        threshold_profile_name=threshold_profile_name,
    )


def consume_rolling_capture_trigger(path=None):
    return _consume_rolling_capture_trigger(path or ROLLING_CAPTURE_TRIGGER_PATH)


def write_rolling_capture_trigger(path=None):
    return _write_rolling_capture_trigger(path or ROLLING_CAPTURE_TRIGGER_PATH)


class DirectionEventRuntime(_DirectionEventRuntime):
    def __init__(
        self,
        sample_rate,
        channel_count,
        *,
        window_seconds=AST_DIRECTION_EVENT_WINDOW_SECONDS,
        interval_seconds=AST_DIRECTION_EVENT_INTERVAL,
        top_k=AST_DIRECTION_EVENT_TOP_K,
        device=AST_DIRECTION_EVENT_DEVICE,
        dtype=AST_DIRECTION_EVENT_DTYPE,
        attn_implementation=AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION,
        compile_model=AST_DIRECTION_EVENT_COMPILE,
        teacher_model=AST_DIRECTION_EVENT_TEACHER_MODEL,
        model_id=None,
        channel_map=None,
        executor=None,
        score_fn=None,
        latency_clock=None,
        warmup=AST_DIRECTION_EVENT_WARMUP,
    ):
        super().__init__(
            sample_rate,
            channel_count,
            window_seconds=window_seconds,
            interval_seconds=interval_seconds,
            top_k=top_k,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            compile_model=compile_model,
            teacher_model=teacher_model,
            model_id=model_id,
            channel_map=channel_map,
            executor=executor,
            score_fn=score_fn,
            latency_clock=latency_clock,
            warmup=warmup,
        )


def unpack_audio_queue_item(item):
    if isinstance(item, TimedAudioBlock):
        return item.samples, float(item.captured_at)
    return item, None


def drain_audio_queue_with_timing(channel_count, audio_queue=None):
    audio_queue = q if audio_queue is None else audio_queue
    max_values = np.zeros(channel_count, dtype=np.float32)
    blocks = []
    latest_capture_time = None
    while True:
        try:
            item = audio_queue.get_nowait()
        except queue.Empty:
            break
        data, captured_at = unpack_audio_queue_item(item)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.size == 0:
            continue
        block = data[:, :channel_count]
        blocks.append(block)
        if captured_at is not None:
            latest_capture_time = captured_at if latest_capture_time is None else max(latest_capture_time, captured_at)
        block_max = np.nanmax(np.abs(block), axis=0)
        max_values[: len(block_max)] = np.maximum(max_values[: len(block_max)], block_max)
    return max_values / maxSoundValue, blocks, latest_capture_time


def drain_audio_queue(channel_count, audio_queue=None):
    max_values, blocks, _ = drain_audio_queue_with_timing(channel_count, audio_queue=audio_queue)
    return max_values, blocks


def getMaxSound(channel_count):
    max_values, _ = drain_audio_queue(channel_count)
    return max_values


def enhancer(value):
    if value < minThreshold:
        return 0.0
    normalized = (value - minThreshold) / (1.0 - minThreshold)
    return normalized ** 0.7


def expfilter(value):
    return 1 - math.exp(-5 * value)


def initfilter(values, threshold):
    filtered = np.array(values, copy=True)
    filtered[filtered < threshold] = 0
    return np.fromiter((expfilter(value) for value in filtered), filtered.dtype)


def apply_fade(current_value, elapsed_time, decay_rate=2.0):
    return current_value * math.exp(-decay_rate * elapsed_time)


def directional_difference(primary, secondary, min_ratio=None):
    return _directional_difference(primary, secondary, maxdifmain if min_ratio is None else min_ratio)


def is_directional_louder(primary, secondary, current_max):
    return directional_difference(primary, secondary) > current_max


def compute_direction_levels(max_values, channel_map):
    return _compute_direction_levels(max_values, channel_map, min_ratio=maxdifmain)


def update_sector_state(frame, position, candidate, now):
    if candidate > prevmax[position]:
        prevmax[position] = enhancer(candidate)
        frame["tupdate"] = now
        frame["fistFlag"] = True
    elif frame["fistFlag"] and now - frame["tupdate"] > minTFU:
        prevmax[position] = apply_fade(prevmax[position], now - frame["tupdate"], fade_decay_rate)
        frame["fistFlag"] = False
        frame["tupdate"] = now
    elif not frame["fistFlag"] and now - frame["tupdate"] > minTBU:
        prevmax[position] = apply_fade(prevmax[position], now - frame["tupdate"], fade_decay_rate)
        frame["tupdate"] = now

    if prevmax[position] < 0.01:
        prevmax[position] = 0.0


def update_direction_event_runtime(radarObject, audio_blocks, now, capture_time=None):
    runtime = getattr(radarObject, "direction_event_runtime", None)
    if runtime is None:
        return None
    runtime.append_blocks(audio_blocks, capture_time=capture_time)
    prediction = runtime.maybe_submit(now)
    radarObject.ast_latency_ms = getattr(runtime, "latest_latency_ms", None)
    if prediction is not None:
        if DIRECTION_EVENT_SMOOTHING_ENABLED and DIRECTION_EVENT_SMOOTHING_WINDOW > 1:
            history = getattr(radarObject, "_direction_event_prediction_history", [])
            history.append(prediction)
            history = history[-int(DIRECTION_EVENT_SMOOTHING_WINDOW) :]
            radarObject._direction_event_prediction_history = history
            prediction = smooth_direction_event_predictions(history, window=DIRECTION_EVENT_SMOOTHING_WINDOW)
        else:
            radarObject._direction_event_prediction_history = [prediction]
        radarObject.latest_direction_event_prediction = prediction
    return prediction


def update_rolling_capture(radarObject, audio_blocks, capture_time=None):
    rolling_capture = getattr(radarObject, "rolling_capture", None)
    if rolling_capture is None:
        return
    rolling_capture.append_blocks(audio_blocks, capture_time=capture_time)


def maybe_save_rolling_capture(radarObject, now):
    rolling_capture = getattr(radarObject, "rolling_capture", None)
    if rolling_capture is None or not consume_rolling_capture_trigger():
        return None
    snapshot = rolling_capture.snapshot()
    if np.asarray(snapshot.audio).size == 0:
        print("Rolling capture trigger ignored: no audio buffered yet.", file=sys.stderr)
        return None
    audio_path, metadata_path = write_rolling_capture_snapshot(
        snapshot,
        directory=ROLLING_CAPTURE_DIR,
        now=now,
        prediction=getattr(radarObject, "latest_direction_event_prediction", None),
        display_debug=getattr(radarObject, "latest_direction_event_display_debug", None),
        threshold_profile_name=getattr(radarObject, "threshold_profile_name", None),
    )
    print(f"Rolling capture saved: {audio_path}")
    print(f"Rolling capture metadata: {metadata_path}")
    return audio_path, metadata_path


def updateRadar(radarObject):
    while True:
        time.sleep(refreshtime)
        raw_max_values, audio_blocks, latest_capture_time = drain_audio_queue_with_timing(n_channel)
        max_values = initfilter(raw_max_values, minThreshold)
        direction_levels = compute_direction_levels(max_values, mapping)
        now = time.time()
        if latest_capture_time is not None:
            radarObject.radar_latency_ms = max(0.0, (time.perf_counter() - latest_capture_time) * 1000.0)
        update_rolling_capture(radarObject, audio_blocks, latest_capture_time)
        event_prediction = update_direction_event_runtime(radarObject, audio_blocks, now, latest_capture_time)

        if DEBUG:
            debug_channels = sorted(value for value in set(mapping.values()) if value is not None)
            print(max_values[debug_channels] * 100)

        if SHOW_RIPPLES:
            new_pulses = create_pulses_from_levels(
                direction_levels,
                now,
                threshold=RIPPLE_THRESHOLD,
                cooldown=RIPPLE_COOLDOWN,
                last_pulse_times=radarObject._last_pulse_times,
                previous_levels=radarObject._previous_direction_levels,
            )
            radarObject.add_pulses(new_pulses)
            event_display_debug = {}
            event_pulses = (
                create_pulses_from_direction_events(
                    event_prediction,
                    now,
                    threshold=AST_DIRECTION_EVENT_THRESHOLD,
                    cooldown=AST_DIRECTION_EVENT_COOLDOWN,
                    last_pulse_times=radarObject._last_event_pulse_times,
                    last_global_event_times=radarObject._last_global_event_times,
                    display_debug=event_display_debug,
                )
                if event_prediction is not None
                else []
            )
            if event_prediction is not None:
                radarObject.latest_direction_event_display_debug = event_display_debug.get("debug")
            radarObject.add_pulses(event_pulses)
            radarObject.prune_pulses(now)
            radarObject._previous_direction_levels = np.array(direction_levels, copy=True)
        maybe_save_rolling_capture(radarObject, now)

        for position, frame in radarObject.popframes.items():
            update_sector_state(frame, position, direction_levels[position], now)
            radarObject.updateBrush([0, prevmax[position] * maxColorRange, 0], position)

        if hasattr(radarObject, "ripple_overlay"):
            radarObject.ripple_overlay.update()

        if DEBUG:
            print(prevmax)
            print("----")
        QtWidgets.QApplication.processEvents()


def find_device_auto(search_keywords, device_type="input", preferred_min_channels=8):
    devices = sd.query_devices()
    candidates = []

    for keyword in search_keywords:
        keyword_lower = keyword.lower()
        for index, device in enumerate(devices):
            device_name = device["name"].lower()
            max_input = device.get("max_input_channels", 0)
            if keyword_lower not in device_name:
                continue
            if device_type == "input" and max_input > 0:
                candidates.append((index, device))
            elif device_type == "any":
                candidates.append((index, device))

    for index, device in candidates:
        if device.get("max_input_channels", 0) >= preferred_min_channels:
            return index, device
    return candidates[0] if candidates else (None, None)


def get_default_output_channel_count():
    try:
        output_device = sd.query_devices(None, "output")
        return int(output_device.get("max_output_channels", 0))
    except Exception:
        return None


def create_main_window():
    window = ParentWidget()
    window_size = fitted_window_size()
    window.resize(window_size, window_size)
    window.center_on_screen()
    window.show()
    window.center_on_screen()
    window.ensure_on_top()
    print(f"Radar window size: {window_size}x{window_size}")
    return window


def select_input_device():
    search_keywords = [
        "BlackHole 16ch",
        "BlackHole",
        "Loopback",
        "CABLE Output",
        "VB-Cable",
        "VB-Audio Virtual Cable",
        "VB-Audio",
    ]
    device_id, device_info = find_device_auto(search_keywords, "input")
    if device_id is not None:
        print(f"✓ Device found automatically: {device_info['name']} (ID: {device_id})")
        return device_id, device_info

    print(sd.query_devices())
    device_id = int(input("device id:"))
    return device_id, sd.query_devices(device_id, "input")


def configure_audio_mapping(device_info):
    global n_chans, n_channel, mapping, channel_mode, prevmax

    n_chans = int(device_info["max_input_channels"])
    n_channel = n_chans
    output_channel_count = get_default_output_channel_count()
    mapping, channel_mode = build_channel_mapping(n_chans, output_channel_count=output_channel_count)
    prevmax = np.zeros(RADAR_SECTORS)

    print(f"Input channels: {n_chans} ({channel_mode})")
    if output_channel_count is not None:
        print(f"Default output channels: {output_channel_count}")
    if "stereo" in channel_mode:
        print(
            "Warning: current output is stereo, so SoundRadar can only approximate "
            "left/right direction. Use a 7.1-capable output/source for surround direction."
        )
    elif n_chans < 8:
        print("Warning: this device is not exposing 7.1 input. Direction display is limited.")


def configure_direction_event_runtime(window, sample_rate, channel_count, channel_map=None):
    if not ENABLE_AST_DIRECTION_EVENTS:
        return False
    window.direction_event_runtime = DirectionEventRuntime(
        sample_rate=sample_rate,
        channel_count=channel_count,
        window_seconds=AST_DIRECTION_EVENT_WINDOW_SECONDS,
        interval_seconds=AST_DIRECTION_EVENT_INTERVAL,
        top_k=AST_DIRECTION_EVENT_TOP_K,
        device=AST_DIRECTION_EVENT_DEVICE,
        dtype=AST_DIRECTION_EVENT_DTYPE,
        attn_implementation=AST_DIRECTION_EVENT_ATTN_IMPLEMENTATION,
        compile_model=AST_DIRECTION_EVENT_COMPILE,
        teacher_model=AST_DIRECTION_EVENT_TEACHER_MODEL,
        model_id=AST_DIRECTION_EVENT_MODEL_ID,
        channel_map=channel_map,
    )
    print(
        "Audio direction-event overlay enabled "
        f"({AST_DIRECTION_EVENT_TEACHER_MODEL}, {AST_DIRECTION_EVENT_WINDOW_SECONDS:.1f}s window, "
        f"{AST_DIRECTION_EVENT_INTERVAL:.2f}s interval, {AST_DIRECTION_EVENT_DEVICE}, {AST_DIRECTION_EVENT_DTYPE})."
    )
    return True


def configure_rolling_capture(window, sample_rate, channel_count):
    if not ROLLING_CAPTURE_ENABLED or ROLLING_CAPTURE_SECONDS <= 0:
        window.rolling_capture = None
        return False
    window.rolling_capture = RollingAudioCapture(sample_rate, channel_count, ROLLING_CAPTURE_SECONDS)
    print(
        f"Rolling capture enabled ({ROLLING_CAPTURE_SECONDS:.1f}s). "
        f"Create {ROLLING_CAPTURE_TRIGGER_PATH} to save the latest buffer to {ROLLING_CAPTURE_DIR}."
    )
    return True


mapping, channel_mode = build_channel_mapping(n_chans)


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    config_path = runtime_config_path()
    runtime_config = load_runtime_config(config_path)
    apply_runtime_config(runtime_config)
    qt_argv, threshold_profile_name = parse_threshold_profile_args(argv, runtime_config=runtime_config)
    active_profile = apply_threshold_profile(threshold_profile_name)
    app = QtWidgets.QApplication(qt_argv)
    window = create_main_window()
    device_id, device_info = select_input_device()
    assert device_info is not None
    configure_audio_mapping(device_info)
    configure_direction_event_runtime(window, device_info.get("default_samplerate", 44_100), n_chans, mapping)
    configure_rolling_capture(window, device_info.get("default_samplerate", 44_100), n_chans)
    window.threshold_profile_name = active_profile.name
    if runtime_config:
        print(f"Runtime config: {config_path}")
    print(f"Threshold profile: {active_profile.name}")
    print("Make sure system Sound Output is set to BlackHole/Loopback/VB-Cable or a Multi-Output device that includes it.")

    stream = sd.InputStream(
        dtype=np.float32,
        device=device_id,
        channels=n_chans,
        samplerate=device_info["default_samplerate"],
        callback=audio_callback,
    )
    with stream:
        updateRadar(window)
    return app.exec_()


if __name__ == "__main__":
    main()
