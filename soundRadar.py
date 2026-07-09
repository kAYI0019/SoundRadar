import concurrent.futures
import ctypes
import ctypes.util
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import random
import sys
import time

import numpy as np
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets


RADAR_SECTORS = 12
FRONT_LEFT = "avg"
FRONT_RIGHT = "avd"
CENTER = "c"
LEFT = "g"
RIGHT = "d"
REAR_LEFT = "arg"
REAR_RIGHT = "ard"

# GLOBAL PARAMETERS (kept as simple module settings for easy tuning)
n_chans = 8
n_channel = n_chans
maxSoundValue = 1.0  # float32 streams are normalized to -1.0..1.0

STRENGTH_MODE = 2
minTFU = 0.5  # minimum Time needed for First Update (upper sound value)
minTBU = 0.1  # minimum Time needed Between Update (lower sound value)
maxdifmain = 0.01  # min ratio difference between paired directional channels
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
RIPPLE_THRESHOLD = 0.03
RIPPLE_COOLDOWN = 0.18
RIPPLE_DURATION = 0.65
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
AST_DIRECTION_EVENT_WINDOW_SECONDS = 1.0
AST_DIRECTION_EVENT_INTERVAL = 1.25
AST_DIRECTION_EVENT_THRESHOLD = 0.10
AST_DIRECTION_EVENT_COOLDOWN = 0.55
AST_DIRECTION_EVENT_WARMUP = True
SHOW_EVENT_DEBUG_TEXT = True
EVENT_DEBUG_MAX_LINES = 7
DIRECTION_EVENT_GUNSHOT_DISPLAY_THRESHOLD = 0.10
DIRECTION_EVENT_GUNSHOT_SECONDARY_THRESHOLD = 0.35
DIRECTION_EVENT_DISPLAY_THRESHOLDS = {
    "gunshot": DIRECTION_EVENT_GUNSHOT_DISPLAY_THRESHOLD,
    "explosion": 0.85,
    "vehicle": 0.60,
    "footstep": 0.85,
}
DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS = {
    "gunshot": DIRECTION_EVENT_GUNSHOT_SECONDARY_THRESHOLD,
    "explosion": 0.85,
    "vehicle": 0.60,
    "footstep": 0.85,
}
DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN = 0.25
DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION = 2
DIRECTION_EVENT_SMOOTHING_ENABLED = True
DIRECTION_EVENT_SMOOTHING_WINDOW = 3

# Gunshot direction tuning: keep detection permissive, then suppress spatial
# bleed and rapid repeats at display time so distant shots are still visible.
GUNSHOT_SPATIAL_MAX_DIRECTIONS = 2
GUNSHOT_GLOBAL_COOLDOWN = 0.18

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
EVENT_ICON_DURATION = 3.5
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

q = queue.Queue()


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


@dataclass
class SoundPulse:
    sector: int
    strength: float
    created_at: float
    duration: float = RIPPLE_DURATION
    kind: str = "unknown"
    lane_index: int = 0
    lane_count: int = 1


@dataclass(frozen=True)
class TimedAudioBlock:
    samples: object
    captured_at: float


@dataclass(frozen=True)
class RollingCaptureSnapshot:
    audio: object
    sample_rate: int
    channel_count: int
    start_capture_time: float | None = None
    end_capture_time: float | None = None


class RollingAudioCapture:
    def __init__(self, sample_rate, channel_count, seconds=ROLLING_CAPTURE_SECONDS):
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self.seconds = float(seconds)
        self.max_samples = max(1, int(round(self.sample_rate * self.seconds)))
        self._audio = np.zeros((0, self.channel_count), dtype=np.float32)
        self.end_capture_time = None

    def append_blocks(self, blocks, capture_time=None):
        prepared = []
        for block in blocks:
            audio = np.asarray(block, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            if audio.ndim != 2 or audio.shape[0] == 0:
                continue
            if audio.shape[1] < self.channel_count:
                padded = np.zeros((audio.shape[0], self.channel_count), dtype=np.float32)
                padded[:, : audio.shape[1]] = audio
                audio = padded
            prepared.append(audio[:, : self.channel_count])
        if prepared:
            self._audio = np.concatenate([self._audio, *prepared], axis=0)[-self.max_samples :]
            if capture_time is not None:
                self.end_capture_time = float(capture_time)

    def snapshot(self):
        audio = np.array(self._audio, copy=True)
        if self.end_capture_time is None:
            start_capture_time = None
        else:
            start_capture_time = self.end_capture_time - (audio.shape[0] / float(self.sample_rate))
        return RollingCaptureSnapshot(
            audio=audio,
            sample_rate=self.sample_rate,
            channel_count=self.channel_count,
            start_capture_time=start_capture_time,
            end_capture_time=self.end_capture_time,
        )


@dataclass(frozen=True)
class WatercolorBlob:
    angle_deg: float
    distance: float
    radius: float
    opacity: float
    stretch: float = 1.0
    rotation_deg: float = 0.0
    flow_deg: float = 0.0


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


def classify_basic_sound_event(strength, previous_strength=0.0):
    strength = clamp(float(strength))
    previous_strength = clamp(float(previous_strength))
    if strength >= 0.85:
        return "impact"
    if strength - previous_strength >= 0.25:
        return "sharp"
    return "unknown"


def should_emit_pulse(last_time, now, cooldown=RIPPLE_COOLDOWN):
    return now - last_time >= cooldown


def create_pulses_from_levels(
    levels,
    now,
    threshold=RIPPLE_THRESHOLD,
    cooldown=RIPPLE_COOLDOWN,
    last_pulse_times=None,
    previous_levels=None,
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
                duration=RIPPLE_DURATION,
                kind=classify_basic_sound_event(strength, previous_strength),
            )
        )
        if last_pulse_times is not None:
            last_pulse_times[sector] = now
    return pulses


def dominant_active_event(
    scores,
    active_events,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
    active = set(active_events or [])
    best_event = None
    best_score = 0.0
    gunshot_score = clamp(float(scores.get("gunshot", 0.0)))
    gunshot_threshold = max(float(threshold), float(event_thresholds.get("gunshot", threshold)))
    for event_name in DIRECTION_EVENT_PRIORITY:
        score = clamp(float(scores.get(event_name, 0.0)))
        event_threshold = max(float(threshold), float(event_thresholds.get(event_name, threshold)))
        if score >= event_threshold and (not active or event_name in active):
            if best_event is None or score > best_score:
                best_event = event_name
                best_score = score
    if (
        best_event in ("explosion", "footstep")
        and gunshot_score >= gunshot_threshold
        and (not active or "gunshot" in active)
        and best_score - gunshot_score <= DIRECTION_EVENT_GUNSHOT_BIAS_MARGIN
    ):
        return "gunshot", gunshot_score
    return best_event, best_score


def event_display_threshold(event_name, threshold, event_thresholds):
    return max(float(threshold), float(event_thresholds.get(event_name, threshold)))


def active_event_candidates(
    scores,
    active_events,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
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
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    secondary_thresholds=DIRECTION_EVENT_SECONDARY_DISPLAY_THRESHOLDS,
    max_events=DIRECTION_EVENT_MAX_ICONS_PER_DIRECTION,
):
    candidates = active_event_candidates(scores, active_events, threshold, event_thresholds)
    if not candidates:
        return []

    primary_event, primary_score = dominant_active_event(scores, active_events, threshold, event_thresholds)
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
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
):
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
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
):
    """Return gunshot display candidates and spatial suppression decisions.

    The threshold stays permissive for distant shots, but adjacent directions are
    clustered and only the local maximum from each cluster is displayed.
    """

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
        winner_score = candidate_scores[winner]
        allowed.append((winner, winner_score))

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
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    event_thresholds=DIRECTION_EVENT_DISPLAY_THRESHOLDS,
    max_directions=GUNSHOT_SPATIAL_MAX_DIRECTIONS,
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


def smooth_direction_event_predictions(predictions, *, window=DIRECTION_EVENT_SMOOTHING_WINDOW):
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
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
    cooldown=AST_DIRECTION_EVENT_COOLDOWN,
    last_pulse_times=None,
    last_global_event_times=None,
    display_debug=None,
):
    pulses = []
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
    gunshot_decision = gunshot_display_decision(
        scores_by_direction,
        active_by_direction,
        threshold,
    )
    allowed_gunshot_directions = set(gunshot_decision.allowed_directions)
    emitted_gunshot_directions = []
    global_cooldown_blocked_gunshot_directions = []
    sector_cooldown_blocked_gunshot_directions = []

    for direction, sector in DIRECTION_EVENT_SECTORS.items():
        scores = scores_by_direction.get(direction, {})
        events = displayed_active_events(scores, active_by_direction.get(direction, ()), threshold)
        events = suppress_displayed_events_for_direction(direction, events, allowed_gunshot_directions)
        if not events:
            continue
        if (
            last_global_event_times is not None
            and any(event_name == "gunshot" for event_name, _ in events)
            and not should_emit_pulse(last_global_event_times.get("gunshot", -float("inf")), now, GUNSHOT_GLOBAL_COOLDOWN)
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
                    duration=EVENT_ICON_DURATION,
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


def compact_score(score):
    return f"{clamp(float(score)):.2f}".lstrip("0")


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
):
    teacher_label = str(teacher_model or AST_DIRECTION_EVENT_TEACHER_MODEL or "teacher")
    if status is None and requested_device is None and resolved_device is None and resolved_dtype is None:
        return f"{teacher_label} events"
    return f"{teacher_label} {status or 'idle'} {direction_event_device_label(requested_device=requested_device, resolved_device=resolved_device, resolved_dtype=resolved_dtype)}"


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


def direction_event_debug_cell(direction, scores_by_direction, active_by_direction, threshold=AST_DIRECTION_EVENT_THRESHOLD):
    label = DIRECTION_DEBUG_LABELS[direction]
    scores = scores_by_direction.get(direction, {}) or {}
    events = displayed_active_events(scores, active_by_direction.get(direction, ()), threshold)
    if not events:
        return f"{label}: --"
    event_labels = "/".join(EVENT_DEBUG_LABELS[event_name] for event_name, _ in events)
    scores_label = "/".join(compact_score(score) for _, score in events)
    return f"{label}: {event_labels} {scores_label}"


def direction_event_gunshot_debug_line(
    prediction,
    display_debug=None,
    threshold=AST_DIRECTION_EVENT_THRESHOLD,
):
    if display_debug is not None:
        decision = display_debug.gunshot_decision
        shown_directions = display_debug.gunshot_emitted_directions
    elif prediction is not None:
        scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
        active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
        decision = gunshot_display_decision(scores_by_direction, active_by_direction, threshold)
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
    header = direction_event_debug_header(status, requested_device, resolved_device, resolved_dtype, teacher_model)
    if prediction is None:
        lines = [
            header,
            direction_latency_debug_line(radar_latency_ms, ast_latency_ms),
            direction_event_gunshot_debug_line(None, display_debug, threshold),
            "waiting for audio/model...",
        ]
        return lines[: int(max_lines)] if max_lines else lines

    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
    cell = lambda direction: direction_event_debug_cell(direction, scores_by_direction, active_by_direction, threshold)
    lines = [
        header,
        direction_latency_debug_line(radar_latency_ms, ast_latency_ms),
        direction_event_gunshot_debug_line(prediction, display_debug, threshold),
        f"        {cell('front')}",
        f"{cell('front_left')}    {cell('front_right')}",
        f"{cell('left')}    {cell('right')}",
        f"{cell('rear_left')}    {cell('rear_right')}",
    ]
    return lines[: int(max_lines)] if max_lines else lines


def normalize_degrees(angle):
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def sector_mid_angle_deg(sector):
    return normalize_degrees(arc_start_deg_for_position(sector) + 15)


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
    angle_rad = math.radians(sector_mid_angle_deg(sector))
    distance = float(min_side) * float(distance_ratio)
    lane_count = max(1, int(lane_count))
    lane_index = clamp(float(lane_index), 0.0, float(lane_count - 1))
    lane_offset = (lane_index - (lane_count - 1) * 0.5) * float(min_side) * float(lane_spacing_ratio)
    return QtCore.QPointF(
        float(center_x) + math.cos(angle_rad) * distance - math.sin(angle_rad) * lane_offset,
        float(center_y) - math.sin(angle_rad) * distance - math.cos(angle_rad) * lane_offset,
    )


def event_icon_size(pulse, now, min_side):
    strength = clamp(float(getattr(pulse, "strength", 0.0)))
    base_ratio = EVENT_ICON_MIN_SIZE_RATIO + (EVENT_ICON_MAX_SIZE_RATIO - EVENT_ICON_MIN_SIZE_RATIO) * math.sqrt(strength)
    age = pulse_age_ratio(pulse, now)
    pop = 1.0 + EVENT_ICON_POP_SCALE * max(0.0, 1.0 - age / EVENT_ICON_POP_AGE_RATIO)
    return float(min_side) * base_ratio * pop * float(EVENT_ICON_SIZE_SCALE)


def event_icon_opacity(pulse, now):
    age = pulse_age_ratio(pulse, now)
    if age >= 1.0:
        return 0.0
    visibility = 0.68 + 0.32 * clamp(float(getattr(pulse, "strength", 0.0)))
    if age <= EVENT_ICON_HOLD_AGE_RATIO:
        return visibility
    fade = (1.0 - age) / max(1e-6, 1.0 - EVENT_ICON_HOLD_AGE_RATIO)
    return visibility * (fade ** 1.25)


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


def pulse_ripple_radius(pulse, now, min_side):
    age = pulse_age_ratio(pulse, now)
    return min_side * (0.38 + 0.16 * age)


def watercolor_pulse_seed(pulse):
    strength_key = int(round(clamp(float(pulse.strength)) * 1000))
    time_key = int(round(float(pulse.created_at) * 1000))
    return ((int(pulse.sector) + 1) * 1_000_003 + time_key * 9_176 + strength_key * 37) & 0xFFFFFFFF


def watercolor_blob_specs(pulse, now, min_side, blob_count=WATERCOLOR_BLOBS):
    age = pulse_age_ratio(pulse, now)
    strength = clamp(float(pulse.strength))
    opacity = pulse_opacity(pulse, now)
    base_distance = max(
        pulse_ripple_radius(pulse, now, min_side),
        min_side * WATERCOLOR_INNER_SAFE_RATIO,
    )
    center_angle = sector_mid_angle_deg(pulse.sector)
    rng = random.Random(watercolor_pulse_seed(pulse))
    specs = []
    for index in range(blob_count):
        angle_offset = rng.uniform(-WATERCOLOR_ANGLE_SPREAD, WATERCOLOR_ANGLE_SPREAD)
        flow_deg = rng.uniform(-WATERCOLOR_FLOW_DRIFT_DEG, WATERCOLOR_FLOW_DRIFT_DEG)
        distance = base_distance + min_side * (0.006 * index + rng.uniform(0.0, 0.026))
        radius = min_side * rng.uniform(0.038, 0.078) * (0.82 + 0.58 * strength) * (0.9 + 0.28 * age)
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


def arc_start_deg_for_position(position):
    """Return QPainter.drawArc start degrees for clock-like radar positions.

    QPainter uses 0° at 3 o'clock and positive angles counter-clockwise.
    Radar positions are clock-like: 0=front/top, 3=right, 6=rear/bottom, 9=left.
    """
    angle = 75 - position * 30
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


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
    timestamp = time.time() if now is None else float(now)
    base = Path(directory or ROLLING_CAPTURE_DIR).expanduser()
    local = time.localtime(timestamp)
    millis = int((timestamp - int(timestamp)) * 1000)
    return base / f"soundradar-rolling-{time.strftime('%Y%m%d-%H%M%S', local)}-{millis:03d}.wav"


def rolling_capture_metadata_path(audio_path):
    return Path(audio_path).with_suffix(".json")


def rolling_capture_metadata_payload(
    snapshot,
    *,
    audio_path,
    saved_at=None,
    prediction=None,
    display_debug=None,
    threshold_profile_name=None,
):
    from sound_model.capture_direction_sample import capture_sanity_lines, channel_peak_summary

    audio = np.asarray(snapshot.audio, dtype=np.float32)
    saved_at = time.time() if saved_at is None else float(saved_at)
    payload = {
        "schema_version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(saved_at)),
        "audio_path": str(Path(audio_path)),
        "sample_rate": int(snapshot.sample_rate),
        "channel_count": int(snapshot.channel_count),
        "duration_seconds": float(audio.shape[0]) / float(snapshot.sample_rate) if snapshot.sample_rate else 0.0,
        "start_capture_time": snapshot.start_capture_time,
        "end_capture_time": snapshot.end_capture_time,
        "threshold_profile": threshold_profile_name,
        "peak_summary": channel_peak_summary(audio),
        "sanity_lines": capture_sanity_lines(audio, expected_channels=snapshot.channel_count),
    }
    if prediction is not None:
        payload["prediction"] = prediction.to_jsonable() if hasattr(prediction, "to_jsonable") else None
        payload["hud_summary_lines"] = direction_event_debug_lines(
            prediction,
            threshold=AST_DIRECTION_EVENT_THRESHOLD,
            display_debug=display_debug,
        )
    return payload


def write_rolling_capture_snapshot(
    snapshot,
    *,
    directory=None,
    now=None,
    prediction=None,
    display_debug=None,
    threshold_profile_name=None,
):
    from sound_model.audio_features import write_wav

    audio_path = rolling_capture_output_path(directory, now=now)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_path, np.asarray(snapshot.audio, dtype=np.float32), snapshot.sample_rate)
    metadata = rolling_capture_metadata_payload(
        snapshot,
        audio_path=audio_path,
        saved_at=now,
        prediction=prediction,
        display_debug=display_debug,
        threshold_profile_name=threshold_profile_name,
    )
    metadata_path = rolling_capture_metadata_path(audio_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return audio_path, metadata_path


def consume_rolling_capture_trigger(path=None):
    path = Path(path or ROLLING_CAPTURE_TRIGGER_PATH).expanduser()
    if not str(path):
        return False
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        pass
    return True


def write_rolling_capture_trigger(path=None):
    path = Path(path or ROLLING_CAPTURE_TRIGGER_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()), encoding="utf-8")
    return path


class DirectionEventRuntime:
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
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self.window_seconds = float(window_seconds)
        self.max_samples = max(1, int(self.sample_rate * self.window_seconds))
        self.interval_seconds = float(interval_seconds)
        self.top_k = int(top_k)
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.compile_model = compile_model
        self.teacher_model = teacher_model
        self.model_id = model_id
        self.channel_map = dict(channel_map) if channel_map is not None else None
        self._score_fn = score_fn or self._score_with_ast_teacher
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="soundradar-ast")
        self._latency_clock = latency_clock or time.perf_counter
        self._future = None
        self._future_capture_time = None
        self._teacher = None
        self._warmup_future = None
        self._last_submit_time = -float("inf")
        self._audio = np.zeros((0, self.channel_count), dtype=np.float32)
        self.latest_audio_capture_time = None
        self.latest_latency_ms = None
        self.latest_prediction = None
        self.disabled_reason = None
        self.resolved_device = None
        self.resolved_dtype = None
        if warmup and score_fn is None:
            self._warmup_future = self._executor.submit(self._warmup_ast_teacher)
            self.poll_warmup()

    def append_blocks(self, blocks, capture_time=None):
        prepared = []
        for block in blocks:
            audio = np.asarray(block, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            if audio.ndim != 2 or audio.shape[0] == 0:
                continue
            if audio.shape[1] < self.channel_count:
                padded = np.zeros((audio.shape[0], self.channel_count), dtype=np.float32)
                padded[:, : audio.shape[1]] = audio
                audio = padded
            prepared.append(audio[:, : self.channel_count])
        if prepared:
            self._audio = np.concatenate([self._audio, *prepared], axis=0)[-self.max_samples :]
            if capture_time is not None:
                self.latest_audio_capture_time = float(capture_time)

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        try:
            self.latest_prediction = self._future.result()
            if self._future_capture_time is not None:
                self.latest_latency_ms = max(0.0, (self._latency_clock() - self._future_capture_time) * 1000.0)
        except Exception as exc:
            self.disabled_reason = str(exc)
            print(f"Direction event teacher disabled: {exc}", file=sys.stderr)
            self.latest_prediction = None
        finally:
            self._future = None
            self._future_capture_time = None
        return self.latest_prediction

    def poll_warmup(self):
        if self._warmup_future is None:
            return True
        if not self._warmup_future.done():
            return False
        try:
            self._warmup_future.result()
        except Exception as exc:
            self.disabled_reason = str(exc)
            print(f"Direction event teacher disabled during warmup: {exc}", file=sys.stderr)
        finally:
            self._warmup_future = None
        return self.disabled_reason is None

    def maybe_submit(self, now):
        prediction = self.poll()
        self.poll_warmup()
        if self.disabled_reason is not None or self._future is not None or self._warmup_future is not None:
            return prediction
        if self._audio.shape[0] < self.max_samples:
            return prediction
        if now - self._last_submit_time < self.interval_seconds:
            return prediction
        window = np.array(self._audio, copy=True)
        self._last_submit_time = now
        self._future_capture_time = self.latest_audio_capture_time
        self._future = self._executor.submit(self._score_fn, window, self.sample_rate, self.top_k, "<live>")
        return self.poll() if self._future.done() else prediction

    def _ensure_ast_teacher(self):
        from sound_model.ast_teacher import create_audio_event_teacher

        if self._teacher is None:
            self._teacher = create_audio_event_teacher(
                self.teacher_model,
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
                compile_model=self.compile_model,
            )
        resolved_device = str(getattr(self._teacher, "device", self.device))
        resolved_dtype = str(getattr(self._teacher, "dtype", self.dtype)).replace("torch.", "")
        if self.resolved_device != resolved_device or self.resolved_dtype != resolved_dtype:
            self.resolved_device = resolved_device
            self.resolved_dtype = resolved_dtype
            print(f"Direction event teacher {self.teacher_model} device: {resolved_device} dtype: {resolved_dtype}")
        return self._teacher

    def _warmup_ast_teacher(self):
        teacher = self._ensure_ast_teacher()
        warmup = getattr(teacher, "warmup_direction_batch", None)
        if callable(warmup):
            warmup(sample_rate=self.sample_rate, seconds=self.window_seconds, direction_count=7)

    def _score_with_ast_teacher(self, audio, sample_rate, top_k, source_path):
        from sound_model.direction_events import score_direction_events

        teacher = self._ensure_ast_teacher()
        return score_direction_events(
            audio,
            sample_rate,
            teacher,
            top_k=top_k,
            source_path=source_path,
            channel_map=self.channel_map,
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


def build_channel_mapping(channel_count, output_channel_count=None):
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


def positive_difference(primary, secondary):
    return max(0.0, float(primary) - float(secondary))


def directional_difference(primary, secondary, min_ratio=None):
    diff = positive_difference(primary, secondary)
    if diff <= 0:
        return 0.0
    min_ratio = maxdifmain if min_ratio is None else min_ratio
    baseline = min(float(primary), float(secondary))
    if baseline <= 0:
        return diff
    return diff if diff / baseline > min_ratio else 0.0


def is_directional_louder(primary, secondary, current_max):
    return directional_difference(primary, secondary) > current_max


def centered_pair_strength(first, second, balance_tolerance=0.25):
    strongest = max(float(first), float(second))
    if strongest <= 0:
        return 0.0
    if abs(float(first) - float(second)) / strongest > balance_tolerance:
        return 0.0
    return (float(first) + float(second)) / 2


def mapped_channel_value(values, channel_map, key):
    index = channel_map.get(key)
    if index is None:
        return 0.0
    return float(values[index])


def compute_direction_levels(max_values, channel_map):
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
    levels[1] = directional_difference(front_right, front_left)
    levels[2] = centered_pair_strength(front_right, right)
    levels[3] = right
    levels[4] = centered_pair_strength(right, rear_right)
    levels[5] = directional_difference(rear_right, rear_left)
    levels[6] = centered_pair_strength(rear_left, rear_right)
    levels[7] = directional_difference(rear_left, rear_right)
    levels[8] = centered_pair_strength(rear_left, left)
    levels[9] = left
    levels[10] = centered_pair_strength(left, front_left)
    levels[11] = directional_difference(front_left, front_right)
    return levels


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
