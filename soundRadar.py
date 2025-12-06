import ctypes
import ctypes.util
from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class WatercolorBlob:
    angle_deg: float
    distance: float
    radius: float
    opacity: float
    stretch: float = 1.0
    rotation_deg: float = 0.0
    flow_deg: float = 0.0


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


def normalize_degrees(angle):
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def sector_mid_angle_deg(sector):
    return normalize_degrees(arc_start_deg_for_position(sector) + 15)


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


def watercolor_color(strength, opacity):
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


def pulse_color(strength, opacity):
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
        painter.setPen(QtGui.QPen(pulse_color(pulse.strength, opacity), line_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawArc(rect, int(start * 16), int(span * 16))

        inner_radius = radius - min_side * 0.045
        if inner_radius > min_side * 0.34:
            inner_rect = QtCore.QRectF(center_x - inner_radius, center_y - inner_radius, 2 * inner_radius, 2 * inner_radius)
            painter.setPen(QtGui.QPen(pulse_color(pulse.strength, opacity * 0.45), max(1.0, line_width * 0.55), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
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
                color = watercolor_color(pulse.strength, spec.opacity * (0.82 - 0.16 * trail_ratio))
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
            color = watercolor_color(pulse.strength, spec.opacity)
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
            color = watercolor_color(pulse.strength, spec.opacity)
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
            if RIPPLE_STYLE == "watercolor":
                self._paint_watercolor_pulse(painter, pulse, now, min_side, center_x, center_y)
            else:
                self._paint_arc_ripple(painter, pulse, now, min_side, center_x, center_y)
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
        self._previous_direction_levels = np.zeros(RADAR_SECTORS)
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
    q.put(indata.copy())


def getMaxSound(channel_count):
    max_values = np.zeros(channel_count, dtype=np.float32)
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        block_max = np.nanmax(np.abs(data), axis=0)
        max_values = np.maximum(max_values, block_max[:channel_count])
    return max_values / maxSoundValue


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


def updateRadar(radarObject):
    while True:
        time.sleep(refreshtime)
        max_values = initfilter(getMaxSound(n_channel), minThreshold)
        direction_levels = compute_direction_levels(max_values, mapping)
        now = time.time()

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
            radarObject.prune_pulses(now)
            radarObject._previous_direction_levels = np.array(direction_levels, copy=True)

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


mapping, channel_mode = build_channel_mapping(n_chans)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = create_main_window()
    device_id, device_info = select_input_device()
    configure_audio_mapping(device_info)
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
