"""PyQt GUI for capturing real multichannel SoundRadar tuning samples."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys

from PyQt5 import QtCore, QtWidgets

from .audio_features import read_wav, write_wav
from .capture_direction_sample import (
    capture_sanity_lines,
    channel_peak_summary,
    choose_recording_channels,
    device_default_sample_rate,
    device_input_channels,
    device_sanity_lines,
    record_input_audio,
    select_input_device,
)
from .direction_events import DIRECTION_NAMES


EVENT_SCORE_ORDER = ("gunshot", "vehicle", "footstep", "explosion")
DIRECTION_LABELS = {
    "front_left": "FL",
    "front": "F",
    "front_right": "FR",
    "left": "L",
    "right": "R",
    "rear_left": "RL",
    "rear_right": "RR",
}
EVENT_LABELS = {
    "gunshot": "GUN",
    "vehicle": "VEH",
    "footstep": "FOOT",
    "explosion": "EXP",
}
SAMPLE_TAGS = ("gunshot", "vehicle", "footstep", "unknown", "bad sample")
LIBRARY_FIELDS = (
    "created_at",
    "audio_path",
    "tag",
    "notes",
    "analysis_path",
    "peak_summary",
)


def default_capture_path(now=None, directory: str | Path = "/tmp") -> Path:
    now = now or datetime.now()
    return Path(directory) / f"soundradar-capture-{now.strftime('%Y%m%d-%H%M%S')}.wav"


def format_device_label(index: int, device_info) -> str:
    name = str(device_info.get("name", "Unknown"))
    inputs = device_input_channels(device_info)
    sample_rate = device_default_sample_rate(device_info)
    return f"{index}: {name} ({inputs}ch, {sample_rate} Hz)"


def direction_events_command(path: str | Path, *, teacher_model: str = "ast", device: str = "auto", top_k: int = 5) -> str:
    return " ".join(
        shlex.quote(part)
        for part in (
            ".venv/bin/python",
            "-m",
            "sound_model.direction_events",
            str(path),
            "--teacher-model",
            teacher_model,
            "--device",
            device,
            "--top-k",
            str(top_k),
        )
    )


def analysis_result_path(audio_path: str | Path) -> Path:
    return Path(audio_path).with_suffix(".analysis.json")


def profile_comparison_result_path(audio_path: str | Path) -> Path:
    return Path(audio_path).with_suffix(".profile-comparison.json")


def default_library_path(directory: str | Path | None = None) -> Path:
    base = Path(directory) if directory is not None else Path.home() / "SoundRadarSamples"
    return base / "sample_library.csv"


def available_threshold_profiles() -> tuple[str, ...]:
    try:
        import soundRadar

        return soundRadar.threshold_profile_names()
    except Exception:
        return ("default",)


def trigger_rolling_capture(path: str | Path | None = None) -> Path:
    import soundRadar

    if path is None:
        soundRadar.apply_runtime_config(soundRadar.load_runtime_config())
    return soundRadar.write_rolling_capture_trigger(path)


def compact_score(score) -> str:
    return f"{max(0.0, min(1.0, float(score))):.2f}".lstrip("0")


def _format_top_labels(labels, max_labels: int = 3) -> str:
    formatted = []
    for item in list(labels or ())[: int(max_labels)]:
        label = str(item.get("label", "") if isinstance(item, dict) else item)
        score = item.get("score", None) if isinstance(item, dict) else None
        if score is None:
            formatted.append(label)
        else:
            formatted.append(f"{label} {compact_score(score)}")
    return ", ".join(formatted) if formatted else "--"


def direction_score_summary_lines(prediction) -> list[str]:
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
    top_labels_by_direction = getattr(prediction, "top_labels_by_direction", {}) or {}
    lines = ["direction scores:"]
    for direction in DIRECTION_NAMES:
        scores = scores_by_direction.get(direction, {}) or {}
        active = "/".join(active_by_direction.get(direction, ())) or "--"
        score_text = " ".join(
            f"{EVENT_LABELS[event_name]} {compact_score(scores.get(event_name, 0.0))}"
            for event_name in EVENT_SCORE_ORDER
        )
        top_labels = _format_top_labels(top_labels_by_direction.get(direction, ()))
        lines.append(f"{DIRECTION_LABELS[direction]:>2} active {active:<18} {score_text} top {top_labels}")
    return lines


def displayed_events_for_prediction(prediction, *, threshold_profile: str = "default"):
    import soundRadar

    soundRadar.apply_threshold_profile(threshold_profile)
    display_debug = {}
    pulses = soundRadar.create_pulses_from_direction_events(
        prediction,
        now=0.0,
        threshold=soundRadar.AST_DIRECTION_EVENT_THRESHOLD,
        cooldown=soundRadar.AST_DIRECTION_EVENT_COOLDOWN,
        display_debug=display_debug,
    )
    direction_by_sector = {sector: direction for direction, sector in soundRadar.DIRECTION_EVENT_SECTORS.items()}
    events = [
        {
            "direction": direction_by_sector.get(int(pulse.sector), str(pulse.sector)),
            "sector": int(pulse.sector),
            "event": str(pulse.kind),
            "score": float(pulse.strength),
        }
        for pulse in pulses
        if soundRadar.is_classified_event_kind(getattr(pulse, "kind", ""))
    ]
    return events, display_debug.get("debug")


def _display_debug_for_prediction(prediction, *, threshold_profile: str = "default"):
    _, display_debug = displayed_events_for_prediction(prediction, threshold_profile=threshold_profile)
    return display_debug


def hud_summary_lines(prediction, *, display_debug=None, threshold_profile: str = "default") -> list[str]:
    import soundRadar

    soundRadar.apply_threshold_profile(threshold_profile)
    display_debug = (
        _display_debug_for_prediction(prediction, threshold_profile=threshold_profile)
        if display_debug is None
        else display_debug
    )
    return soundRadar.direction_event_debug_lines(
        prediction,
        threshold=soundRadar.AST_DIRECTION_EVENT_THRESHOLD,
        display_debug=display_debug,
    )


def prediction_summary_text(prediction, *, threshold_profile: str = "default") -> str:
    lines = ["HUD summary:"]
    lines.extend(hud_summary_lines(prediction, threshold_profile=threshold_profile))
    lines.append("")
    lines.extend(direction_score_summary_lines(prediction))
    return "\n".join(lines)


def max_event_score(prediction, event_name: str) -> float:
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    return max(
        (float((scores or {}).get(event_name, 0.0)) for scores in scores_by_direction.values()),
        default=0.0,
    )


def profile_summary_payload(prediction, profile_name: str) -> dict[str, object]:
    events, display_debug = displayed_events_for_prediction(prediction, threshold_profile=profile_name)
    hud_lines = hud_summary_lines(prediction, display_debug=display_debug, threshold_profile=profile_name)
    return {
        "profile": str(profile_name),
        "shown_events": events,
        "shown_event_count": len(events),
        "shown_event_types": sorted({event["event"] for event in events}),
        "max_scores": {event_name: max_event_score(prediction, event_name) for event_name in EVENT_SCORE_ORDER},
        "hud_summary_lines": hud_lines,
        "gunshot_display": _gunshot_debug_payload(display_debug),
    }


def compare_threshold_profiles(prediction, profiles=None) -> list[dict[str, object]]:
    profiles = tuple(profiles or available_threshold_profiles())
    return [profile_summary_payload(prediction, profile_name) for profile_name in profiles]


def format_profile_comparison(comparisons) -> str:
    lines = ["profile comparison:"]
    for summary in comparisons:
        shown = summary.get("shown_events", ())
        if shown:
            event_text = ", ".join(
                f"{event['event']}:{DIRECTION_LABELS.get(event['direction'], event['direction'])} {compact_score(event['score'])}"
                for event in shown
            )
        else:
            event_text = "--"
        gunshot_display = summary.get("gunshot_display", {}) or {}
        gunshot_shown = "/".join(
            DIRECTION_LABELS.get(direction, direction)
            for direction in gunshot_display.get("shown_directions", ())
        ) or "--"
        gunshot_suppressed = "/".join(
            DIRECTION_LABELS.get(direction, direction)
            for direction in gunshot_display.get("spatially_suppressed_directions", ())
        ) or "--"
        lines.append(
            f"{summary['profile']:<10} shown {event_text} "
            f"gun show {gunshot_shown} sup {gunshot_suppressed}"
        )
    return "\n".join(lines)


def profile_comparison_payload(
    prediction,
    *,
    audio_path: str | Path,
    teacher_model: str,
    device: str,
    top_k: int,
    peak_summary: str | None = None,
    profiles=None,
    analyzed_at: datetime | None = None,
) -> dict[str, object]:
    analyzed_at = analyzed_at or datetime.now()
    comparisons = compare_threshold_profiles(prediction, profiles=profiles)
    return {
        "schema_version": 1,
        "analyzed_at": analyzed_at.isoformat(timespec="seconds"),
        "audio_path": str(Path(audio_path)),
        "peak_summary": peak_summary,
        "teacher_model": str(teacher_model),
        "device": str(device),
        "top_k": int(top_k),
        "profiles": comparisons,
        "prediction": prediction.to_jsonable() if hasattr(prediction, "to_jsonable") else None,
    }


def _prediction_scores_payload(prediction) -> dict[str, object]:
    scores_by_direction = getattr(prediction, "direction_event_scores", {}) or {}
    raw_scores_by_direction = getattr(prediction, "raw_direction_event_scores", {}) or {}
    active_by_direction = getattr(prediction, "active_events_by_direction", {}) or {}
    top_labels_by_direction = getattr(prediction, "top_labels_by_direction", {}) or {}
    label_evidence_by_direction = getattr(prediction, "event_label_evidence_by_direction", {}) or {}
    resolver_evidence_by_direction = getattr(prediction, "vehicle_gun_evidence_by_direction", {}) or {}
    resolver_decisions_by_direction = getattr(prediction, "vehicle_gun_decisions_by_direction", {}) or {}
    score_semantics_by_direction = getattr(prediction, "label_score_semantics_by_direction", {}) or {}
    payload = {}
    for direction in DIRECTION_NAMES:
        scores = scores_by_direction.get(direction, {}) or {}
        raw_scores = raw_scores_by_direction.get(direction, scores) or {}
        resolver_evidence = resolver_evidence_by_direction.get(direction)
        resolver_decision = resolver_decisions_by_direction.get(direction)
        payload[direction] = {
            "scores": {event_name: float(scores.get(event_name, 0.0)) for event_name in EVENT_SCORE_ORDER},
            "raw_scores": {event_name: float(raw_scores.get(event_name, 0.0)) for event_name in EVENT_SCORE_ORDER},
            "active_events": list(active_by_direction.get(direction, ())),
            "top_labels": [
                {
                    "label": str(item.get("label", "")),
                    "score": float(item.get("score", 0.0)),
                }
                for item in top_labels_by_direction.get(direction, ())
                if isinstance(item, dict)
            ],
            "event_label_evidence": label_evidence_by_direction.get(direction, {}),
            "label_score_semantics": str(score_semantics_by_direction.get(direction, "unknown")),
            "vehicle_gun_evidence": asdict(resolver_evidence) if is_dataclass(resolver_evidence) else resolver_evidence,
            "vehicle_gun_decision": asdict(resolver_decision) if is_dataclass(resolver_decision) else resolver_decision,
        }
    return payload


def _gunshot_debug_payload(display_debug) -> dict[str, object]:
    if display_debug is None:
        return {
            "candidate_scores": [],
            "shown_directions": [],
            "spatially_suppressed_directions": [],
            "global_cooldown_blocked_directions": [],
            "sector_cooldown_blocked_directions": [],
        }
    decision = display_debug.gunshot_decision
    return {
        "candidate_scores": [
            {"direction": str(direction), "score": float(score)}
            for direction, score in decision.candidate_scores
        ],
        "shown_directions": list(display_debug.gunshot_emitted_directions),
        "spatially_suppressed_directions": sorted(decision.spatially_suppressed_directions),
        "global_cooldown_blocked_directions": list(display_debug.gunshot_global_cooldown_blocked_directions),
        "sector_cooldown_blocked_directions": list(display_debug.gunshot_sector_cooldown_blocked_directions),
    }


def peak_summary_for_file(audio_path: str | Path) -> str | None:
    try:
        audio, _ = read_wav(audio_path)
    except Exception:
        return None
    return channel_peak_summary(audio)


def prediction_result_payload(
    prediction,
    *,
    audio_path: str | Path,
    teacher_model: str,
    device: str,
    top_k: int,
    peak_summary: str | None = None,
    threshold_profile: str = "default",
    analyzed_at: datetime | None = None,
) -> dict[str, object]:
    analyzed_at = analyzed_at or datetime.now()
    display_debug = _display_debug_for_prediction(prediction, threshold_profile=threshold_profile)
    hud_lines = hud_summary_lines(prediction, display_debug=display_debug, threshold_profile=threshold_profile)
    return {
        "schema_version": 1,
        "analyzed_at": analyzed_at.isoformat(timespec="seconds"),
        "audio_path": str(Path(audio_path)),
        "peak_summary": peak_summary,
        "teacher_model": str(teacher_model),
        "device": str(device),
        "top_k": int(top_k),
        "threshold_profile": str(threshold_profile),
        "prediction": prediction.to_jsonable() if hasattr(prediction, "to_jsonable") else None,
        "hud_summary_lines": hud_lines,
        "direction_scores": _prediction_scores_payload(prediction),
        "gunshot_display": _gunshot_debug_payload(display_debug),
    }


def write_prediction_result_json(
    audio_path: str | Path,
    payload: dict[str, object],
    *,
    result_path: str | Path | None = None,
) -> Path:
    path = Path(result_path) if result_path is not None else analysis_result_path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_profile_comparison_json(
    audio_path: str | Path,
    payload: dict[str, object],
    *,
    result_path: str | Path | None = None,
) -> Path:
    path = Path(result_path) if result_path is not None else profile_comparison_result_path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def sample_library_record(
    *,
    audio_path: str | Path,
    tag: str,
    notes: str = "",
    analysis_path: str | Path | None = None,
    peak_summary: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, str]:
    if tag not in SAMPLE_TAGS:
        raise ValueError(f"unknown sample tag: {tag}")
    created_at = created_at or datetime.now()
    return {
        "created_at": created_at.isoformat(timespec="seconds"),
        "audio_path": str(Path(audio_path)),
        "tag": tag,
        "notes": str(notes),
        "analysis_path": "" if analysis_path is None else str(Path(analysis_path)),
        "peak_summary": "" if peak_summary is None else str(peak_summary),
    }


def append_sample_library_record(library_path: str | Path, record: dict[str, str]) -> Path:
    path = Path(library_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LIBRARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in LIBRARY_FIELDS})
    return path


def _plain_device_info(device_info):
    return {
        "name": str(device_info.get("name", "")),
        "max_input_channels": int(device_info.get("max_input_channels", 0) or 0),
        "max_output_channels": int(device_info.get("max_output_channels", 0) or 0),
        "default_samplerate": float(device_info.get("default_samplerate", 48_000) or 48_000),
    }


class CaptureWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, str, str, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, *, device_index: int, device_name: str, seconds: float, output_path: Path, channels: int, sample_rate: int):
        super().__init__()
        self.device_index = int(device_index)
        self.device_name = str(device_name)
        self.seconds = float(seconds)
        self.output_path = Path(output_path)
        self.channels = int(channels)
        self.sample_rate = int(sample_rate)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            import sounddevice as sd

            audio = record_input_audio(
                sd,
                device=self.device_index,
                seconds=self.seconds,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            write_wav(self.output_path, audio, self.sample_rate)
            summary = channel_peak_summary(audio)
            sanity = "\n".join(capture_sanity_lines(audio, expected_channels=self.channels))
            self.finished.emit(str(self.output_path), summary, self.device_name, sanity)
        except Exception as exc:
            self.failed.emit(str(exc))


class AnalysisWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, str, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        audio_path: Path,
        teacher_model: str,
        device: str,
        top_k: int,
        peak_summary: str | None = None,
        threshold_profile: str = "default",
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.teacher_model = str(teacher_model)
        self.device = str(device)
        self.top_k = int(top_k)
        self.peak_summary = peak_summary
        self.threshold_profile = str(threshold_profile)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from .direction_events import predict_direction_events_file

            prediction = predict_direction_events_file(
                self.audio_path,
                teacher_model=self.teacher_model,
                device=self.device,
                top_k=self.top_k,
            )
            peak_summary = self.peak_summary if self.peak_summary is not None else peak_summary_for_file(self.audio_path)
            result_path = write_prediction_result_json(
                self.audio_path,
                prediction_result_payload(
                    prediction,
                    audio_path=self.audio_path,
                    teacher_model=self.teacher_model,
                    device=self.device,
                    top_k=self.top_k,
                    peak_summary=peak_summary,
                    threshold_profile=self.threshold_profile,
                ),
            )
            self.finished.emit(
                str(self.audio_path),
                prediction_summary_text(prediction, threshold_profile=self.threshold_profile),
                str(result_path),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ProfileCompareWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, str, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        audio_path: Path,
        teacher_model: str,
        device: str,
        top_k: int,
        peak_summary: str | None = None,
        profiles=None,
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.teacher_model = str(teacher_model)
        self.device = str(device)
        self.top_k = int(top_k)
        self.peak_summary = peak_summary
        self.profiles = tuple(profiles or available_threshold_profiles())

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from .direction_events import predict_direction_events_file

            prediction = predict_direction_events_file(
                self.audio_path,
                teacher_model=self.teacher_model,
                device=self.device,
                top_k=self.top_k,
            )
            peak_summary = self.peak_summary if self.peak_summary is not None else peak_summary_for_file(self.audio_path)
            payload = profile_comparison_payload(
                prediction,
                audio_path=self.audio_path,
                teacher_model=self.teacher_model,
                device=self.device,
                top_k=self.top_k,
                peak_summary=peak_summary,
                profiles=self.profiles,
            )
            result_path = write_profile_comparison_json(self.audio_path, payload)
            self.finished.emit(
                str(self.audio_path),
                format_profile_comparison(payload["profiles"]),
                str(result_path),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class CaptureWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._worker = None
        self._last_capture_path = None
        self._last_analysis_path = None
        self._last_peak_summary = None
        self.setWindowTitle("SoundRadar Capture")
        self.setMinimumWidth(620)

        self.device_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.seconds_spin = QtWidgets.QDoubleSpinBox()
        self.seconds_spin.setRange(1.0, 300.0)
        self.seconds_spin.setDecimals(1)
        self.seconds_spin.setSingleStep(5.0)
        self.seconds_spin.setValue(20.0)

        self.channels_spin = QtWidgets.QSpinBox()
        self.channels_spin.setRange(1, 64)
        self.channels_spin.setValue(16)
        self.channels_auto = QtWidgets.QCheckBox("Auto")
        self.channels_auto.setChecked(True)

        self.sample_rate_spin = QtWidgets.QSpinBox()
        self.sample_rate_spin.setRange(8_000, 192_000)
        self.sample_rate_spin.setSingleStep(1_000)
        self.sample_rate_spin.setValue(48_000)
        self.sample_rate_auto = QtWidgets.QCheckBox("Auto")
        self.sample_rate_auto.setChecked(True)
        self.sanity_label = QtWidgets.QLabel("")
        self.sanity_label.setWordWrap(True)

        self.output_edit = QtWidgets.QLineEdit(str(default_capture_path()))
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.record_button = QtWidgets.QPushButton("Record")
        self.record_button.setDefault(True)
        self.save_rolling_button = QtWidgets.QPushButton("Save Last 5s")
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.compare_button = QtWidgets.QPushButton("Compare Profiles")

        self.teacher_combo = QtWidgets.QComboBox()
        self.teacher_combo.addItems(["ast", "efficientat-mn10", "efficientat-mn20"])
        self.analysis_device_combo = QtWidgets.QComboBox()
        self.analysis_device_combo.addItems(["auto", "cpu", "mps", "cuda"])
        self.threshold_profile_combo = QtWidgets.QComboBox()
        self.threshold_profile_combo.addItems(list(available_threshold_profiles()))
        self.top_k_spin = QtWidgets.QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)

        self.tag_combo = QtWidgets.QComboBox()
        self.tag_combo.addItems(list(SAMPLE_TAGS))
        self.notes_edit = QtWidgets.QLineEdit()
        self.library_edit = QtWidgets.QLineEdit(str(default_library_path()))
        self.library_browse_button = QtWidgets.QPushButton("Browse")
        self.save_tag_button = QtWidgets.QPushButton("Save Tag")

        self.status_label = QtWidgets.QLabel("Ready")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)

        self._build_layout()
        self._connect_signals()
        self.refresh_devices()

    def _build_layout(self):
        layout = QtWidgets.QVBoxLayout(self)

        device_row = QtWidgets.QHBoxLayout()
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_button)
        layout.addLayout(self._labeled_row("Input", device_row))

        seconds_row = QtWidgets.QHBoxLayout()
        seconds_row.addWidget(self.seconds_spin)
        seconds_row.addStretch(1)
        layout.addLayout(self._labeled_row("Seconds", seconds_row))

        channels_row = QtWidgets.QHBoxLayout()
        channels_row.addWidget(self.channels_spin)
        channels_row.addWidget(self.channels_auto)
        channels_row.addStretch(1)
        layout.addLayout(self._labeled_row("Channels", channels_row))

        rate_row = QtWidgets.QHBoxLayout()
        rate_row.addWidget(self.sample_rate_spin)
        rate_row.addWidget(self.sample_rate_auto)
        rate_row.addStretch(1)
        layout.addLayout(self._labeled_row("Sample Rate", rate_row))

        sanity_row = QtWidgets.QHBoxLayout()
        sanity_row.addWidget(self.sanity_label, 1)
        layout.addLayout(self._labeled_row("Sanity", sanity_row))

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.browse_button)
        layout.addLayout(self._labeled_row("Output", output_row))

        analyze_row = QtWidgets.QHBoxLayout()
        analyze_row.addWidget(QtWidgets.QLabel("Teacher"))
        analyze_row.addWidget(self.teacher_combo)
        analyze_row.addWidget(QtWidgets.QLabel("Device"))
        analyze_row.addWidget(self.analysis_device_combo)
        analyze_row.addWidget(QtWidgets.QLabel("Profile"))
        analyze_row.addWidget(self.threshold_profile_combo)
        analyze_row.addWidget(QtWidgets.QLabel("Top K"))
        analyze_row.addWidget(self.top_k_spin)
        analyze_row.addStretch(1)
        analyze_row.addWidget(self.analyze_button)
        analyze_row.addWidget(self.compare_button)
        layout.addLayout(self._labeled_row("Analyze", analyze_row))

        tag_row = QtWidgets.QHBoxLayout()
        tag_row.addWidget(self.tag_combo)
        tag_row.addWidget(QtWidgets.QLabel("Notes"))
        tag_row.addWidget(self.notes_edit, 1)
        tag_row.addWidget(QtWidgets.QLabel("Library"))
        tag_row.addWidget(self.library_edit, 1)
        tag_row.addWidget(self.library_browse_button)
        tag_row.addWidget(self.save_tag_button)
        layout.addLayout(self._labeled_row("Tag", tag_row))

        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.save_rolling_button)
        action_row.addWidget(self.record_button)
        layout.addLayout(action_row)
        layout.addWidget(self.log)

    def _labeled_row(self, label, row_layout):
        outer = QtWidgets.QHBoxLayout()
        label_widget = QtWidgets.QLabel(label)
        label_widget.setMinimumWidth(92)
        outer.addWidget(label_widget)
        outer.addLayout(row_layout, 1)
        return outer

    def _connect_signals(self):
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.device_combo.currentIndexChanged.connect(self._sync_device_defaults)
        self.channels_auto.toggled.connect(self._sync_device_defaults)
        self.channels_spin.valueChanged.connect(self._sync_sanity_label)
        self.sample_rate_auto.toggled.connect(self._sync_device_defaults)
        self.browse_button.clicked.connect(self.browse_output)
        self.record_button.clicked.connect(self.start_recording)
        self.save_rolling_button.clicked.connect(self.request_rolling_capture)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.compare_button.clicked.connect(self.start_profile_comparison)
        self.library_browse_button.clicked.connect(self.browse_library)
        self.save_tag_button.clicked.connect(self.save_tag)

    def append_log(self, text):
        self.log.appendPlainText(str(text))

    def refresh_devices(self):
        try:
            import sounddevice as sd

            devices = [_plain_device_info(device) for device in sd.query_devices()]
            self.device_combo.clear()
            selected_index = None
            try:
                selected_index, _ = select_input_device(devices)
            except ValueError:
                pass

            for index, device in enumerate(devices):
                if device_input_channels(device) <= 0:
                    continue
                self.device_combo.addItem(format_device_label(index, device), (index, device))
                if selected_index == index:
                    self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
            if self.device_combo.count() == 0:
                self.status_label.setText("No input devices")
                self.append_log("No input devices found. Check BlackHole/Loopback setup.")
            else:
                self.status_label.setText("Ready")
                self._sync_device_defaults()
        except Exception as exc:
            self.status_label.setText("Device error")
            self.append_log(f"Device refresh failed: {exc}")

    def selected_device(self):
        data = self.device_combo.currentData()
        if data is None:
            raise ValueError("select an input device")
        return data

    def _sync_device_defaults(self):
        data = self.device_combo.currentData()
        if data is None:
            return
        _, device = data
        max_channels = max(1, device_input_channels(device))
        self.channels_spin.setMaximum(max_channels)
        if self.channels_auto.isChecked():
            self.channels_spin.setValue(max_channels)
            self.channels_spin.setEnabled(False)
        else:
            self.channels_spin.setEnabled(True)
        if self.sample_rate_auto.isChecked():
            self.sample_rate_spin.setValue(device_default_sample_rate(device))
            self.sample_rate_spin.setEnabled(False)
        else:
            self.sample_rate_spin.setEnabled(True)
        self._sync_sanity_label()

    def _sync_sanity_label(self):
        data = self.device_combo.currentData()
        if data is None:
            self.sanity_label.setText("")
            return
        _, device = data
        self.sanity_label.setText(" | ".join(device_sanity_lines(device, selected_channels=self.channels_spin.value())))

    def browse_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Capture", self.output_edit.text(), "WAV files (*.wav)")
        if path:
            self.output_edit.setText(path)

    def browse_library(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Library", self.library_edit.text(), "CSV files (*.csv)")
        if path:
            self.library_edit.setText(path)

    def request_rolling_capture(self):
        try:
            path = trigger_rolling_capture()
        except Exception as exc:
            self.status_label.setText("Rolling trigger failed")
            self.append_log(f"Cannot request rolling capture: {exc}")
            return
        self.status_label.setText("Rolling requested")
        self.append_log(f"Rolling capture requested: {path}")

    def start_recording(self):
        if self._thread is not None:
            return
        try:
            device_index, device = self.selected_device()
            seconds = float(self.seconds_spin.value())
            output_path = Path(self.output_edit.text()).expanduser()
            channels = choose_recording_channels(device, None if self.channels_auto.isChecked() else self.channels_spin.value())
            sample_rate = device_default_sample_rate(device) if self.sample_rate_auto.isChecked() else int(self.sample_rate_spin.value())
        except Exception as exc:
            self.append_log(f"Cannot start: {exc}")
            return

        self.record_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.save_tag_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Recording...")
        self.append_log(f"Recording {seconds:.1f}s from {device['name']} ({channels}ch, {sample_rate} Hz)")
        for line in device_sanity_lines(device, selected_channels=channels):
            self.append_log(f"Sanity {line}")

        self._thread = QtCore.QThread(self)
        self._worker = CaptureWorker(
            device_index=device_index,
            device_name=device["name"],
            seconds=seconds,
            output_path=output_path,
            channels=channels,
            sample_rate=sample_rate,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._recording_finished)
        self._worker.failed.connect(self._recording_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    @QtCore.pyqtSlot(str, str, str, str)
    def _recording_finished(self, path, summary, device_name, sanity):
        self._last_capture_path = Path(path)
        self._last_analysis_path = None
        self._last_peak_summary = summary
        self.status_label.setText("Saved")
        self.append_log(f"Saved {path}")
        self.append_log(f"Device {device_name}")
        self.append_log(f"Peaks {summary}")
        if sanity:
            self.append_log(sanity)
        self.append_log(f"Analyze: {direction_events_command(path)}")
        self.output_edit.setText(str(default_capture_path()))

    @QtCore.pyqtSlot(str)
    def _recording_failed(self, message):
        self.status_label.setText("Failed")
        self.append_log(f"Recording failed: {message}")

    def analysis_path(self) -> Path:
        if self._last_capture_path is not None:
            return Path(self._last_capture_path)
        path = Path(self.output_edit.text()).expanduser()
        if path.exists():
            return path
        raise ValueError("record a sample first, or choose an existing WAV path")

    def start_analysis(self):
        if self._thread is not None:
            return
        try:
            audio_path = self.analysis_path()
            teacher_model = self.teacher_combo.currentText()
            device = self.analysis_device_combo.currentText()
            threshold_profile = self.threshold_profile_combo.currentText()
            top_k = int(self.top_k_spin.value())
        except Exception as exc:
            self.append_log(f"Cannot analyze: {exc}")
            return

        self.record_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.save_tag_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Analyzing...")
        self.append_log(f"Analyzing {audio_path}")
        self.append_log(f"Command: {direction_events_command(audio_path, teacher_model=teacher_model, device=device, top_k=top_k)}")

        self._thread = QtCore.QThread(self)
        self._worker = AnalysisWorker(
            audio_path=audio_path,
            teacher_model=teacher_model,
            device=device,
            top_k=top_k,
            peak_summary=self._last_peak_summary if self._last_capture_path == audio_path else None,
            threshold_profile=threshold_profile,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._analysis_finished)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    @QtCore.pyqtSlot(str, str, str)
    def _analysis_finished(self, path, summary, result_path):
        self._last_capture_path = Path(path)
        self._last_analysis_path = Path(result_path)
        self.status_label.setText("Analyzed")
        self.append_log(f"Analysis complete: {path}")
        self.append_log(f"Analysis saved: {result_path}")
        self.append_log(summary)

    @QtCore.pyqtSlot(str)
    def _analysis_failed(self, message):
        self.status_label.setText("Analysis failed")
        self.append_log(f"Analysis failed: {message}")

    def start_profile_comparison(self):
        if self._thread is not None:
            return
        try:
            audio_path = self.analysis_path()
            teacher_model = self.teacher_combo.currentText()
            device = self.analysis_device_combo.currentText()
            top_k = int(self.top_k_spin.value())
            profiles = available_threshold_profiles()
        except Exception as exc:
            self.append_log(f"Cannot compare profiles: {exc}")
            return

        self.record_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.save_tag_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Comparing...")
        self.append_log(f"Comparing profiles for {audio_path}")

        self._thread = QtCore.QThread(self)
        self._worker = ProfileCompareWorker(
            audio_path=audio_path,
            teacher_model=teacher_model,
            device=device,
            top_k=top_k,
            peak_summary=self._last_peak_summary if self._last_capture_path == audio_path else None,
            profiles=profiles,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._profile_comparison_finished)
        self._worker.failed.connect(self._profile_comparison_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    @QtCore.pyqtSlot(str, str, str)
    def _profile_comparison_finished(self, path, summary, result_path):
        self._last_capture_path = Path(path)
        self.status_label.setText("Compared")
        self.append_log(f"Profile comparison complete: {path}")
        self.append_log(f"Profile comparison saved: {result_path}")
        self.append_log(summary)

    @QtCore.pyqtSlot(str)
    def _profile_comparison_failed(self, message):
        self.status_label.setText("Comparison failed")
        self.append_log(f"Profile comparison failed: {message}")

    def save_tag(self):
        try:
            audio_path = self.analysis_path()
            record = sample_library_record(
                audio_path=audio_path,
                tag=self.tag_combo.currentText(),
                notes=self.notes_edit.text(),
                analysis_path=self._last_analysis_path,
                peak_summary=self._last_peak_summary if self._last_capture_path == audio_path else None,
            )
            library_path = append_sample_library_record(self.library_edit.text(), record)
        except Exception as exc:
            self.append_log(f"Cannot save tag: {exc}")
            return
        self.status_label.setText("Tagged")
        self.append_log(f"Tagged {audio_path} as {record['tag']}")
        self.append_log(f"Library {library_path}")

    def _cleanup_worker(self):
        self.record_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.compare_button.setEnabled(True)
        self.save_tag_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = CaptureWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
