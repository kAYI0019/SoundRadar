"""Qt-free rolling audio buffer and capture-file helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from sound_model.audio_features import write_wav
from sound_model.capture_direction_sample import capture_sanity_lines, channel_peak_summary


DEFAULT_ROLLING_CAPTURE_SECONDS = 5.0


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
    def __init__(self, sample_rate, channel_count, seconds=DEFAULT_ROLLING_CAPTURE_SECONDS):
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


def rolling_capture_output_path(directory, now=None):
    timestamp = time.time() if now is None else float(now)
    base = Path(directory).expanduser()
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
    hud_summary_lines=None,
    threshold_profile_name=None,
):
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
        payload["hud_summary_lines"] = list(hud_summary_lines or ())
    return payload


def write_rolling_capture_snapshot(
    snapshot,
    *,
    directory,
    now=None,
    prediction=None,
    hud_summary_lines=None,
    threshold_profile_name=None,
):
    audio_path = rolling_capture_output_path(directory, now=now)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_path, np.asarray(snapshot.audio, dtype=np.float32), snapshot.sample_rate)
    metadata = rolling_capture_metadata_payload(
        snapshot,
        audio_path=audio_path,
        saved_at=now,
        prediction=prediction,
        hud_summary_lines=hud_summary_lines,
        threshold_profile_name=threshold_profile_name,
    )
    metadata_path = rolling_capture_metadata_path(audio_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return audio_path, metadata_path


def consume_rolling_capture_trigger(path):
    path = Path(path).expanduser()
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        pass
    return True


def write_rolling_capture_trigger(path, now=None):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.time() if now is None else float(now)
    path.write_text(str(timestamp), encoding="utf-8")
    return path
