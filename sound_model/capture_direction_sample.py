"""Capture real multichannel input samples for SoundRadar tuning.

This records from a virtual input such as BlackHole 16ch and writes a PCM WAV
that can be replayed through ``sound_model.direction_events``.  It is intended
for real game/app samples, complementing ``surround_probe`` which only verifies
channel routing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .audio_features import write_wav


DEFAULT_INPUT_KEYWORDS = (
    "BlackHole 16ch",
    "BlackHole",
    "Loopback",
    "CABLE Output",
    "VB-Cable",
    "VB-Audio Virtual Cable",
    "VB-Audio",
)


def parse_device_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def device_input_channels(device_info: Mapping[str, object]) -> int:
    return int(device_info.get("max_input_channels", 0) or 0)


def device_default_sample_rate(device_info: Mapping[str, object], fallback: int = 48_000) -> int:
    sample_rate = device_info.get("default_samplerate", fallback)
    return int(round(float(sample_rate)))


def _device_name(device_info: Mapping[str, object]) -> str:
    return str(device_info.get("name", ""))


def _input_device_items(devices: Sequence[Mapping[str, object]]):
    for index, device in enumerate(devices):
        if device_input_channels(device) > 0:
            yield index, device


def _match_requested_device(devices: Sequence[Mapping[str, object]], requested_device):
    parsed = parse_device_id(requested_device)
    if parsed is None:
        return None
    if isinstance(parsed, int):
        if parsed < 0 or parsed >= len(devices):
            raise ValueError(f"input device index out of range: {parsed}")
        device = devices[parsed]
        if device_input_channels(device) <= 0:
            raise ValueError(f"device {parsed} has no input channels: {_device_name(device)}")
        return parsed, device

    requested_lower = parsed.lower()
    exact_matches = [
        (index, device)
        for index, device in _input_device_items(devices)
        if _device_name(device).lower() == requested_lower
    ]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [
        (index, device)
        for index, device in _input_device_items(devices)
        if requested_lower in _device_name(device).lower()
    ]
    if partial_matches:
        return partial_matches[0]
    raise ValueError(f"input device not found: {requested_device}")


def select_input_device(
    devices: Sequence[Mapping[str, object]],
    requested_device=None,
    *,
    keywords: Sequence[str] = DEFAULT_INPUT_KEYWORDS,
    preferred_min_channels: int = 8,
):
    """Return ``(index, device_info)`` for a requested or auto-detected input."""

    requested_match = _match_requested_device(devices, requested_device)
    if requested_match is not None:
        return requested_match

    candidates = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for index, device in _input_device_items(devices):
            if keyword_lower in _device_name(device).lower():
                candidates.append((index, device))

    for index, device in candidates:
        if device_input_channels(device) >= int(preferred_min_channels):
            return index, device
    if candidates:
        return candidates[0]
    raise ValueError("no BlackHole/Loopback/VB-Cable style input device found; pass --device")


def choose_recording_channels(device_info: Mapping[str, object], requested_channels=None) -> int:
    max_channels = device_input_channels(device_info)
    if max_channels <= 0:
        raise ValueError(f"selected device has no input channels: {_device_name(device_info)}")
    if requested_channels is None:
        return max_channels
    channels = int(requested_channels)
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels > max_channels:
        raise ValueError(f"requested {channels} channels, but device exposes {max_channels}")
    return channels


def sample_count_for_duration(seconds: float, sample_rate: int) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return max(1, int(round(float(seconds) * int(sample_rate))))


def record_input_audio(sd_module, *, device, seconds: float, sample_rate: int, channels: int) -> np.ndarray:
    frames = sample_count_for_duration(seconds, sample_rate)
    audio = sd_module.rec(
        frames,
        samplerate=int(sample_rate),
        channels=int(channels),
        dtype=np.float32,
        device=device,
    )
    sd_module.wait()
    return np.asarray(audio, dtype=np.float32)


def channel_peak_summary(audio: np.ndarray, *, max_channels: int = 8) -> str:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.size == 0:
        return "empty"
    channel_count = min(samples.shape[1], int(max_channels))
    peaks = np.max(np.abs(samples[:, :channel_count]), axis=0)
    return " ".join(f"ch{index}={peak:.3f}" for index, peak in enumerate(peaks))


def active_channel_indices(audio: np.ndarray, *, threshold: float = 1.0e-4, max_channels: int | None = 8) -> tuple[int, ...]:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.size == 0:
        return tuple()
    channel_count = samples.shape[1] if max_channels is None else min(samples.shape[1], int(max_channels))
    peaks = np.max(np.abs(samples[:, :channel_count]), axis=0)
    return tuple(index for index, peak in enumerate(peaks) if float(peak) >= float(threshold))


def active_channel_summary(audio: np.ndarray, *, threshold: float = 1.0e-4, max_channels: int | None = 8) -> str:
    active = active_channel_indices(audio, threshold=threshold, max_channels=max_channels)
    return "active channels: " + (",".join(f"ch{index}" for index in active) if active else "none")


def device_sanity_lines(device_info: Mapping[str, object], *, selected_channels: int | None = None) -> list[str]:
    max_channels = device_input_channels(device_info)
    selected = max_channels if selected_channels is None else int(selected_channels)
    sample_rate = device_default_sample_rate(device_info)
    lines = []
    if max_channels >= 8 and selected >= 8:
        lines.append(f"OK: input exposes {max_channels} channels and will capture {selected} channels")
    elif max_channels >= 8:
        lines.append(f"Warning: device exposes {max_channels} channels, but capture is limited to {selected}")
    else:
        lines.append(f"Warning: input exposes only {max_channels} channel(s); true 7.1 capture is not available")
    if sample_rate != 48_000:
        lines.append(f"Note: device default sample rate is {sample_rate} Hz, not 48000 Hz")
    return lines


def capture_sanity_lines(
    audio: np.ndarray,
    *,
    expected_channels: int,
    active_threshold: float = 1.0e-4,
) -> list[str]:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    captured_channels = 0 if samples.size == 0 else samples.shape[1]
    active = active_channel_indices(samples, threshold=active_threshold, max_channels=min(max(1, expected_channels), 8))
    lines = [active_channel_summary(samples, threshold=active_threshold, max_channels=min(max(1, expected_channels), 8))]
    if expected_channels >= 8 and captured_channels < 8:
        lines.append(f"Warning: expected {expected_channels} channels but captured only {captured_channels}")
    elif expected_channels >= 8 and len(active) <= 2:
        lines.append("Warning: multichannel capture has signal on only two or fewer 7.1 channels; source may be stereo/downmixed")
    elif expected_channels >= 8:
        lines.append("OK: captured signal spans more than two 7.1 channels")
    return lines


def print_device_list(sd_module) -> None:
    print(sd_module.query_devices())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture real multichannel input audio for SoundRadar tuning")
    parser.add_argument("--device", default=None, help='Input device name or index, e.g. "BlackHole 16ch"')
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, required=False, default=Path("/tmp/soundradar-capture.wav"))
    parser.add_argument("--channels", type=int, default=None, help="Input channels to record; defaults to the device max")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate; defaults to the device default")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import sounddevice as sd

    if args.list_devices:
        print_device_list(sd)
        return 0

    devices = list(sd.query_devices())
    device_index, device_info = select_input_device(devices, args.device)
    channels = choose_recording_channels(device_info, args.channels)
    sample_rate = int(args.sample_rate or device_default_sample_rate(device_info))

    print(
        f"Recording {args.seconds:.1f}s from {_device_name(device_info)} "
        f"(ID {device_index}, {channels}ch, {sample_rate} Hz)"
    )
    audio = record_input_audio(
        sd,
        device=device_index,
        seconds=args.seconds,
        sample_rate=sample_rate,
        channels=channels,
    )
    write_wav(args.out, audio, sample_rate)
    print(f"wrote {args.out}")
    print(f"peaks {channel_peak_summary(audio)}")
    for line in capture_sanity_lines(audio, expected_channels=channels):
        print(line)
    if channels < 8:
        print("warning: fewer than 8 channels were captured, so replay will use stereo/mono fallback")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
