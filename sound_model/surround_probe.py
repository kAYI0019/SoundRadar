"""Generate or play channel-discrete 7.1 routing probes for SoundRadar.

The probe follows Audio MIDI / SoundRadar 7.1 order:
FL=0, FR=1, C=2, LFE=3, side-left=4, side-right=5, rear-left=6,
rear-right=7.  LFE is intentionally silent because SoundRadar ignores it.
This checks channel routing, not classifier quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


AUDIO_MIDI_7_1_CHANNELS = {
    "front_left": 0,
    "front": 2,
    "front_right": 1,
    "left": 4,
    "right": 5,
    "rear_left": 6,
    "rear_right": 7,
}
PROBE_DIRECTIONS = tuple(AUDIO_MIDI_7_1_CHANNELS)
PROBE_KINDS = ("gunshot", "vehicle")


@dataclass(frozen=True)
class ProbeEvent:
    kind: str
    direction: str
    channel: int
    start_seconds: float
    duration_seconds: float


def _sample_count(sample_rate: int, seconds: float) -> int:
    return max(0, int(round(float(sample_rate) * float(seconds))))


def _normalize_probe_waveform(samples: np.ndarray, amplitude: float) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1.0e-9:
        return samples.astype(np.float32, copy=False)
    return (samples / peak * float(amplitude)).astype(np.float32, copy=False)


def gunshot_probe_waveform(sample_rate: int, duration_seconds: float = 0.28, amplitude: float = 0.82) -> np.ndarray:
    """Return a short deterministic impulse-like waveform."""

    count = max(1, _sample_count(sample_rate, duration_seconds))
    t = np.arange(count, dtype=np.float32) / float(sample_rate)
    rng = np.random.default_rng(8811)
    click = rng.uniform(-1.0, 1.0, count).astype(np.float32) * np.exp(-42.0 * t)
    body = np.sin(2.0 * np.pi * 145.0 * t).astype(np.float32) * np.exp(-18.0 * t)
    snap = np.sin(2.0 * np.pi * 2100.0 * t).astype(np.float32) * np.exp(-90.0 * t)
    return _normalize_probe_waveform(0.64 * click + 0.28 * body + 0.08 * snap, amplitude)


def vehicle_probe_waveform(sample_rate: int, duration_seconds: float = 1.20, amplitude: float = 0.46) -> np.ndarray:
    """Return a deterministic low rumble waveform useful for routing checks."""

    count = max(1, _sample_count(sample_rate, duration_seconds))
    t = np.arange(count, dtype=np.float32) / float(sample_rate)
    rng = np.random.default_rng(5271)
    rumble = (
        np.sin(2.0 * np.pi * 48.0 * t)
        + 0.55 * np.sin(2.0 * np.pi * 73.0 * t)
        + 0.22 * np.sin(2.0 * np.pi * 118.0 * t)
    ).astype(np.float32)
    modulation = (0.70 + 0.30 * np.sin(2.0 * np.pi * 2.6 * t)).astype(np.float32)
    noise = rng.normal(0.0, 0.08, count).astype(np.float32)
    attack = np.minimum(1.0, t / 0.18)
    release = np.minimum(1.0, np.maximum(0.0, (duration_seconds - t) / 0.22))
    envelope = (attack * release).astype(np.float32)
    return _normalize_probe_waveform((0.82 * rumble * modulation + noise) * envelope, amplitude)


def probe_waveform(kind: str, sample_rate: int, duration_seconds: float, amplitude: float) -> np.ndarray:
    if kind == "gunshot":
        return gunshot_probe_waveform(sample_rate, duration_seconds, amplitude)
    if kind == "vehicle":
        return vehicle_probe_waveform(sample_rate, duration_seconds, amplitude)
    raise ValueError(f"unknown probe kind: {kind}")


def default_event_seconds(kind: str) -> float:
    return 0.34 if kind == "gunshot" else 1.20


def render_probe_sequence(
    kind: str,
    *,
    sample_rate: int = 48_000,
    channels: int = 16,
    event_seconds: float | None = None,
    gap_seconds: float = 0.35,
    amplitude: float = 0.78,
    directions: tuple[str, ...] = PROBE_DIRECTIONS,
) -> tuple[np.ndarray, tuple[ProbeEvent, ...]]:
    if channels < 8:
        raise ValueError("7.1 probe output needs at least 8 channels")
    event_seconds = default_event_seconds(kind) if event_seconds is None else float(event_seconds)
    gap_seconds = max(0.0, float(gap_seconds))
    event_count = len(directions)
    total_seconds = event_count * event_seconds + max(0, event_count - 1) * gap_seconds
    audio = np.zeros((max(1, _sample_count(sample_rate, total_seconds)), int(channels)), dtype=np.float32)
    events = []
    cursor = 0
    gap_count = _sample_count(sample_rate, gap_seconds) if gap_seconds else 0

    for direction in directions:
        channel = AUDIO_MIDI_7_1_CHANNELS[direction]
        waveform = probe_waveform(kind, sample_rate, event_seconds, amplitude)
        end = min(audio.shape[0], cursor + waveform.shape[0])
        audio[cursor:end, channel] += waveform[: end - cursor]
        events.append(
            ProbeEvent(
                kind=kind,
                direction=direction,
                channel=channel,
                start_seconds=cursor / float(sample_rate),
                duration_seconds=waveform.shape[0] / float(sample_rate),
            )
        )
        cursor = end + gap_count
    return np.clip(audio, -1.0, 1.0), tuple(events)


def render_probe_program(
    kinds: tuple[str, ...],
    *,
    sample_rate: int = 48_000,
    channels: int = 16,
    gap_seconds: float = 0.35,
    section_gap_seconds: float = 1.0,
    amplitude: float = 0.78,
) -> tuple[np.ndarray, tuple[ProbeEvent, ...]]:
    chunks = []
    events = []
    offset_seconds = 0.0
    section_gap = np.zeros((_sample_count(sample_rate, section_gap_seconds), int(channels)), dtype=np.float32)

    for index, kind in enumerate(kinds):
        chunk, chunk_events = render_probe_sequence(
            kind,
            sample_rate=sample_rate,
            channels=channels,
            gap_seconds=gap_seconds,
            amplitude=amplitude,
        )
        chunks.append(chunk)
        for event in chunk_events:
            events.append(
                ProbeEvent(
                    kind=event.kind,
                    direction=event.direction,
                    channel=event.channel,
                    start_seconds=event.start_seconds + offset_seconds,
                    duration_seconds=event.duration_seconds,
                )
            )
        offset_seconds += chunk.shape[0] / float(sample_rate)
        if index < len(kinds) - 1:
            chunks.append(section_gap)
            offset_seconds += section_gap.shape[0] / float(sample_rate)

    if not chunks:
        return np.zeros((0, int(channels)), dtype=np.float32), tuple()
    return np.concatenate(chunks, axis=0), tuple(events)


def write_probe_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(audio.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def _parse_device(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _print_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def play_probe(audio: np.ndarray, sample_rate: int, *, device=None) -> None:
    import sounddevice as sd

    sd.play(audio, samplerate=int(sample_rate), device=device)
    sd.wait()


def _print_schedule(events: tuple[ProbeEvent, ...]) -> None:
    for event in events:
        print(
            f"{event.start_seconds:6.2f}s  {event.kind:7s}  "
            f"{event.direction:11s}  ch{event.channel}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play or write a channel-discrete 7.1 SoundRadar routing probe")
    parser.add_argument("--kind", choices=("gunshot", "vehicle", "both"), default="both")
    parser.add_argument("--device", default=None, help='sounddevice output device name or index, e.g. "BlackHole 16ch"')
    parser.add_argument("--channels", type=int, default=16, help="Output channel count; use 16 for BlackHole 16ch")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--gap", type=float, default=0.35, help="Seconds between directional events")
    parser.add_argument("--section-gap", type=float, default=1.0, help="Seconds between gunshot and vehicle sections")
    parser.add_argument("--amplitude", type=float, default=0.78)
    parser.add_argument("--write-wav", type=Path, default=None)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        _print_devices()
        return 0

    kinds = PROBE_KINDS if args.kind == "both" else (args.kind,)
    audio, events = render_probe_program(
        kinds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        gap_seconds=args.gap,
        section_gap_seconds=args.section_gap,
        amplitude=args.amplitude,
    )
    _print_schedule(events)

    if args.write_wav is not None:
        write_probe_wav(args.write_wav, audio, args.sample_rate)
        print(f"wrote {args.write_wav}")

    if not args.no_play:
        play_probe(audio, args.sample_rate, device=_parse_device(args.device))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
