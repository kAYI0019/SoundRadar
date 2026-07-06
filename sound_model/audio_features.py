"""Audio loading and log-mel feature extraction for the V0 sound model.

This deliberately avoids heavyweight audio dependencies.  The implementation is
small enough for tests and smoke training, while keeping the same feature shape
that a later PyTorch CNN/CRNN implementation can consume: stereo-derived
``[left, right, mid, side]`` log-mel channels over a one-second window.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import wave

import numpy as np

DEFAULT_CLASSES: tuple[str, ...] = (
    "background",
    "footstep",
    "gunshot",
    "vehicle",
    "explosion",
)

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_N_FFT = 1_024
DEFAULT_STFT_HOP = 320  # 20 ms at 16 kHz; runtime inference still slides every 100 ms.
DEFAULT_MEL_BINS = 64
_EPS = 1e-8


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as float32 ``(samples, channels)`` in [-1, 1]."""

    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"Unsupported compressed WAV: {path}")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        data = _decode_pcm24(raw).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")

    if channels <= 0:
        raise ValueError(f"Invalid channel count {channels}: {path}")
    if data.size % channels != 0:
        raise ValueError(f"Malformed WAV data length for {channels} channels: {path}")
    return data.reshape(-1, channels).astype(np.float32, copy=False), sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio ``(samples, channels)`` to a 16-bit PCM WAV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(audio.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _decode_pcm24(raw: bytes) -> np.ndarray:
    triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    values = (
        triples[:, 0].astype(np.int32)
        | (triples[:, 1].astype(np.int32) << 8)
        | (triples[:, 2].astype(np.int32) << 16)
    )
    sign_bit = 1 << 23
    values = (values ^ sign_bit) - sign_bit
    return values


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return exactly two channels, duplicating mono and dropping extra channels."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def resample_linear(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample with deterministic linear interpolation."""

    if original_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if len(audio) == 0:
        return audio.copy()

    new_length = max(1, int(round(len(audio) * float(target_rate) / float(original_rate))))
    old_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=True)
    channels = [np.interp(new_positions, old_positions, audio[:, ch]) for ch in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def fix_length(audio: np.ndarray, sample_rate: int, window_sec: float = DEFAULT_WINDOW_SEC) -> np.ndarray:
    """Pad or crop audio to the exact model window length."""

    target_samples = int(round(sample_rate * window_sec))
    if target_samples <= 0:
        raise ValueError("window_sec must produce at least one sample")
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == target_samples:
        return audio
    if len(audio) > target_samples:
        return audio[:target_samples]
    padding = np.zeros((target_samples - len(audio), audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio, padding], axis=0)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    mel_bins: int = DEFAULT_MEL_BINS,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create a triangular mel filterbank with shape ``(mel_bins, fft_bins)``."""

    if f_max is None:
        f_max = sample_rate / 2.0
    if not (0 <= f_min < f_max <= sample_rate / 2.0):
        raise ValueError("invalid mel frequency range")

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), mel_bins + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    filters = np.zeros((mel_bins, len(fft_freqs)), dtype=np.float32)
    for idx in range(mel_bins):
        left, center, right = hz_points[idx : idx + 3]
        if center <= left or right <= center:
            continue
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        filters[idx] = np.maximum(0.0, np.minimum(up, down))
        area = filters[idx].sum()
        if area > 0:
            filters[idx] /= area
    return filters


def _power_spectrogram(channel: np.ndarray, n_fft: int, hop_samples: int) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    if len(channel) < n_fft:
        channel = np.pad(channel, (0, n_fft - len(channel)))
    frame_count = 1 + max(0, (len(channel) - n_fft) // hop_samples)
    starts = np.arange(frame_count) * hop_samples
    frames = np.stack([channel[start : start + n_fft] for start in starts], axis=0)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return power / (np.sum(window ** 2) + _EPS)


def log_mel_feature_channels(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_rate: int = DEFAULT_SAMPLE_RATE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    n_fft: int = DEFAULT_N_FFT,
    stft_hop: int = DEFAULT_STFT_HOP,
    mel_bins: int = DEFAULT_MEL_BINS,
) -> np.ndarray:
    """Return ``(4, frames, mel_bins)`` log-mel features.

    The channel order follows the plan: left, right, mid=(L+R)/2, side=(L-R)/2.
    """

    stereo = ensure_stereo(audio)
    stereo = resample_linear(stereo, sample_rate, target_rate)
    stereo = fix_length(stereo, target_rate, window_sec)

    left = stereo[:, 0]
    right = stereo[:, 1]
    sources = [left, right, (left + right) * 0.5, (left - right) * 0.5]
    filters = mel_filterbank(target_rate, n_fft, mel_bins)

    channels: list[np.ndarray] = []
    for source in sources:
        power = _power_spectrogram(source, n_fft, stft_hop)
        mel_power = np.maximum(power @ filters.T, _EPS)
        channels.append(np.log(mel_power).astype(np.float32))
    return np.stack(channels, axis=0)


def extract_feature_vector(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_rate: int = DEFAULT_SAMPLE_RATE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    n_fft: int = DEFAULT_N_FFT,
    stft_hop: int = DEFAULT_STFT_HOP,
    mel_bins: int = DEFAULT_MEL_BINS,
) -> np.ndarray:
    """Aggregate log-mel channels into a compact fixed-length feature vector."""

    features = log_mel_feature_channels(
        audio,
        sample_rate,
        target_rate=target_rate,
        window_sec=window_sec,
        n_fft=n_fft,
        stft_hop=stft_hop,
        mel_bins=mel_bins,
    )
    means = features.mean(axis=1)
    stds = features.std(axis=1)
    peaks = features.max(axis=1)
    return np.concatenate([means, stds, peaks], axis=0).reshape(-1).astype(np.float32)
