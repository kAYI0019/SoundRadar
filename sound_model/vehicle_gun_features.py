"""Lightweight acoustic evidence for vehicle-versus-gunshot resolution.

All features are computed with NumPy at the input sample rate.  Frequency
bands are clipped at Nyquist, and invalid, silent, clipped, or very short
waveforms return finite values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_EPS = 1.0e-12


@dataclass(frozen=True)
class BandEnergyFeatures:
    """Relative FFT energy in 20-250 Hz, 250-2000 Hz, and >=2000 Hz."""

    low_ratio: float = 0.0
    mid_ratio: float = 0.0
    high_ratio: float = 0.0


@dataclass(frozen=True)
class EnvelopeFeatures:
    """Timing and concentration measurements from a short-time RMS envelope.

    ``attack_time_ms`` is the time from the first 10%-of-peak crossing through
    the first 90% crossing. ``peak_hold_time_ms`` is the total time at or above
    90% of peak. ``decay_time_ms`` runs from the peak to the first subsequent
    10% crossing (or the window end). ``onset_duration_ms`` spans the first 10%
    crossing through that post-peak 10% crossing. ``energy_concentration`` is
    the fraction of total waveform energy in a 50 ms region centered on the
    highest-energy envelope frame. Times use the input sample rate.
    """

    attack_time_ms: float = 0.0
    peak_hold_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    onset_duration_ms: float = 0.0
    energy_concentration: float = 0.0


@dataclass(frozen=True)
class VehicleGunAcousticFeatures:
    """Normalized spectral evidence plus envelope measurements in milliseconds."""

    spectral_flux: float = 0.0
    low_frequency_ratio: float = 0.0
    mid_frequency_ratio: float = 0.0
    high_frequency_ratio: float = 0.0
    attack_time_ms: float = 0.0
    peak_hold_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    onset_duration_ms: float = 0.0
    energy_concentration: float = 0.0


def _safe_waveform(waveform: np.ndarray) -> np.ndarray:
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    return np.clip(np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def _frame_samples(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    frame_length = max(2, int(frame_length))
    hop_length = max(1, int(hop_length))
    if samples.size < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.size))
    frame_count = 1 + int(math.ceil(max(0, samples.size - frame_length) / hop_length))
    padded_length = frame_length + (frame_count - 1) * hop_length
    if samples.size < padded_length:
        samples = np.pad(samples, (0, padded_length - samples.size))
    starts = np.arange(frame_count) * hop_length
    return np.stack([samples[start : start + frame_length] for start in starts])


def normalized_spectral_flux(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 32.0,
    hop_ms: float = 10.0,
) -> float:
    """Return peak positive spectral flux in [0, 1].

    Each magnitude spectrum is L2-normalized before positive frame-to-frame
    differences are measured.  Peak rather than mean flux preserves short
    impulses inside a one-second analysis window.  A single frame has no
    temporal change and therefore returns zero.
    """

    samples = _safe_waveform(waveform)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.size < 2 or float(np.max(np.abs(samples), initial=0.0)) <= 1.0e-7:
        return 0.0
    frame_length = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frames = _frame_samples(samples, frame_length, hop_length)
    if frames.shape[0] < 2:
        return 0.0
    window = np.hanning(frame_length).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    norms = np.linalg.norm(magnitudes, axis=1, keepdims=True)
    normalized = np.divide(magnitudes, norms, out=np.zeros_like(magnitudes), where=norms > _EPS)
    positive_changes = np.maximum(0.0, np.diff(normalized, axis=0))
    frame_flux = np.linalg.norm(positive_changes, axis=1)
    return float(np.clip(np.max(frame_flux, initial=0.0), 0.0, 1.0))


def band_energy_features(waveform: np.ndarray, sample_rate: int) -> BandEnergyFeatures:
    """Return normalized 20 Hz-to-Nyquist band energies with a finite zero fallback."""

    samples = _safe_waveform(waveform)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.size < 2 or float(np.max(np.abs(samples), initial=0.0)) <= 1.0e-7:
        return BandEnergyFeatures()
    window = np.hanning(samples.size).astype(np.float32)
    power = np.abs(np.fft.rfft(samples * window)) ** 2
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    low = float(np.sum(power[(frequencies >= 20.0) & (frequencies < 250.0)]))
    mid = float(np.sum(power[(frequencies >= 250.0) & (frequencies < 2000.0)]))
    high = float(np.sum(power[frequencies >= 2000.0]))
    total = low + mid + high
    if not math.isfinite(total) or total <= _EPS:
        return BandEnergyFeatures()
    return BandEnergyFeatures(low / total, mid / total, high / total)


def envelope_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 5.0,
) -> EnvelopeFeatures:
    """Return attack, peak-hold, decay, onset duration, and energy concentration."""

    samples = _safe_waveform(waveform)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.size < 2:
        return EnvelopeFeatures()
    total_energy = float(np.sum(samples.astype(np.float64) ** 2))
    if not math.isfinite(total_energy) or total_energy <= _EPS:
        return EnvelopeFeatures()
    frame_length = max(2, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frames = _frame_samples(samples, frame_length, hop_length)
    envelope = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak_index = int(np.argmax(envelope))
    peak = float(envelope[peak_index])
    if peak <= _EPS:
        return EnvelopeFeatures()
    normalized = envelope / peak
    before_peak = normalized[: peak_index + 1]
    ten_indices = np.flatnonzero(before_peak >= 0.10)
    ninety_indices = np.flatnonzero(before_peak >= 0.90)
    attack_start = int(ten_indices[0]) if ten_indices.size else peak_index
    attack_end = int(ninety_indices[0]) if ninety_indices.size else peak_index
    after_peak = normalized[peak_index:]
    decay_indices = np.flatnonzero(after_peak <= 0.10)
    decay_end = peak_index + (int(decay_indices[0]) if decay_indices.size else len(after_peak) - 1)
    hop_ms_actual = hop_length * 1000.0 / sample_rate
    concentration_length = max(1, int(round(sample_rate * 0.050)))
    peak_center = peak_index * hop_length + frame_length // 2
    concentration_start = max(
        0,
        min(max(0, samples.size - concentration_length), peak_center - concentration_length // 2),
    )
    concentration_end = min(samples.size, concentration_start + concentration_length)
    concentration = float(
        np.sum(samples[concentration_start:concentration_end].astype(np.float64) ** 2) / total_energy
    )
    return EnvelopeFeatures(
        attack_time_ms=max(0.0, (attack_end - attack_start) * hop_ms_actual),
        peak_hold_time_ms=float(np.count_nonzero(normalized >= 0.90) * hop_ms_actual),
        decay_time_ms=max(0.0, (decay_end - peak_index) * hop_ms_actual),
        onset_duration_ms=max(0.0, (decay_end - attack_start) * hop_ms_actual),
        energy_concentration=float(np.clip(concentration, 0.0, 1.0)),
    )


def extract_vehicle_gun_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> VehicleGunAcousticFeatures:
    """Compute the complete finite feature set used by the rule resolver."""

    bands = band_energy_features(waveform, sample_rate)
    envelope = envelope_features(waveform, sample_rate)
    return VehicleGunAcousticFeatures(
        spectral_flux=normalized_spectral_flux(waveform, sample_rate),
        low_frequency_ratio=bands.low_ratio,
        mid_frequency_ratio=bands.mid_ratio,
        high_frequency_ratio=bands.high_ratio,
        attack_time_ms=envelope.attack_time_ms,
        peak_hold_time_ms=envelope.peak_hold_time_ms,
        decay_time_ms=envelope.decay_time_ms,
        onset_duration_ms=envelope.onset_duration_ms,
        energy_concentration=envelope.energy_concentration,
    )
