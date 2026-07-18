import math
import unittest

import numpy as np

from sound_model.direction_events import transient_gunshot_score
from sound_model.vehicle_gun_features import (
    band_energy_features,
    extract_vehicle_gun_acoustic_features,
)


class VehicleGunFeatureTests(unittest.TestCase):
    def test_short_impulse_has_high_transient_flux_and_concentration(self):
        sample_rate = 16_000
        waveform = np.zeros(sample_rate, dtype=np.float32)
        burst = np.hanning(48).astype(np.float32)
        waveform[8_000 : 8_000 + len(burst)] = burst

        features = extract_vehicle_gun_acoustic_features(waveform, sample_rate)

        self.assertGreater(transient_gunshot_score(waveform, sample_rate), 0.65)
        self.assertGreater(features.spectral_flux, 0.70)
        self.assertGreater(features.energy_concentration, 0.70)
        self.assertLess(features.onset_duration_ms, 80.0)

    def test_sustained_low_frequency_tone_has_high_low_band_ratio(self):
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.5 * np.sin(2.0 * np.pi * 120.0 * time)

        bands = band_energy_features(waveform, sample_rate)

        self.assertGreater(bands.low_ratio, 0.95)
        self.assertAlmostEqual(bands.low_ratio + bands.mid_ratio + bands.high_ratio, 1.0, places=6)

    def test_silence_short_and_clipped_inputs_are_finite(self):
        inputs = (
            np.zeros(16_000, dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([np.nan, np.inf, -np.inf, 2.0, -2.0], dtype=np.float32),
        )

        for waveform in inputs:
            features = extract_vehicle_gun_acoustic_features(waveform, 16_000)
            for value in features.__dict__.values():
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_frequency_bands_follow_input_nyquist(self):
        sample_rate = 3_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = np.sin(2.0 * np.pi * 900.0 * time)

        bands = band_energy_features(waveform, sample_rate)

        self.assertGreater(bands.mid_ratio, 0.99)
        self.assertEqual(bands.high_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
