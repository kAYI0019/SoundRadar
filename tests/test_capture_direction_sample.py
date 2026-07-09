import tempfile
import unittest
import wave

import numpy as np

from sound_model.capture_direction_sample import (
    active_channel_indices,
    capture_sanity_lines,
    channel_peak_summary,
    choose_recording_channels,
    device_default_sample_rate,
    device_sanity_lines,
    parse_device_id,
    record_input_audio,
    sample_count_for_duration,
    select_input_device,
)
from sound_model.audio_features import write_wav


class FakeSoundDevice:
    def __init__(self):
        self.rec_calls = []
        self.wait_called = False

    def rec(self, frames, *, samplerate, channels, dtype, device):
        self.rec_calls.append((frames, samplerate, channels, dtype, device))
        return np.full((frames, channels), 0.25, dtype=dtype)

    def wait(self):
        self.wait_called = True


class CaptureDirectionSampleTests(unittest.TestCase):
    def test_parse_device_id_keeps_names_and_parses_indices(self):
        self.assertEqual(parse_device_id("4"), 4)
        self.assertEqual(parse_device_id("BlackHole 16ch"), "BlackHole 16ch")
        self.assertIsNone(parse_device_id(None))

    def test_select_input_device_prefers_multichannel_blackhole(self):
        devices = [
            {"name": "Built-in Microphone", "max_input_channels": 1, "default_samplerate": 44100},
            {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "BlackHole 16ch", "max_input_channels": 16, "default_samplerate": 48000},
        ]

        index, device = select_input_device(devices)

        self.assertEqual(index, 2)
        self.assertEqual(device["name"], "BlackHole 16ch")

    def test_select_input_device_accepts_requested_name_or_index(self):
        devices = [
            {"name": "Built-in Output", "max_input_channels": 0},
            {"name": "Loopback Audio", "max_input_channels": 16},
        ]

        self.assertEqual(select_input_device(devices, "Loopback")[0], 1)
        self.assertEqual(select_input_device(devices, 1)[0], 1)
        with self.assertRaises(ValueError):
            select_input_device(devices, 0)

    def test_choose_recording_channels_defaults_to_device_max_and_rejects_too_many(self):
        device = {"name": "BlackHole 16ch", "max_input_channels": 16}

        self.assertEqual(choose_recording_channels(device, None), 16)
        self.assertEqual(choose_recording_channels(device, 8), 8)
        with self.assertRaises(ValueError):
            choose_recording_channels(device, 17)

    def test_device_default_sample_rate_rounds_to_int(self):
        self.assertEqual(device_default_sample_rate({"default_samplerate": 47999.6}), 48000)
        self.assertEqual(device_default_sample_rate({}), 48000)

    def test_sample_count_for_duration_rejects_non_positive_seconds(self):
        self.assertEqual(sample_count_for_duration(1.5, 1000), 1500)
        with self.assertRaises(ValueError):
            sample_count_for_duration(0.0, 1000)

    def test_record_input_audio_uses_requested_shape_and_waits(self):
        fake_sd = FakeSoundDevice()

        audio = record_input_audio(fake_sd, device=3, seconds=0.25, sample_rate=1000, channels=8)

        self.assertEqual(audio.shape, (250, 8))
        self.assertEqual(fake_sd.rec_calls, [(250, 1000, 8, np.float32, 3)])
        self.assertTrue(fake_sd.wait_called)

    def test_channel_peak_summary_reports_first_channels(self):
        audio = np.zeros((3, 10), dtype=np.float32)
        audio[0, 0] = -0.25
        audio[1, 7] = 0.75
        audio[2, 8] = 1.0

        summary = channel_peak_summary(audio)

        self.assertIn("ch0=0.250", summary)
        self.assertIn("ch7=0.750", summary)
        self.assertNotIn("ch8=", summary)

    def test_active_channel_indices_reports_channels_above_threshold(self):
        audio = np.zeros((4, 8), dtype=np.float32)
        audio[:, 0] = 0.01
        audio[:, 5] = 0.20

        self.assertEqual(active_channel_indices(audio, threshold=0.001), (0, 5))

    def test_capture_sanity_warns_when_multichannel_capture_looks_stereo(self):
        audio = np.zeros((4, 8), dtype=np.float32)
        audio[:, 0] = 0.1
        audio[:, 1] = 0.1

        lines = capture_sanity_lines(audio, expected_channels=8)

        self.assertIn("active channels: ch0,ch1", lines[0])
        self.assertTrue(any("source may be stereo/downmixed" in line for line in lines))

    def test_device_sanity_lines_warn_for_non_surround_input(self):
        lines = device_sanity_lines({"name": "Built-in Mic", "max_input_channels": 1, "default_samplerate": 44100})

        self.assertTrue(any("true 7.1 capture is not available" in line for line in lines))
        self.assertTrue(any("44100" in line for line in lines))

    def test_capture_output_wav_can_preserve_16_channels(self):
        audio = np.zeros((12, 16), dtype=np.float32)
        audio[:, 6] = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/capture.wav"
            write_wav(path, audio, 48000)

            with wave.open(path, "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 16)
                self.assertEqual(wav_file.getframerate(), 48000)
                self.assertEqual(wav_file.getnframes(), 12)


if __name__ == "__main__":
    unittest.main()
