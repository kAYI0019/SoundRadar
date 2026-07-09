import tempfile
import unittest
import wave

import numpy as np

from sound_model.surround_probe import (
    AUDIO_MIDI_7_1_CHANNELS,
    PROBE_DIRECTIONS,
    render_probe_program,
    render_probe_sequence,
    write_probe_wav,
)


class SurroundProbeTests(unittest.TestCase):
    def test_render_probe_sequence_uses_audio_midi_7_1_channels(self):
        audio, events = render_probe_sequence(
            "gunshot",
            sample_rate=1000,
            channels=16,
            event_seconds=0.08,
            gap_seconds=0.03,
            amplitude=0.7,
        )

        self.assertEqual(audio.shape[1], 16)
        self.assertEqual(tuple(event.direction for event in events), PROBE_DIRECTIONS)
        self.assertTrue(np.allclose(audio[:, 3], 0.0))
        self.assertTrue(np.allclose(audio[:, 8:], 0.0))
        for event in events:
            self.assertEqual(event.channel, AUDIO_MIDI_7_1_CHANNELS[event.direction])
            start = int(round(event.start_seconds * 1000))
            end = start + int(round(event.duration_seconds * 1000))
            peaks = np.max(np.abs(audio[start:end, :8]), axis=0)
            active_channels = {index for index, peak in enumerate(peaks) if peak > 0.01}

            self.assertEqual(active_channels, {event.channel})

    def test_render_probe_program_can_include_gunshot_and_vehicle_sections(self):
        audio, events = render_probe_program(
            ("gunshot", "vehicle"),
            sample_rate=1000,
            channels=8,
            gap_seconds=0.02,
            section_gap_seconds=0.05,
        )

        self.assertEqual(audio.shape[1], 8)
        self.assertEqual(len(events), len(PROBE_DIRECTIONS) * 2)
        self.assertEqual(events[0].kind, "gunshot")
        self.assertEqual(events[-1].kind, "vehicle")
        self.assertGreater(events[-1].start_seconds, events[len(PROBE_DIRECTIONS) - 1].start_seconds)

    def test_write_probe_wav_preserves_multichannel_shape(self):
        audio, _ = render_probe_sequence(
            "vehicle",
            sample_rate=1000,
            channels=8,
            event_seconds=0.05,
            gap_seconds=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/probe.wav"
            write_probe_wav(path, audio, 1000)

            with wave.open(path, "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 8)
                self.assertEqual(wav_file.getframerate(), 1000)
                self.assertEqual(wav_file.getnframes(), audio.shape[0])


if __name__ == "__main__":
    unittest.main()
