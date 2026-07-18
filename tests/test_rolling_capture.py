import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sound_model import rolling_capture


class RollingCaptureHelperTests(unittest.TestCase):
    def test_buffer_keeps_only_the_recent_window_without_qt(self):
        capture = rolling_capture.RollingAudioCapture(sample_rate=10, channel_count=2, seconds=0.3)
        capture.append_blocks([np.ones((4, 2), dtype=np.float32)], capture_time=10.0)
        capture.append_blocks([np.full((2, 2), 2.0, dtype=np.float32)], capture_time=10.2)

        snapshot = capture.snapshot()

        self.assertEqual(snapshot.audio.shape, (3, 2))
        np.testing.assert_array_equal(snapshot.audio[-1], [2.0, 2.0])
        self.assertAlmostEqual(snapshot.start_capture_time, 9.9)
        self.assertAlmostEqual(snapshot.end_capture_time, 10.2)

    def test_snapshot_writer_records_metadata_and_supplied_hud_lines(self):
        class Prediction:
            def to_jsonable(self):
                return {"mode": "test"}

        capture = rolling_capture.RollingAudioCapture(sample_rate=10, channel_count=8, seconds=1.0)
        audio = np.zeros((5, 8), dtype=np.float32)
        audio[:, :2] = 0.5
        capture.append_blocks([audio], capture_time=10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, metadata_path = rolling_capture.write_rolling_capture_snapshot(
                capture.snapshot(),
                directory=tmpdir,
                now=10.25,
                prediction=Prediction(),
                hud_summary_lines=["hud test"],
                threshold_profile_name="default",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertTrue(str(audio_path).endswith(".wav"))
        self.assertEqual(metadata["sample_rate"], 10)
        self.assertEqual(metadata["prediction"], {"mode": "test"})
        self.assertEqual(metadata["hud_summary_lines"], ["hud test"])
        self.assertIn("ch0=0.500", metadata["peak_summary"])

    def test_trigger_file_is_written_and_consumed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trigger"

            written = rolling_capture.write_rolling_capture_trigger(path, now=12.5)
            self.assertEqual(written.read_text(encoding="utf-8"), "12.5")
            consumed = rolling_capture.consume_rolling_capture_trigger(path)

            self.assertTrue(consumed)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
