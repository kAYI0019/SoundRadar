from datetime import datetime
from types import SimpleNamespace
import unittest

from sound_model.capture_direction_sample_gui import (
    default_capture_path,
    direction_events_command,
    direction_score_summary_lines,
    format_device_label,
    prediction_summary_text,
)


class CaptureDirectionSampleGuiTests(unittest.TestCase):
    def test_default_capture_path_uses_timestamped_tmp_wav(self):
        path = default_capture_path(datetime(2026, 7, 9, 13, 4, 5), directory="/tmp")

        self.assertEqual(str(path), "/tmp/soundradar-capture-20260709-130405.wav")

    def test_format_device_label_shows_index_channels_and_rate(self):
        label = format_device_label(
            3,
            {
                "name": "BlackHole 16ch",
                "max_input_channels": 16,
                "default_samplerate": 48000.0,
            },
        )

        self.assertEqual(label, "3: BlackHole 16ch (16ch, 48000 Hz)")

    def test_direction_events_command_quotes_paths(self):
        command = direction_events_command("/tmp/sample with space.wav", teacher_model="ast", device="auto", top_k=5)

        self.assertIn("sound_model.direction_events", command)
        self.assertIn("'/tmp/sample with space.wav'", command)
        self.assertIn("--teacher-model ast", command)
        self.assertIn("--device auto", command)
        self.assertIn("--top-k 5", command)

    def test_direction_score_summary_lines_show_scores_and_top_labels(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "vehicle": 0.2, "footstep": 0.0, "explosion": 0.1},
            },
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={"right": [{"label": "Gunshot", "score": 0.83}]},
        )

        lines = direction_score_summary_lines(prediction)

        right_line = [line for line in lines if line.startswith(" R ")][0]
        self.assertIn("active gunshot", right_line)
        self.assertIn("GUN .80", right_line)
        self.assertIn("VEH .20", right_line)
        self.assertIn("Gunshot .83", right_line)

    def test_prediction_summary_text_includes_hud_and_direction_scores(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={"right": [{"label": "Gunshot", "score": 0.83}]},
        )

        summary = prediction_summary_text(prediction)

        self.assertIn("HUD summary:", summary)
        self.assertIn("gun cand R .80", summary)
        self.assertIn("direction scores:", summary)
        self.assertIn("R active gunshot", summary)


if __name__ == "__main__":
    unittest.main()
