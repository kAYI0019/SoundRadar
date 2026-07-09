from datetime import datetime
import csv
import json
import tempfile
from types import SimpleNamespace
import unittest

from sound_model.capture_direction_sample_gui import (
    analysis_result_path,
    append_sample_library_record,
    available_threshold_profiles,
    default_capture_path,
    default_library_path,
    direction_events_command,
    direction_score_summary_lines,
    format_device_label,
    prediction_result_payload,
    prediction_summary_text,
    sample_library_record,
    write_prediction_result_json,
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

    def test_available_threshold_profiles_exposes_soundradar_profiles(self):
        self.assertIn("quiet", available_threshold_profiles())

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

    def test_analysis_result_path_uses_sidecar_json_name(self):
        path = analysis_result_path("/tmp/sample.wav")

        self.assertEqual(str(path), "/tmp/sample.analysis.json")

    def test_prediction_result_payload_contains_analysis_metadata(self):
        prediction = SimpleNamespace(
            source_path="/tmp/sample.wav",
            sample_rate=48000,
            mode="test",
            direction_event_scores={
                "right": {"gunshot": 0.8, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={"right": [{"label": "Gunshot", "score": 0.83}]},
            to_jsonable=lambda: {"source_path": "/tmp/sample.wav", "sample_rate": 48000},
        )

        payload = prediction_result_payload(
            prediction,
            audio_path="/tmp/sample.wav",
            teacher_model="ast",
            device="auto",
            top_k=5,
            peak_summary="ch0=0.100",
            analyzed_at=datetime(2026, 7, 9, 13, 5, 0),
        )

        self.assertEqual(payload["audio_path"], "/tmp/sample.wav")
        self.assertEqual(payload["peak_summary"], "ch0=0.100")
        self.assertEqual(payload["teacher_model"], "ast")
        self.assertEqual(payload["threshold_profile"], "default")
        self.assertIn("gun cand R .80", "\n".join(payload["hud_summary_lines"]))
        self.assertEqual(payload["direction_scores"]["right"]["scores"]["gunshot"], 0.8)
        self.assertEqual(payload["gunshot_display"]["shown_directions"], ["right"])

    def test_write_prediction_result_json_writes_sidecar(self):
        payload = {"schema_version": 1, "audio_path": "/tmp/sample.wav"}

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = f"{tmpdir}/sample.wav"
            result_path = write_prediction_result_json(audio_path, payload)

            self.assertEqual(result_path.name, "sample.analysis.json")
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), payload)

    def test_sample_library_record_and_append_csv(self):
        record = sample_library_record(
            audio_path="/tmp/sample.wav",
            tag="gunshot",
            notes="near right",
            analysis_path="/tmp/sample.analysis.json",
            peak_summary="ch0=0.100",
            created_at=datetime(2026, 7, 9, 13, 6, 0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = append_sample_library_record(default_library_path(tmpdir), record)

            with library_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["tag"], "gunshot")
        self.assertEqual(rows[0]["notes"], "near right")
        self.assertEqual(rows[0]["analysis_path"], "/tmp/sample.analysis.json")

    def test_sample_library_record_rejects_unknown_tags(self):
        with self.assertRaises(ValueError):
            sample_library_record(audio_path="/tmp/sample.wav", tag="music")


if __name__ == "__main__":
    unittest.main()
