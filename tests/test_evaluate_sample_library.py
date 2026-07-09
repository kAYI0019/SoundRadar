import csv
import tempfile
from types import SimpleNamespace
import unittest

import soundRadar
from sound_model.evaluate_sample_library import (
    default_evaluation_path,
    default_summary_path,
    evaluation_rows_for_prediction,
    resolve_audio_path,
    summarize_evaluation_rows,
    write_evaluation_csv,
    write_summary_csv,
)


class EvaluateSampleLibraryTests(unittest.TestCase):
    def tearDown(self):
        soundRadar.apply_threshold_profile("default")

    def test_default_evaluation_path_uses_library_stem(self):
        self.assertEqual(
            str(default_evaluation_path("/tmp/sample_library.csv")),
            "/tmp/sample_library.evaluation.csv",
        )

    def test_default_summary_path_uses_evaluation_stem(self):
        self.assertEqual(
            str(default_summary_path("/tmp/sample_library.evaluation.csv")),
            "/tmp/sample_library.evaluation.summary.csv",
        )

    def test_resolve_audio_path_uses_library_directory_for_relative_paths(self):
        self.assertEqual(
            str(resolve_audio_path("/tmp/library/sample_library.csv", "clips/a.wav")),
            "/tmp/library/clips/a.wav",
        )

    def test_evaluation_rows_for_prediction_compare_profiles(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.80, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0},
            },
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={},
        )

        rows = evaluation_rows_for_prediction(
            {"audio_path": "/tmp/sample.wav", "tag": "gunshot", "notes": "right"},
            prediction,
            profiles=("default", "quiet"),
            teacher_model="ast",
            device="auto",
            top_k=5,
        )

        self.assertEqual([row["profile"] for row in rows], ["default", "quiet"])
        self.assertEqual(rows[0]["expected_detected"], "yes")
        self.assertIn("gunshot:right:0.800", rows[0]["shown_events"])
        self.assertEqual(rows[0]["gunshot_shown"], "right")
        self.assertEqual(rows[0]["max_gunshot"], "0.800")

    def test_write_evaluation_csv_writes_header_and_rows(self):
        rows = [
            {
                "audio_path": "/tmp/sample.wav",
                "tag": "gunshot",
                "profile": "default",
                "shown_event_count": "1",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_evaluation_csv(f"{tmpdir}/out.csv", rows)
            with path.open(newline="", encoding="utf-8") as csv_file:
                written = list(csv.DictReader(csv_file))

        self.assertEqual(written[0]["audio_path"], "/tmp/sample.wav")
        self.assertEqual(written[0]["profile"], "default")
        self.assertEqual(written[0]["shown_event_count"], "1")

    def test_summarize_evaluation_rows_counts_profile_quality_signals(self):
        rows = [
            {
                "profile": "default",
                "tag": "gunshot",
                "shown_event_count": "1",
                "expected_detected": "yes",
                "gunshot_shown": "right",
                "gunshot_suppressed": "",
            },
            {
                "profile": "default",
                "tag": "vehicle",
                "shown_event_count": "0",
                "expected_detected": "no",
                "gunshot_shown": "",
                "gunshot_suppressed": "front",
            },
            {
                "profile": "default",
                "tag": "bad sample",
                "shown_event_count": "2",
                "expected_detected": "",
                "gunshot_shown": "",
                "gunshot_suppressed": "",
            },
        ]

        summary = summarize_evaluation_rows(rows)[0]

        self.assertEqual(summary["profile"], "default")
        self.assertEqual(summary["rows"], "3")
        self.assertEqual(summary["target_rows"], "2")
        self.assertEqual(summary["target_detected"], "1")
        self.assertEqual(summary["target_missed"], "1")
        self.assertEqual(summary["unknown_or_bad_with_icons"], "1")
        self.assertEqual(summary["multi_icon_rows"], "1")
        self.assertEqual(summary["gunshot_shown_rows"], "1")
        self.assertEqual(summary["gunshot_suppressed_rows"], "1")

    def test_write_summary_csv_writes_profile_rows(self):
        rows = [{"profile": "default", "rows": "1", "target_missed": "0"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_summary_csv(f"{tmpdir}/summary.csv", rows)
            with path.open(newline="", encoding="utf-8") as csv_file:
                written = list(csv.DictReader(csv_file))

        self.assertEqual(written[0]["profile"], "default")
        self.assertEqual(written[0]["rows"], "1")


if __name__ == "__main__":
    unittest.main()
