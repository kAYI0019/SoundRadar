import csv
import json
from pathlib import Path
import tempfile
import unittest

from sound_model.sample_library import build_sample_record, read_library, upsert_library_record


class SampleLibraryTests(unittest.TestCase):
    def test_legacy_rows_are_read_as_reviewed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample_library.csv"
            with path.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=("created_at", "audio_path", "tag", "notes", "analysis_path", "peak_summary"))
                writer.writeheader()
                writer.writerow({"created_at": "now", "audio_path": "/tmp/a.wav", "tag": "gunshot"})

            row = read_library(path)[0]

        self.assertEqual(row["review_status"], "reviewed")
        self.assertEqual(json.loads(row["reviewed_labels"]), ["gunshot"])

    def test_same_audio_is_updated_without_duplicate_and_keeps_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.wav"
            audio.write_bytes(b"same audio")
            library = Path(tmpdir) / "sample_library.csv"
            first = build_sample_record(audio_path=audio, provisional_tag="vehicle", review_status="pending", sample_id="fixed-id")
            upsert_library_record(library, first)
            reviewed = build_sample_record(audio_path=audio, reviewed_labels=("vehicle", "gunshot"), review_status="reviewed")
            _, created = upsert_library_record(library, reviewed)
            rows = read_library(library)

        self.assertFalse(created)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "fixed-id")
        self.assertEqual(json.loads(rows[0]["reviewed_labels"]), ["vehicle", "gunshot"])


if __name__ == "__main__":
    unittest.main()
