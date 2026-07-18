from types import SimpleNamespace
import unittest

from sound_model.analysis_presenter import build_analysis_summary, summary_from_payload


class AnalysisPresenterTests(unittest.TestCase):
    def prediction(self):
        return SimpleNamespace(
            sample_rate=48000,
            inference_latency_ms=125.4,
            direction_event_scores={
                "front_right": {"gunshot": 0.80, "vehicle": 0.72, "footstep": 0.0, "explosion": 0.1},
                "left": {"gunshot": 0.0, "vehicle": 0.0, "footstep": 0.0, "explosion": 0.0},
            },
            top_labels_by_direction={"front_right": [{"label": "Gunshot", "score": 0.8}]},
        )

    def test_summary_uses_loaded_model_and_korean_ready_structured_rows(self):
        summary = build_analysis_summary(
            self.prediction(),
            audio_path="/tmp/sample.wav",
            requested_model="efficientat-mn10",
            loaded_model="efficientat-mn10",
            device="mps",
            threshold_profile="default",
            analyzed_at="2026-07-19T12:00:00",
            displayed_events=[{"direction": "front_right", "event": "gunshot", "score": 0.8}],
            channel_count=8,
            active_channel_count=8,
        )

        self.assertTrue(summary.model_consistent)
        self.assertEqual(summary.primary_event, "gunshot")
        self.assertEqual(summary.primary_direction, "front_right")
        self.assertIn("gunshot_vehicle_ambiguous", summary.warnings)
        self.assertTrue(summary.direction_rows[0].noteworthy)
        self.assertFalse(summary.direction_rows[1].noteworthy)

    def test_model_mismatch_is_explicit_warning(self):
        summary = build_analysis_summary(
            self.prediction(),
            audio_path="/tmp/sample.wav",
            requested_model="efficientat-mn10",
            loaded_model="efficientat-mn20",
            device="mps",
            threshold_profile="default",
            analyzed_at="2026-07-19T12:00:00",
        )

        self.assertFalse(summary.model_consistent)
        self.assertIn("model_mismatch", summary.warnings)

    def test_legacy_payload_falls_back_to_teacher_model(self):
        summary = summary_from_payload({"audio_path": "/tmp/a.wav", "teacher_model": "ast", "device": "cpu"})

        self.assertEqual(summary["requested_model"], "ast")
        self.assertEqual(summary["loaded_model"], "ast")
        self.assertEqual(summary["display_model"], "ast")


if __name__ == "__main__":
    unittest.main()
