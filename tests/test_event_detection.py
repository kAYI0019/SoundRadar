import unittest
from types import SimpleNamespace

import numpy as np

from sound_model import event_detection


class EventDetectionTests(unittest.TestCase):
    def test_level_pulses_are_created_without_overlay_dependencies(self):
        levels = np.zeros(12)
        levels[5] = 0.7

        pulses = event_detection.create_pulses_from_levels(levels, now=10.0, threshold=0.02)

        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0].sector, 5)
        self.assertEqual(pulses[0].kind, "sharp")

    def test_gunshot_display_keeps_adjacent_local_maximum(self):
        scores = {
            "front": {"gunshot": 0.31},
            "front_right": {"gunshot": 0.72},
            "right": {"gunshot": 0.26},
        }
        active = {direction: ["gunshot"] for direction in scores}

        decision = event_detection.gunshot_display_decision(scores, active, threshold=0.1)

        self.assertEqual(decision.allowed_directions, frozenset(("front_right",)))
        self.assertEqual(decision.spatially_suppressed_directions, frozenset(("front", "right")))

    def test_smoothing_weights_recent_predictions(self):
        first = SimpleNamespace(
            sample_rate=48_000,
            direction_event_scores={"right": {"gunshot": 0.90, "vehicle": 0.0}},
            active_events_by_direction={"right": ["gunshot"]},
            top_labels_by_direction={"right": [{"label": "Gunshot", "score": 0.90}]},
            source_path="<live>",
        )
        second = SimpleNamespace(
            sample_rate=48_000,
            direction_event_scores={"right": {"gunshot": 0.0, "vehicle": 0.30}},
            active_events_by_direction={"right": ["vehicle"]},
            top_labels_by_direction={"right": [{"label": "Vehicle", "score": 0.30}]},
            source_path="<live>",
        )

        smoothed = event_detection.smooth_direction_event_predictions([first, second], window=2)

        self.assertAlmostEqual(smoothed.direction_event_scores["right"]["gunshot"], 0.30)
        self.assertAlmostEqual(smoothed.direction_event_scores["right"]["vehicle"], 0.20)
        self.assertEqual(smoothed.active_events_by_direction["right"], ["gunshot", "vehicle"])


if __name__ == "__main__":
    unittest.main()
