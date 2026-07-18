import unittest
from types import SimpleNamespace

from sound_model import event_debug
from sound_model.event_detection import DirectionEventPulseDebug, gunshot_display_decision


class EventDebugHelperTests(unittest.TestCase):
    def test_device_and_latency_labels_are_formatted_without_qt(self):
        device = event_debug.direction_event_device_label(
            requested_device="auto",
            resolved_device="mps",
            resolved_dtype="float16",
        )
        latency = event_debug.direction_latency_debug_line(18.4, 255.0)

        self.assertEqual(device, "auto→mps/float16")
        self.assertEqual(latency, "lag radar 18ms model 255ms Δ+237ms")

    def test_debug_lines_render_compass_cells(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "right": {"gunshot": 0.8, "footstep": 0.0, "vehicle": 0.0, "explosion": 0.0},
                "left": {"gunshot": 0.0, "footstep": 0.92, "vehicle": 0.0, "explosion": 0.0},
                "rear_right": {"gunshot": 0.0, "footstep": 0.0, "vehicle": 0.88, "explosion": 0.0},
            },
            active_events_by_direction={
                "right": ["gunshot"],
                "left": ["footstep"],
                "rear_right": ["vehicle"],
            },
        )

        lines = event_debug.direction_event_debug_lines(
            prediction,
            threshold=0.1,
            status="idle",
            teacher_model="efficientat-mn10",
            requested_device="auto",
            resolved_device="mps",
            radar_latency_ms=18.4,
            ast_latency_ms=255.0,
        )

        self.assertEqual(lines[0], "efficientat-mn10 idle auto→mps")
        self.assertIn("L: FOOT .92", lines[5])
        self.assertIn("R: GUN .80", lines[5])
        self.assertIn("RR: VEH .88", lines[6])

    def test_gunshot_line_includes_suppression_and_cooldown(self):
        prediction = SimpleNamespace(
            direction_event_scores={
                "front": {"gunshot": 0.31},
                "front_right": {"gunshot": 0.72},
                "right": {"gunshot": 0.26},
            },
            active_events_by_direction={
                "front": ["gunshot"],
                "front_right": ["gunshot"],
                "right": ["gunshot"],
            },
        )
        decision = gunshot_display_decision(
            prediction.direction_event_scores,
            prediction.active_events_by_direction,
            threshold=0.1,
        )
        display_debug = DirectionEventPulseDebug(
            gunshot_decision=decision,
            gunshot_emitted_directions=(),
            gunshot_global_cooldown_blocked_directions=("front_right",),
            gunshot_sector_cooldown_blocked_directions=(),
        )

        line = event_debug.direction_event_gunshot_debug_line(
            prediction,
            display_debug,
            threshold=0.1,
        )

        self.assertIn("cand FR .72,F .31,R .26", line)
        self.assertIn("show --", line)
        self.assertIn("sup F/R", line)
        self.assertIn("cd G:FR", line)


if __name__ == "__main__":
    unittest.main()
