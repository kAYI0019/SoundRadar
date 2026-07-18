from types import SimpleNamespace
import unittest

from sound_model.event_temporal_state import EventTemporalState
from sound_model.vehicle_gun_resolver import VehicleGunEvidence, resolve_vehicle_gun


def prediction(gunshot=0.0, vehicle=0.0, *, road_vehicle=None):
    road_vehicle = vehicle if road_vehicle is None else road_vehicle
    evidence = VehicleGunEvidence(
        gunshot_teacher_score=gunshot,
        vehicle_teacher_score=vehicle,
        gunshot_label_score=gunshot,
        road_vehicle_label_score=road_vehicle,
        transient_score=gunshot,
        peak=gunshot,
        rms=max(gunshot, vehicle) * 0.1,
        crest_factor=12.0 if gunshot else 3.0,
    )
    decision = resolve_vehicle_gun(evidence)
    scores = {"background": 0.0, "footstep": 0.0, "gunshot": gunshot, "vehicle": vehicle, "explosion": 0.0}
    return SimpleNamespace(
        direction_event_scores={"right": dict(scores)},
        raw_direction_event_scores={"right": dict(scores)},
        active_events_by_direction={"right": [name for name in ("gunshot", "vehicle") if scores[name] > 0.0]},
        vehicle_gun_evidence_by_direction={"right": evidence},
        vehicle_gun_decisions_by_direction={"right": decision},
    )


class EventTemporalStateTests(unittest.TestCase):
    def test_vehicle_one_frame_rise_does_not_activate(self):
        state = EventTemporalState()

        result = state.apply(prediction(vehicle=0.80))

        self.assertEqual(result.direction_event_scores["right"]["vehicle"], 0.0)
        self.assertNotIn("vehicle", result.active_events_by_direction["right"])
        self.assertEqual(result.vehicle_gun_evidence_by_direction["right"].vehicle_persistence, 0.5)

    def test_vehicle_persistence_activates_and_keep_threshold_holds(self):
        state = EventTemporalState()
        state.apply(prediction(vehicle=0.80))

        active = state.apply(prediction(vehicle=0.75))
        kept = state.apply(prediction(vehicle=0.45))

        self.assertIn("vehicle", active.active_events_by_direction["right"])
        self.assertGreaterEqual(active.direction_event_scores["right"]["vehicle"], 0.60)
        self.assertIn("vehicle", kept.active_events_by_direction["right"])
        self.assertGreaterEqual(kept.direction_event_scores["right"]["vehicle"], 0.40)

    def test_vehicle_releases_after_configured_missing_frames(self):
        state = EventTemporalState()
        state.apply(prediction(vehicle=0.80))
        state.apply(prediction(vehicle=0.80))

        first_missing = state.apply(prediction())
        second_missing = state.apply(prediction())

        self.assertIn("vehicle", first_missing.active_events_by_direction["right"])
        self.assertNotIn("vehicle", second_missing.active_events_by_direction["right"])
        self.assertEqual(second_missing.direction_event_scores["right"]["vehicle"], 0.0)

    def test_gunshot_activates_immediately_and_decays_below_threshold_next_frame(self):
        state = EventTemporalState()

        onset = state.apply(prediction(gunshot=0.85, road_vehicle=0.0))
        after = state.apply(prediction())

        self.assertIn("gunshot", onset.active_events_by_direction["right"])
        self.assertGreaterEqual(onset.direction_event_scores["right"]["gunshot"], 0.85)
        self.assertLess(after.direction_event_scores["right"]["gunshot"], 0.10)
        self.assertNotIn("gunshot", after.active_events_by_direction["right"])

    def test_temporal_result_keeps_raw_resolver_and_temporal_layers_distinct(self):
        state = EventTemporalState()

        result = state.apply(prediction(vehicle=0.80))

        self.assertEqual(result.raw_direction_event_scores["right"]["vehicle"], 0.80)
        self.assertGreater(result.resolver_direction_event_scores["right"]["vehicle"], 0.0)
        self.assertEqual(result.temporal_direction_event_scores["right"]["vehicle"], 0.0)


if __name__ == "__main__":
    unittest.main()
