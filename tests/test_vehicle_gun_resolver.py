import unittest

from sound_model.vehicle_gun_resolver import (
    VehicleGunEvidence,
    resolve_vehicle_gun,
    strong_road_vehicle_evidence,
)


class VehicleGunResolverTests(unittest.TestCase):
    def test_high_transient_and_gunshot_scores_resolve_to_gunshot(self):
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.85,
                vehicle_teacher_score=0.15,
                gunshot_label_score=0.80,
                road_vehicle_label_score=0.05,
                transient_score=0.80,
                peak=0.80,
                rms=0.08,
                crest_factor=10.0,
            )
        )

        self.assertEqual(decision.label, "gunshot")
        self.assertGreater(decision.gunshot_evidence, decision.vehicle_evidence)

    def test_high_vehicle_label_and_low_transient_resolve_to_vehicle(self):
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.10,
                vehicle_teacher_score=0.80,
                gunshot_label_score=0.05,
                road_vehicle_label_score=0.75,
                transient_score=0.10,
                peak=0.30,
                rms=0.10,
                crest_factor=3.0,
            )
        )

        self.assertEqual(decision.label, "vehicle")
        self.assertIn("road_vehicle_strong=yes", decision.reason)

    def test_low_engine_score_does_not_block_gunshot(self):
        self.assertFalse(strong_road_vehicle_evidence(0.05, 0.70))
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.75,
                vehicle_teacher_score=0.40,
                gunshot_label_score=0.70,
                road_vehicle_label_score=0.05,
                transient_score=0.80,
                peak=0.80,
                rms=0.08,
                crest_factor=10.0,
            )
        )

        self.assertEqual(decision.label, "gunshot")

    def test_road_vehicle_must_clear_absolute_score_and_gunshot_margin(self):
        self.assertFalse(strong_road_vehicle_evidence(0.29, 0.05))
        self.assertFalse(strong_road_vehicle_evidence(0.50, 0.45))
        self.assertTrue(strong_road_vehicle_evidence(0.80, 0.40))

        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.35,
                vehicle_teacher_score=0.80,
                gunshot_label_score=0.40,
                road_vehicle_label_score=0.80,
                transient_score=0.20,
                peak=0.30,
                rms=0.08,
                crest_factor=4.0,
            )
        )
        self.assertEqual(decision.label, "vehicle")

    def test_similar_evidence_resolves_to_unknown(self):
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.55,
                vehicle_teacher_score=0.55,
                gunshot_label_score=0.50,
                road_vehicle_label_score=0.50,
                transient_score=0.20,
                peak=0.20,
                rms=0.08,
                crest_factor=4.0,
            )
        )

        self.assertEqual(decision.label, "unknown")
        self.assertIn("ambiguous evidence", decision.reason)

    def test_all_low_scores_resolve_to_unknown(self):
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.02,
                vehicle_teacher_score=0.03,
                gunshot_label_score=0.01,
                road_vehicle_label_score=0.02,
                transient_score=0.02,
                peak=0.02,
                rms=0.01,
                crest_factor=2.0,
            )
        )

        self.assertEqual(decision.label, "unknown")
        self.assertIn("low evidence", decision.reason)

    def test_strong_waveform_teacher_conflict_resolves_to_unknown(self):
        decision = resolve_vehicle_gun(
            VehicleGunEvidence(
                gunshot_teacher_score=0.50,
                vehicle_teacher_score=0.75,
                gunshot_label_score=0.45,
                road_vehicle_label_score=0.65,
                transient_score=0.80,
                peak=0.80,
                rms=0.08,
                crest_factor=10.0,
            )
        )

        self.assertEqual(decision.label, "unknown")
        self.assertIn("waveform/teacher conflict", decision.reason)


if __name__ == "__main__":
    unittest.main()
