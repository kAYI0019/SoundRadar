import unittest

from sound_model.event_detection import SoundPulse
from sound_model import radar_visuals


class RadarVisualHelperTests(unittest.TestCase):
    def test_pulse_lifecycle_is_computed_without_qt(self):
        pulse = SoundPulse(sector=3, strength=0.8, created_at=10.0, duration=1.0)

        self.assertEqual(radar_visuals.pulse_age_ratio(pulse, 10.0), 0.0)
        self.assertGreater(radar_visuals.pulse_opacity(pulse, 10.5), 0.0)
        self.assertFalse(radar_visuals.pulse_expired(pulse, 10.99))
        self.assertTrue(radar_visuals.pulse_expired(pulse, 11.0))

    def test_event_icon_center_xy_follows_clock_layout_and_lane_offset(self):
        right = radar_visuals.event_icon_center_xy(
            3,
            720,
            360,
            360,
            distance_ratio=0.4,
            lane_spacing_ratio=0.052,
        )
        first_front_lane = radar_visuals.event_icon_center_xy(
            0,
            720,
            360,
            360,
            distance_ratio=0.4,
            lane_index=0,
            lane_count=2,
            lane_spacing_ratio=0.052,
        )
        second_front_lane = radar_visuals.event_icon_center_xy(
            0,
            720,
            360,
            360,
            distance_ratio=0.4,
            lane_index=1,
            lane_count=2,
            lane_spacing_ratio=0.052,
        )

        self.assertAlmostEqual(right[0], 648.0)
        self.assertAlmostEqual(right[1], 360.0)
        self.assertLess(min(first_front_lane[0], second_front_lane[0]), 360.0)
        self.assertGreater(max(first_front_lane[0], second_front_lane[0]), 360.0)
        self.assertAlmostEqual(first_front_lane[1], second_front_lane[1])

    def test_event_icon_size_accepts_runtime_scale(self):
        pulse = SoundPulse(sector=1, strength=0.9, created_at=10.0, duration=3.5, kind="gunshot")
        common = {
            "min_size_ratio": 0.026,
            "max_size_ratio": 0.044,
            "pop_age_ratio": 0.28,
            "pop_scale": 0.08,
        }

        normal = radar_visuals.event_icon_size(pulse, 10.1, 720, size_scale=1.0, **common)
        enlarged = radar_visuals.event_icon_size(pulse, 10.1, 720, size_scale=1.25, **common)

        self.assertAlmostEqual(enlarged, normal * 1.25)

    def test_watercolor_blobs_are_deterministic_and_move_outward(self):
        pulse = SoundPulse(sector=5, strength=0.8, created_at=10.0, duration=1.0)
        style = {
            "blob_count": 7,
            "angle_spread": 26.0,
            "inner_safe_ratio": 0.44,
            "flow_drift_deg": 9.0,
        }

        start = radar_visuals.watercolor_blob_specs(pulse, 10.0, 720, **style)
        repeated = radar_visuals.watercolor_blob_specs(pulse, 10.0, 720, **style)
        later = radar_visuals.watercolor_blob_specs(pulse, 10.5, 720, **style)

        self.assertEqual(start, repeated)
        self.assertTrue(all(blob.distance >= 720 * 0.44 for blob in start))
        self.assertGreater(
            sum(blob.distance for blob in later),
            sum(blob.distance for blob in start),
        )
        self.assertLess(
            sum(blob.opacity for blob in later),
            sum(blob.opacity for blob in start),
        )


if __name__ == "__main__":
    unittest.main()
