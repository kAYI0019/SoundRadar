import unittest

import numpy as np

from sound_model import radar_directions


class RadarDirectionHelperTests(unittest.TestCase):
    def test_arc_positions_follow_clock_layout_without_qt(self):
        self.assertEqual(radar_directions.arc_start_deg_for_position(0), 75)
        self.assertEqual(radar_directions.arc_start_deg_for_position(3), -15)
        self.assertEqual(radar_directions.arc_start_deg_for_position(6), -105)
        self.assertEqual(radar_directions.arc_start_deg_for_position(9), 165)

    def test_channel_mapping_keeps_audio_midi_7_1_order(self):
        mapping, mode = radar_directions.build_channel_mapping(16, output_channel_count=16)

        self.assertEqual(mode, "7.1 surround")
        self.assertEqual(mapping[radar_directions.FRONT_LEFT], 0)
        self.assertEqual(mapping[radar_directions.FRONT_RIGHT], 1)
        self.assertEqual(mapping[radar_directions.CENTER], 2)
        self.assertEqual(mapping[radar_directions.LEFT], 4)
        self.assertEqual(mapping[radar_directions.RIGHT], 5)
        self.assertEqual(mapping[radar_directions.REAR_LEFT], 6)
        self.assertEqual(mapping[radar_directions.REAR_RIGHT], 7)

    def test_direction_levels_accept_explicit_directional_ratio(self):
        mapping, _ = radar_directions.build_channel_mapping(8)
        values = np.zeros(8)
        values[mapping[radar_directions.FRONT_LEFT]] = 1.0
        values[mapping[radar_directions.FRONT_RIGHT]] = 0.99

        quiet = radar_directions.compute_direction_levels(values, mapping, min_ratio=0.02)
        sensitive = radar_directions.compute_direction_levels(values, mapping, min_ratio=0.001)

        self.assertEqual(quiet[11], 0.0)
        self.assertGreater(sensitive[11], 0.0)


if __name__ == "__main__":
    unittest.main()
