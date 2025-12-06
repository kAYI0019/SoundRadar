import ctypes
import sys
import unittest

import soundRadar


class SoundRadarMappingTests(unittest.TestCase):
    def test_arc_positions_follow_clock_layout(self):
        # Position 0 is the front/top wedge. Positions then move clockwise:
        # 3 = right, 6 = back/bottom, 9 = left.
        self.assertEqual(soundRadar.arc_start_deg_for_position(0), 75)
        self.assertEqual(soundRadar.arc_start_deg_for_position(3), -15)
        self.assertEqual(soundRadar.arc_start_deg_for_position(6), -105)
        self.assertEqual(soundRadar.arc_start_deg_for_position(9), 165)

    def test_stereo_output_uses_stereo_mapping_even_when_blackhole_input_has_16_channels(self):
        mapping, mode = soundRadar.build_channel_mapping(16, output_channel_count=2)

        self.assertIn("stereo", mode)
        self.assertEqual(mapping["g"], 0)    # gauche/left
        self.assertEqual(mapping["avg"], 0)  # avant gauche/front-left
        self.assertEqual(mapping["d"], 1)    # droite/right
        self.assertEqual(mapping["avd"], 1)  # avant droite/front-right

    def test_surround_output_keeps_7_1_mapping(self):
        mapping, mode = soundRadar.build_channel_mapping(16, output_channel_count=16)

        self.assertIn("7.1", mode)
        self.assertEqual(mapping["avg"], 0)
        self.assertEqual(mapping["avd"], 1)
        self.assertEqual(mapping["c"], 2)
        self.assertEqual(mapping["g"], 4)
        self.assertEqual(mapping["d"], 5)
        self.assertEqual(mapping["arg"], 6)
        self.assertEqual(mapping["ard"], 7)

    def test_audio_midi_7_1_center_and_rear_channels_are_direct_sectors(self):
        mapping, _ = soundRadar.build_channel_mapping(16, output_channel_count=16)
        cases = [
            (2, 0),  # center -> front/top sector
            (4, 9),  # left surround -> left sector
            (5, 3),  # right surround -> right sector
            (6, 7),  # left rear surround -> rear-left sector
            (7, 5),  # right rear surround -> rear-right sector
        ]

        for channel, expected_sector in cases:
            with self.subTest(channel=channel):
                values = soundRadar.np.zeros(16)
                values[channel] = 0.8
                levels = soundRadar.compute_direction_levels(values, mapping)
                active = [idx for idx, level in enumerate(levels) if level > 0]

                self.assertEqual(active, [expected_sector])

    def test_center_pair_does_not_light_for_single_rear_channel(self):
        self.assertEqual(soundRadar.centered_pair_strength(0.8, 0.0), 0.0)
        self.assertEqual(soundRadar.centered_pair_strength(0.0, 0.8), 0.0)
        self.assertGreater(soundRadar.centered_pair_strength(0.8, 0.75), 0.0)

    def test_direction_levels_keep_rear_left_and_rear_right_distinct(self):
        mapping, _ = soundRadar.build_channel_mapping(16, output_channel_count=16)
        values = soundRadar.np.zeros(16)

        values[mapping["arg"]] = 0.8
        levels = soundRadar.compute_direction_levels(values, mapping)
        self.assertGreater(levels[7], 0.0)  # rear-left sector
        self.assertEqual(levels[6], 0.0)    # rear-center should stay dark
        self.assertEqual(levels[5], 0.0)    # rear-right should stay dark

        values[:] = 0.0
        values[mapping["ard"]] = 0.8
        levels = soundRadar.compute_direction_levels(values, mapping)
        self.assertGreater(levels[5], 0.0)  # rear-right sector
        self.assertEqual(levels[6], 0.0)
        self.assertEqual(levels[7], 0.0)


class SoundRadarWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = soundRadar.QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = soundRadar.QtWidgets.QApplication([])

    def test_overlay_does_not_raise_periodically(self):
        window = soundRadar.ParentWidget()

        self.assertFalse(hasattr(window, "_top_timer") and window._top_timer.isActive())

    def test_overlay_has_native_keep_on_top_pulse_without_qt_raise(self):
        class ProbeWindow(soundRadar.ParentWidget):
            def __init__(self):
                self.raise_called = False
                super().__init__()

            def raise_(self):
                self.raise_called = True
                raise AssertionError("raise_() should not be used")

        window = ProbeWindow()
        self.assertTrue(hasattr(window, "_native_top_timer"))
        self.assertTrue(window._native_top_timer.isActive())
        window.ensure_on_top()
        self.assertFalse(window.raise_called)

    def test_overlay_flags_are_cross_platform_no_activate_topmost(self):
        for platform_name in ("darwin", "win32"):
            flags = soundRadar.overlay_window_flags(platform_name)

            self.assertTrue(flags & soundRadar.QtCore.Qt.WindowStaysOnTopHint)
            self.assertTrue(flags & soundRadar.QtCore.Qt.WindowDoesNotAcceptFocus)
            self.assertTrue(flags & soundRadar.QtCore.Qt.Tool)

    def test_native_overlay_dispatch_returns_false_for_unsupported_platform(self):
        window = soundRadar.ParentWidget()

        self.assertFalse(soundRadar.apply_native_overlay_level(window, platform_name="linux"))

    def test_macos_overlay_level_available(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only overlay level")

        self.assertGreaterEqual(soundRadar.macos_overlay_window_level(), 1000)

    def test_native_overlay_level_can_be_applied_without_raise(self):
        if sys.platform != "darwin":
            self.skipTest("macOS-only overlay level")

        class ProbeWindow(soundRadar.ParentWidget):
            def __init__(self):
                self.raise_called = False
                super().__init__()

            def raise_(self):
                self.raise_called = True
                raise AssertionError("raise_() should not be used")

        window = ProbeWindow()
        window.show()
        self.app.processEvents()

        self.assertTrue(soundRadar.apply_native_overlay_level(window))
        self.assertFalse(window.raise_called)

        ns_window = soundRadar._objc_msg_send(int(window.winId()), "window", restype=ctypes.c_void_p)
        hides_on_deactivate = soundRadar._objc_msg_send(ns_window, "hidesOnDeactivate", restype=ctypes.c_bool)
        ignores_mouse = soundRadar._objc_msg_send(ns_window, "ignoresMouseEvents", restype=ctypes.c_bool)
        self.assertFalse(hides_on_deactivate)
        self.assertTrue(ignores_mouse)


if __name__ == "__main__":
    unittest.main()
