# SoundRadar Agent Notes

This project is a PyQt5 audio-direction overlay. Future agents should read this file before changing windowing, channel mapping, or drawing behavior.

## Run commands

Use the repository venv, not the system Python:

```sh
cd /Users/min/SoundRadar
.venv/bin/python -u soundRadar.py
```

Focused regression tests:

```sh
cd /Users/min/SoundRadar
.venv/bin/python -m unittest discover -s tests -v
```

There is no separate canonical lint/build command in this repo at the moment. For overlay/window behavior that is hard to assert visually, create a temporary `hermes-verify-*.py` script under the OS temp directory, run it, and remove it afterwards. Report that as **ad-hoc verification**, not as a full build green.

## Architecture overview

`/Users/min/SoundRadar/soundRadar.py` intentionally keeps most logic in one file for a small desktop utility, but responsibilities are separated by helper functions:

- **Overlay/windowing**
  - `overlay_window_flags(platform_name=None)` returns shared Qt flags for macOS and Windows.
  - `configure_overlay_widget(widget, platform_name=None)` applies transparent/click-through/non-activating Qt attributes.
  - `apply_native_overlay_level(widget, platform_name=None)` dispatches native topmost behavior.
  - `apply_macos_overlay_level(widget)` uses Objective-C runtime + CoreGraphics without requiring PyObjC.
  - `apply_windows_overlay_level(widget)` uses Win32 `SetWindowLongPtrW` + `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)`.

- **Geometry/drawing**
  - `arc_start_deg_for_position(position)` maps clock-like radar sectors to `QPainter.drawArc()` degrees.
  - Expected cardinal checks: `{0: 75, 3: -15, 6: -105, 9: 165}`.
  - `fitted_window_size()` keeps the square overlay within 90% of the active screen's smallest dimension.

- **Audio/channel semantics**
  - `build_channel_mapping(channel_count, output_channel_count=None)` chooses 7.1 vs stereo/mono fallback mapping.
  - 7.1 mapping used by this app:
    - `avg` front-left: channel 0
    - `avd` front-right: channel 1
    - `c` center: channel 2
    - channel 3 is LFE and intentionally ignored by the visualizer
    - `g` left/side-left: channel 4
    - `d` right/side-right: channel 5
    - `arg` rear-left: channel 6
    - `ard` rear-right: channel 7
  - `compute_direction_levels(max_values, channel_map)` converts channel peaks into the 12 visual sectors.
  - `centered_pair_strength(first, second)` prevents one-sided rear/front channels from also lighting center.

## Cross-platform overlay rules

### Do not reintroduce polling `raise_()`

Repeated `raise_()` calls can keep the overlay above other windows, but on macOS they can steal focus. The desired behavior is:

```text
Always visible when possible + no focus stealing + click-through
```

So keep these properties:

- no `_top_timer` that periodically calls `raise_()`
- `_native_top_timer` may periodically reapply native no-activate topmost state; it must not call Qt `raise_()`
- `WindowStaysOnTopHint`
- `WindowDoesNotAcceptFocus`
- `WA_ShowWithoutActivating`
- `WA_TransparentForMouseEvents`
- `Qt.NoFocus`

### macOS

Qt flags alone are often insufficient on macOS. `apply_macos_overlay_level()` raises the native `NSWindow` level to `kCGScreenSaverWindowLevelKey`, calls `orderFrontRegardless`, disables `hidesOnDeactivate`, ignores mouse events, and applies:

```text
CanJoinAllSpaces | FullScreenAuxiliary
```

This is the closest PyQt-compatible equivalent to a non-activating overlay without depending on PyObjC.

Known macOS limitation: true exclusive fullscreen games, DRM/protected surfaces, or apps with higher private window levels may still cover the overlay. Prefer borderless windowed/fullscreen-window mode for games.

### Windows

Windows usually handles this pattern better through Qt plus Win32:

- `HWND_TOPMOST` via `SetWindowPos`
- `SWP_NOACTIVATE`
- `WS_EX_NOACTIVATE`
- `WS_EX_TRANSPARENT`
- `WS_EX_TOOLWINDOW`

The implementation is guarded so it only executes on `sys.platform == "win32"`.

## Audio routing notes

- BlackHole 16ch is a virtual device, not a speaker. Selecting it as the only system output means the user will not hear audio through the MacBook speakers/AirPods.
- To hear audio and feed SoundRadar simultaneously, use a Multi-Output Device or a routing tool such as Loopback.
- A 16-channel output device does not guarantee 7.1 content. Browser tests and many apps output only stereo even when BlackHole 16ch is selected.
- For real 7.1 validation, use a discrete multichannel source or a Python/sounddevice channel probe.

## Visual pulse/ripple model

- Direction detection produces 12 sector levels from `compute_direction_levels(...)`.
- The visual layer converts meaningful sector levels into short-lived `SoundPulse` events.
- `SoundPulse.kind` is intentionally basic for now (`unknown`, `sharp`, `impact`) so future agents can attach footstep/gunfire/vehicle classification later without changing the renderer API.
- Ripples should stay in the outer screen region and expand outward. Preserve the center safe zone for aiming/gameplay focus.
- Current default ripple style is `RIPPLE_STYLE="watercolor"`: render pulses as multiple soft `WatercolorBlob` radial-gradient ellipses rather than hard `drawArc()` ripple lines. Keep the blob math in `watercolor_blob_specs(...)` deterministic so tests can verify stable shapes that move outward and fade.
- Keep `SHOW_ARCS=True` as a faint/debug baseline until the user explicitly approves ripple-only mode.
- Use per-sector cooldown (`RIPPLE_COOLDOWN`) and threshold (`RIPPLE_THRESHOLD`) to avoid visual spam from sustained sound.
- Do not tie animation directly to raw channel values; use pulse lifecycle helpers (`pulse_opacity`, `pulse_ripple_radius`, `pulse_expired`) so visuals remain stable and testable.

## Regression checks to preserve

The test suite should keep covering:

- clock-like arc mapping for 0/3/6/9 sectors
- stereo fallback when default output exposes fewer than 8 channels
- 7.1 mapping for 16-channel BlackHole/Loopback input
- Audio MIDI 7.1 channel order: center=2, side L/R=4/5, rear L/R=6/7
- rear-left and rear-right do not both light rear-center
- overlay flags are topmost/non-activating for both `darwin` and `win32`
- native macOS overlay level can be applied without calling `raise_()`

## Style guidance

Keep helpers small and testable. Avoid adding visual/windowing behavior directly inside the main loop when it can be represented as a pure helper and covered by `tests/test_soundRadar.py`.
