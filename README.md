# soundRadar

Real-time 360° directional audio radar using virtual surround sound.

A 3D audio visualization tool that detects directional sound from any PC audio source and displays it in real time.

> This project is modified from `soundRadar` by nmatton  
> Original: https://github.com/nmatton/soundRadar  
> License: GPL-3.0 — see [LICENSE](./LICENSE) for details.

---

## Installation & Setup

### macOS

1. Install BlackHole 16ch:

```sh
brew install --cask blackhole-16ch
```

2. Reboot macOS.
3. Open **Audio MIDI Setup**.
4. Select **BlackHole 16ch**.
5. Set the format to **48,000 Hz** and **16 channels**.
6. Click **Configure Speakers** and choose **7.1 Surround** if available.
7. Open **System Settings > Sound > Output** and select **BlackHole 16ch**.

If you also need to hear the audio while SoundRadar analyzes it, create a Multi-Output Device:

1. Open **Audio MIDI Setup**:

```sh
open -a "Audio MIDI Setup"
```

2. Click the **+** button at the bottom-left of the Audio Devices window.
3. Select **Create Multi-Output Device**.
4. Rename the new device to something like **BlackHole + Speakers**.
5. In the right-side device list, enable **Use** for:

```text
BlackHole 16ch
MacBook Air Speakers or your headphones
```

6. Set **Primary Device** to **BlackHole 16ch**.
7. If **Drift Correction** is available, enable it for the speakers/headphones device.
8. Open **System Settings > Sound > Output** and select **BlackHole + Speakers**.

The routing should look like this:

```text
Game/browser/system audio
        ↓
BlackHole + Speakers
        ├─ BlackHole 16ch -> soundRadar.py analyzes the input
        └─ Speakers/headphones -> you hear the audio
```

SoundRadar should still use **BlackHole 16ch** as its input device. The Multi-Output Device is only used as the macOS output device.

Verify that BlackHole is exposed as a multichannel input:

```sh
.venv/bin/python -c "import sounddevice as sd; [print(i, d['name'], d['max_input_channels'], d['max_output_channels']) for i,d in enumerate(sd.query_devices())]"
```

Expected result:

```text
BlackHole 16ch 16 16
```

YouTube and most browser playback are usually stereo/downmixed, so they are not
reliable 7.1 routing tests. Use the local routing probe to verify that discrete
Audio MIDI 7.1 channels reach SoundRadar without browser/player downmixing:

```sh
.venv/bin/python -m sound_model.surround_probe --list-devices
.venv/bin/python -m sound_model.surround_probe --device "BlackHole 16ch" --channels 16 --kind both
```

Run SoundRadar with BlackHole 16ch selected as its input while the probe plays.
The probe is a channel-routing check, not a classifier benchmark: the synthetic
gunshot/vehicle sounds are only there to make per-channel overlay movement easy
to see. It emits directions in Audio MIDI 7.1 order: front-left, front,
front-right, left, right, rear-left, rear-right.

To inspect the channel layout in a DAW or multichannel-capable player, write an
8/16-channel WAV instead:

```sh
.venv/bin/python -m sound_model.surround_probe --kind both --channels 8 --write-wav /tmp/soundradar-7.1-probe.wav --no-play
```

For real game/app tuning, capture the actual BlackHole 16ch input and replay it
through the direction-event teacher:

```sh
.venv/bin/python -m sound_model.capture_direction_sample_gui
```

The capture GUI separates the Korean result summary, direction table, profile
comparison, review queue, evaluation, and raw developer log into tabs. Analysis
also writes a backward-compatible schema-v2 `*.analysis.json` file next to the
WAV. It records the requested model, actually loaded model, device, primary
event/direction, warnings, detailed scores, and the legacy teacher/HUD fields.
Click **프로필 비교** to write a
`*.profile-comparison.json` file and compare `default`, `quiet`, `aggressive`,
and `debug` display behavior. Reviewed labels update the existing CSV record
instead of appending a duplicate; `gunshot + vehicle` is supported. Legacy CSV
rows remain readable. If the live overlay is running, F8/F9/F10 save a pending
5-second gunshot/vehicle/unknown sample and F11 saves a pending 10-second review
sample. The GUI waits for the matching result ID, connects the real WAV, and
adds it to the review queue. These shortcuts are application-wide while the GUI
is running; OS-global registration is not included yet.
The same capture path is available as a CLI:

```sh
.venv/bin/python -m sound_model.capture_direction_sample --device "BlackHole 16ch" --seconds 20 --out /tmp/pubg-sample.wav
.venv/bin/python -m sound_model.direction_events /tmp/pubg-sample.wav --teacher-model ast --device auto --top-k 5
```

This preserves the captured multichannel layout, so it is the preferred way to
debug gunshot/vehicle thresholds and direction suppression with repeatable real
samples.

Re-evaluate a tagged sample library after changing thresholds:

```sh
.venv/bin/python -m sound_model.evaluate_sample_library \
  ~/SoundRadarSamples/sample_library.csv \
  --teacher-model ast \
  --device auto \
  --top-k 5
```

This writes both a per-sample evaluation CSV and a profile summary CSV by
default. The summary counts missed target tags, unknown/bad samples that still
show icons, multi-icon rows, and gunshot suppression rows.

Run the application:

```sh
.venv/bin/python -u soundRadar.py
```

Local runtime settings can be kept out of git:

```sh
cp soundradar.local.example.json soundradar.local.json
```

Edit `soundradar.local.json` for machine-specific settings such as
`teacher_model`, `device`, `top_k`, `threshold_profile`, and rolling capture
paths. The file is ignored by git, so switching between `ast` and
`efficientat-mn20` no longer requires editing `soundRadar.py`. Visual tuning can
also live there: `event_icon_scale`, `event_icon_opacity`,
`event_icon_labels`, `event_smoothing_enabled`, and `event_smoothing_window`.
The default live analysis uses a 1.0 second window every 0.5 seconds. If a
local `interval_seconds` exceeds `window_seconds`, SoundRadar caps the interval
to the window length so no audio interval is skipped.

Threshold/cooldown tuning can be selected without editing code:

```sh
.venv/bin/python -u soundRadar.py --threshold-profile quiet
SOUNDRADAR_THRESHOLD_PROFILE=aggressive .venv/bin/python -u soundRadar.py
```

Available profiles are `default`, `quiet`, `aggressive`, and `debug`. `quiet`
raises thresholds and lengthens cooldowns to reduce visual spam, while
`aggressive` and `debug` make detection/display more permissive for tuning.

While the live overlay is running, save the latest rolling audio buffer for a
bad/interesting moment by creating the trigger file:

```sh
touch /tmp/soundradar-save-rolling
```

By default this writes a WAV plus JSON metadata under
`~/SoundRadarSamples/rolling`. The JSON includes peak summary, channel sanity
messages, threshold profile, and the latest direction-event prediction when one
is available.

### Optional Hugging Face AST model

SoundRadar can load the optional AST AudioSet teacher model for direction-event
hints. On first use, Hugging Face may print an unauthenticated-request warning.
Anonymous downloads work, but setting a token gives higher rate limits:

```sh
export HF_TOKEN=hf_your_token_here
.venv/bin/python -u soundRadar.py
```

If you run the AST teacher manually and see a Transformers mel-filter warning
about `num_mel_filters=128` and `num_frequency_bins=257`, it is from the
pretrained AST feature extractor defaults, not from your BlackHole or speaker
configuration.

If the GUI opens but does not react and debug output stays at zero:

1. Open **System Settings > Privacy & Security > Microphone**.
2. Enable microphone access for the app that launches SoundRadar, such as **Terminal**, **iTerm**, **IntelliJ IDEA**, **PyCharm**, **VS Code**, or **Python**.
3. Quit and restart the terminal/IDE, then run SoundRadar again.
4. Open **System Settings > Sound > Output** and confirm that the selected output is **BlackHole 16ch** or the Multi-Output Device that includes **BlackHole 16ch**.
5. In **Audio MIDI Setup**, confirm that the Multi-Output Device has **Use** enabled for both **BlackHole 16ch** and your speakers/headphones.

### Windows

1. Download and install [VB-Cable](https://vb-audio.com/Cable/).
2. Reboot your PC.
3. Open **Sound Settings**.
4. Go to the **Playback** tab.
5. Set **CABLE Input** as the default device.
6. Click **Configure**.
7. Set sound to **7.1 Surround** and enable all speakers.
8. Go to the **Recording** tab.
9. Select **CABLE Output** > **Properties**.
10. Under the **Listen** tab, enable **Listen to this device**.
11. Set your main audio output device.
12. Run the application:

```sh
.venv\Scripts\python.exe soundRadar.py
```

If the selected input device exposes fewer than 8 input channels, the app will still run but direction display is limited to a mono/stereo approximation.
