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

Run the application:

```sh
.venv/bin/python -u soundRadar.py
```

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
