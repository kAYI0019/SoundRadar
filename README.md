# soundRadar

A 3D audio visualization tool that detects directional sound from any PC audio source and displays it in real-time.

> This project is modified from `soundRadar` by nmatton  
> Original: https://github.com/nmatton/soundRadar  
> License: GPL-3.0 — see [LICENSE](./LICENSE) for details.

---

## 🔧 Installation & Setup

1. Download and install [VB-Cable](https://vb-audio.com/Cable/)
2. Reboot your PC
3. Open **Sound Settings**
4. Go to **Playback** tab
5. Set **CABLE Input** as default device
6. Click **Configure**
7. Set sound to **7.1 Surround** and enable all speakers
8. Go to **Recording** tab
9. Select **CABLE Output** → **Properties**
10. Under **Listen** tab, enable **Listen to this device**
11. Set your main audio output device
12. Run the application:

```sh
python soundRadar.py