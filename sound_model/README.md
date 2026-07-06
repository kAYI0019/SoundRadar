# Sound model V0 prototype

This directory now contains an executable V0 prototype derived from
`pubg_accessibility_sound_model_plan.md`.

## What is implemented

- V0 classes: `background`, `footstep`, `gunshot`, `vehicle`, `explosion`
- Stereo-preserving log-mel features: `[left, right, mid, side]`
- Clip-level multilabel training with sigmoid/BCE-style loss
- Class-imbalance weighting and per-class validation thresholds
- Checkpoint save/load plus single-WAV inference CLI
- Deterministic synthetic smoke dataset generation for end-to-end verification

The first trainable local baseline is a NumPy MLP over log-mel summary features.
It is suitable for validating manifests, preprocessing, checkpointing, and
realtime integration shape. It is not a gameplay-quality PUBG classifier until
trained on real labeled PUBG-only audio.

## Smoke training

```sh
cd /Users/min/SoundRadar
.venv/bin/python -m sound_model.train_v0 --generate-smoke-data --epochs 12
```

Outputs:

```text
sound_model/generated/smoke_dataset/manifest.csv
sound_model/artifacts/model_mlp_v0_smoke.npz
sound_model/artifacts/metrics_v0_smoke.json
```

## Real data manifest

Create a CSV with one audio path column plus binary class columns:

```csv
audio_path,split,background,footstep,gunshot,vehicle,explosion
audio/session001_clip0001.wav,train,0,1,0,0,0
audio/session071_clip0005.wav,valid,0,0,1,0,1
audio/session086_clip0008.wav,test,1,0,0,0,0
```

Then run:

```sh
.venv/bin/python -m sound_model.train_v0   --manifest sound_model/dataset_v0/labels_clip_v0.csv   --dataset-root sound_model/dataset_v0   --model-name model_mlp_v0_real.npz   --metrics-name metrics_v0_real.json
```

## Inference

```sh
.venv/bin/python -m sound_model.infer_v0   sound_model/generated/smoke_dataset/audio/gunshot_0000.wav   --checkpoint sound_model/artifacts/model_mlp_v0_smoke.npz
```

## AST AudioSet teacher inference

For a stronger zero-shot / teacher baseline, use the Hugging Face AST model
`MIT/ast-finetuned-audioset-10-10-0.4593`:

```sh
cd /Users/min/SoundRadar
.venv/bin/python -m pip install torch transformers huggingface_hub
.venv/bin/python -m sound_model.ast_teacher \
  /Users/min/Downloads/gun_sound_v2/ak_0m_center_0000.mp3 \
  --device cpu \
  --top-k 12
```

If Hugging Face prints an unauthenticated-request warning, the model will still
run anonymously, but downloads may be slower or rate-limited. Set `HF_TOKEN` for
higher limits:

```sh
export HF_TOKEN=hf_your_token_here
```

The CLI prints raw AudioSet top labels plus mapped SoundRadar V0 events:

```text
background / footstep / gunshot / vehicle / explosion
```

Notes:

- WAV files are read directly.
- MP3 and other non-WAV files are decoded through macOS `afconvert`.
- The AST feature extractor uses the pretrained model's 128-bin mel setup; a
  Transformers mel-filter warning about that setup is benign.
- AST sigmoid scores are not calibrated for PUBG; use the output as teacher
  evidence and tune thresholds with real game/audio-device samples.

## Direction-wise AST teacher, no extra training

Method B is implemented as a no-training teacher pass: split available audio into
seven coarse directions, run the same AST teacher once per direction, and emit a
`direction x event` score matrix.

```sh
.venv/bin/python -m sound_model.direction_events \
  /Users/min/Downloads/gun_sound_v2/ak_0m_center_0000.mp3 \
  --device cpu \
  --top-k 5
```

Direction order:

```text
front_left, front, front_right, left, right, rear_left, rear_right
```

Input behavior:

- 8+ channel WAV: uses Audio MIDI / SoundRadar 7.1 order and ignores LFE.
- Stereo MP3/WAV: falls back to left/right duplication plus front=(L+R)/2. This
  preserves the API but is not true 7-direction evidence.
- Mono: duplicates the same waveform to all seven directions.

This is intended for teacher/pseudo-labeling and prototyping. Real-time use will
likely need batching, cooldown/smoothing, and eventually a smaller student model.

## Important limitation

Synthetic smoke-data metrics only prove the pipeline runs. Per the plan, real
progress requires PUBG-only recordings with Discord/microphone excluded and
session-based train/valid/test splits.
