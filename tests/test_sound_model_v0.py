import argparse
import tempfile
import unittest
from pathlib import Path

from sound_model.audio_features import extract_feature_vector, read_wav
from sound_model.dataset import DEFAULT_CLASSES, generate_synthetic_dataset, load_manifest
from sound_model.model import load_checkpoint
from sound_model.train_v0 import train_from_manifest


class SoundModelV0Tests(unittest.TestCase):
    def test_log_mel_summary_vector_has_expected_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_synthetic_dataset(tmp, count_per_class=1, sample_rate=4000, seed=123)
            first = load_manifest(manifest)[0]
            audio, sample_rate = read_wav(first.path)

            vector = extract_feature_vector(
                audio,
                sample_rate,
                target_rate=4000,
                window_sec=1.0,
                n_fft=256,
                stft_hop=128,
                mel_bins=12,
            )

        # 4 stereo-derived channels x 3 summary stats x mel bins.
        self.assertEqual(vector.shape, (4 * 3 * 12,))

    def test_smoke_training_writes_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                classes=",".join(DEFAULT_CLASSES),
                generate_smoke_data=True,
                smoke_data_dir=root / "smoke",
                smoke_count_per_class=2,
                sample_rate=4000,
                seed=7,
                manifest=None,
                dataset_root=None,
                output_dir=root / "artifacts",
                model_name="model.npz",
                metrics_name="metrics.json",
                epochs=2,
                batch_size=4,
                hidden_dim=12,
                learning_rate=1e-3,
                l2=1e-4,
                window_sec=1.0,
                n_fft=256,
                stft_hop=128,
                mel_bins=12,
            )

            metrics = train_from_manifest(args)
            checkpoint = Path(metrics["checkpoint"])
            model = load_checkpoint(checkpoint)
            sample = root / "smoke" / "audio" / "gunshot_0000.wav"
            probabilities = model.predict_wav(sample)

            self.assertTrue(checkpoint.exists())
            self.assertTrue((root / "artifacts" / "metrics.json").exists())
            self.assertEqual(set(probabilities), set(DEFAULT_CLASSES))


if __name__ == "__main__":
    unittest.main()
