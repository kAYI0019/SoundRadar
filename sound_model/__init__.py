"""PUBG accessibility sound-event model prototype.

The package implements the V0 path from
``pubg_accessibility_sound_model_plan.md`` using only NumPy and the Python
standard library so it can run inside the current SoundRadar venv.
"""

from .audio_features import DEFAULT_CLASSES, extract_feature_vector, log_mel_feature_channels, read_wav
from .model import LoadedModel, load_checkpoint

__all__ = [
    "DEFAULT_CLASSES",
    "LoadedModel",
    "extract_feature_vector",
    "load_checkpoint",
    "log_mel_feature_channels",
    "read_wav",
]
