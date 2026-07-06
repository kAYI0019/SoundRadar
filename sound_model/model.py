"""Small NumPy multilabel model used for V0 smoke training.

The markdown plan's production target is a lightweight CNN/CRNN.  The current
repository venv does not include PyTorch, so this module provides a dependency-
free MLP baseline over log-mel summary features.  It is intentionally saved with
all preprocessing metadata so it can be replaced by a CNN checkpoint later
without changing the manifest/training workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .audio_features import DEFAULT_CLASSES, extract_feature_vector, read_wav

_EPS = 1e-8


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    positive = logits >= 0
    negative = ~positive
    result = np.empty_like(logits, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[negative])
    result[negative] = exp_logits / (1.0 + exp_logits)
    return result


def bce_with_logits(logits: np.ndarray, targets: np.ndarray, pos_weight: np.ndarray | None = None) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    if pos_weight is not None:
        weights = 1.0 + targets * (pos_weight[None, :] - 1.0)
        loss = loss * weights
    return float(loss.mean())


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std < 1e-5] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean[None, :]) / self.std[None, :]).astype(np.float32)


@dataclass
class TrainingHistory:
    rows: list[dict[str, float]]


class MultiLabelMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": (rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": (rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))).astype(np.float32),
            "b2": np.zeros(output_dim, dtype=np.float32),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.params["W1"] + self.params["b1"][None, :]
        hidden = np.maximum(z1, 0.0)
        logits = hidden @ self.params["W2"] + self.params["b2"][None, :]
        return z1, hidden, logits

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_logits(x))


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    hidden_dim: int = 96,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    l2: float = 1e-4,
) -> tuple[MultiLabelMLP, Standardizer, np.ndarray, TrainingHistory]:
    if len(x_train) == 0:
        raise ValueError("training split is empty")
    if x_valid.size == 0:
        x_valid = x_train
        y_valid = y_train

    standardizer = Standardizer.fit(x_train)
    x_train_n = standardizer.transform(x_train)
    x_valid_n = standardizer.transform(x_valid)

    positive = y_train.sum(axis=0)
    negative = max(1, len(y_train)) - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 25.0).astype(np.float32)

    model = MultiLabelMLP(x_train.shape[1], hidden_dim, y_train.shape[1], seed=seed)
    rng = np.random.default_rng(seed)
    adam_m = {name: np.zeros_like(value) for name, value in model.params.items()}
    adam_v = {name: np.zeros_like(value) for name, value in model.params.items()}
    beta1, beta2 = 0.9, 0.999
    step = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train_n))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            xb = x_train_n[indices]
            yb = y_train[indices]
            z1, hidden, logits = model.forward(xb)
            probs = sigmoid(logits)
            weights = 1.0 + yb * (pos_weight[None, :] - 1.0)
            scale = 1.0 / float(np.prod(yb.shape))
            dlogits = (probs - yb) * weights * scale

            grads: dict[str, np.ndarray] = {}
            grads["W2"] = hidden.T @ dlogits + l2 * model.params["W2"]
            grads["b2"] = dlogits.sum(axis=0)
            dhidden = dlogits @ model.params["W2"].T
            dz1 = dhidden * (z1 > 0.0)
            grads["W1"] = xb.T @ dz1 + l2 * model.params["W1"]
            grads["b1"] = dz1.sum(axis=0)

            step += 1
            for name, grad in grads.items():
                adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * grad
                adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (grad * grad)
                m_hat = adam_m[name] / (1.0 - beta1 ** step)
                v_hat = adam_v[name] / (1.0 - beta2 ** step)
                model.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        train_logits = model.predict_logits(x_train_n)
        valid_logits = model.predict_logits(x_valid_n)
        valid_probs = sigmoid(valid_logits)
        metrics = multilabel_metrics(y_valid, valid_probs)
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": bce_with_logits(train_logits, y_train, pos_weight),
                "valid_loss": bce_with_logits(valid_logits, y_valid, pos_weight),
                "valid_macro_f1": metrics["macro_f1"],
                "valid_micro_f1": metrics["micro_f1"],
            }
        )

    return model, standardizer, pos_weight, TrainingHistory(history_rows)


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for class_idx in range(y_true.shape[1]):
        if y_true[:, class_idx].sum() <= 0:
            continue
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in np.linspace(0.1, 0.9, 17):
            pred = (probabilities[:, class_idx] >= threshold).astype(np.float32)
            tp = float(((pred == 1) & (y_true[:, class_idx] == 1)).sum())
            fp = float(((pred == 1) & (y_true[:, class_idx] == 0)).sum())
            fn = float(((pred == 0) & (y_true[:, class_idx] == 1)).sum())
            precision = tp / max(tp + fp, _EPS)
            recall = tp / max(tp + fn, _EPS)
            f1 = 2.0 * precision * recall / max(precision + recall, _EPS)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[class_idx] = best_threshold
    return thresholds


def multilabel_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> dict[str, object]:
    y_true = y_true.astype(np.float32)
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    pred = (probabilities >= thresholds[None, :]).astype(np.float32)
    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    total_tp = total_fp = total_fn = 0.0
    for idx, name in enumerate(classes):
        tp = float(((pred[:, idx] == 1) & (y_true[:, idx] == 1)).sum())
        fp = float(((pred[:, idx] == 1) & (y_true[:, idx] == 0)).sum())
        fn = float(((pred[:, idx] == 0) & (y_true[:, idx] == 1)).sum())
        precision = tp / max(tp + fp, _EPS)
        recall = tp / max(tp + fn, _EPS)
        f1 = 2.0 * precision * recall / max(precision + recall, _EPS)
        support = float(y_true[:, idx].sum())
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "threshold": float(thresholds[idx]),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / max(total_tp + total_fp, _EPS)
    micro_recall = total_tp / max(total_tp + total_fn, _EPS)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, _EPS)
    return {
        "per_class": per_class,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def save_checkpoint(
    path: str | Path,
    model: MultiLabelMLP,
    standardizer: Standardizer,
    *,
    classes: tuple[str, ...],
    thresholds: np.ndarray,
    feature_config: dict[str, object],
    metrics: dict[str, object],
    pos_weight: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "classes": list(classes),
        "thresholds": thresholds.astype(float).tolist(),
        "feature_config": feature_config,
        "metrics": metrics,
        "architecture": "numpy_logmel_mlp_v0",
        "purpose": "PUBG accessibility sound-event V0 smoke/baseline model",
    }
    np.savez_compressed(
        path,
        W1=model.params["W1"],
        b1=model.params["b1"],
        W2=model.params["W2"],
        b2=model.params["b2"],
        mean=standardizer.mean,
        std=standardizer.std,
        pos_weight=pos_weight,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )


@dataclass
class LoadedModel:
    params: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    classes: tuple[str, ...]
    thresholds: np.ndarray
    feature_config: dict[str, object]
    metadata: dict[str, object]

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        x = ((features[None, :] - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        hidden = np.maximum(x @ self.params["W1"] + self.params["b1"][None, :], 0.0)
        logits = hidden @ self.params["W2"] + self.params["b2"][None, :]
        return sigmoid(logits)[0]

    def predict_audio(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        features = extract_feature_vector(audio, sample_rate, **self.feature_config)
        probabilities = self.predict_proba_from_features(features)
        return {name: float(probabilities[idx]) for idx, name in enumerate(self.classes)}

    def predict_wav(self, path: str | Path) -> dict[str, float]:
        audio, sample_rate = read_wav(path)
        return self.predict_audio(audio, sample_rate)

    def active_labels(self, probabilities: dict[str, float]) -> list[str]:
        return [name for idx, name in enumerate(self.classes) if probabilities[name] >= float(self.thresholds[idx])]


def load_checkpoint(path: str | Path) -> LoadedModel:
    checkpoint = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(checkpoint["metadata_json"]))
    params = {name: checkpoint[name].astype(np.float32) for name in ("W1", "b1", "W2", "b2")}
    return LoadedModel(
        params=params,
        mean=checkpoint["mean"].astype(np.float32),
        std=checkpoint["std"].astype(np.float32),
        classes=tuple(metadata["classes"]),
        thresholds=np.array(metadata["thresholds"], dtype=np.float32),
        feature_config=dict(metadata["feature_config"]),
        metadata=metadata,
    )
