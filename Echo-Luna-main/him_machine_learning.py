"""Machine learning utilities built around the H.I.M. architecture.

This module provides lightweight machine-learning helpers that leverage the
existing H.I.M. model (`HIMModel`) for feature extraction instead of external
transformer embeddings. The goal is to keep experimentation self-contained
within the Echo-Luna ecosystem while still enabling classic ML techniques.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gpu_utils import AccelerationState, get_acceleration_state, require_torch
from him_model import HIMModel


@dataclass
class TrainingExample:
    """Container describing a single training example for ML workflows."""

    text: str
    label: int
    gpu_image: Optional[np.ndarray] = None
    mouse_pos: Optional[Dict[str, int]] = None


class HIMFeatureExtractor:
    """Convert textual stimuli into numerical features using the H.I.M. model."""

    def __init__(self, him_model: Optional[HIMModel] = None, *, reset_between_samples: bool = True):
        self._shared_model = him_model
        self.reset_between_samples = reset_between_samples

    def _select_model(self) -> HIMModel:
        if self.reset_between_samples or self._shared_model is None:
            return HIMModel()
        return self._shared_model

    def transform(
        self,
        texts: Sequence[str],
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
        suppress_output: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        """Transform raw inputs into feature vectors and metadata."""

        features: List[np.ndarray] = []
        metadata: List[Dict[str, object]] = []

        for index, text in enumerate(texts):
            model = self._select_model()
            image = gpu_images[index] if gpu_images is not None else None
            mouse = mouse_positions[index] if mouse_positions is not None else None

            vector, meta = model.extract_feature_vector(
                text,
                gpu_image=image,
                mouse_pos=mouse,
                cleanup=True,
                suppress_output=suppress_output,
            )

            features.append(vector)
            metadata.append(meta)

        if not features:
            return np.empty((0, 0), dtype=np.float32), metadata

        return np.vstack(features), metadata


class HIMLogisticTrainer:
    """Simple logistic regression classifier trained on H.I.M. features."""

    def __init__(self, extractor: Optional[HIMFeatureExtractor] = None):
        self.extractor = extractor or HIMFeatureExtractor()
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.training_loss_: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid helper used across training strategies."""

        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        learning_rate: float = 0.1,
        epochs: int = 200,
        l2_penalty: float = 0.0,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> "HIMLogisticTrainer":
        """Train a logistic regression classifier using gradient descent."""

        X, _ = self.extractor.transform(
            texts,
            gpu_images=gpu_images,
            mouse_positions=mouse_positions,
        )
        if X.size == 0:
            raise ValueError("No features generated; provide at least one training sample.")

        y = np.asarray(labels, dtype=np.float32)
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of labels must match number of samples.")

        y = y.reshape(-1, 1)
        X = X.astype(np.float32)
        self.training_loss_.clear()

        state: AccelerationState = get_acceleration_state()
        if state.backend == "torch" and state.using_gpu:
            try:
                self._fit_torch(X, y, learning_rate, epochs, l2_penalty, state)
                return self
            except Exception as gpu_error:  # pragma: no cover - defensive fallback
                print(f"GPU logistic training fallback: {gpu_error}")
                self.training_loss_.clear()

        self._fit_numpy(X, y, learning_rate, epochs, l2_penalty)
        return self

    def _fit_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        l2_penalty: float,
    ) -> None:
        """Train the classifier using NumPy operations."""

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1), dtype=np.float32)
        self.bias = 0.0

        for epoch in range(epochs):
            logits = X @ self.weights + self.bias
            predictions = self._sigmoid(logits)
            error = predictions - y

            gradient_w = (X.T @ error) / n_samples
            if l2_penalty:
                gradient_w = gradient_w + l2_penalty * self.weights
            gradient_b = float(np.mean(error))

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

            if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
                loss = -np.mean(
                    y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8)
                )
                if l2_penalty:
                    loss += 0.5 * l2_penalty * float(np.sum(self.weights ** 2))
                self.training_loss_.append(float(loss))

    def _fit_torch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        l2_penalty: float,
        state: AccelerationState,
    ) -> None:
        """Train the classifier using PyTorch on the detected accelerator."""

        lib = require_torch()
        device = state.device
        tensor_X = lib.as_tensor(X, dtype=lib.float32, device=device)
        tensor_y = lib.as_tensor(y, dtype=lib.float32, device=device)
        n_samples, n_features = tensor_X.shape

        weights = lib.zeros((n_features, 1), dtype=lib.float32, device=device)
        bias = lib.zeros(1, dtype=lib.float32, device=device)

        if state.using_gpu:
            print(f"Logistic trainer: running on {state.device} acceleration")
        lr = float(learning_rate)

        for epoch in range(epochs):
            logits = tensor_X @ weights + bias
            predictions = lib.sigmoid(logits)
            error = predictions - tensor_y

            gradient_w = (tensor_X.transpose(0, 1) @ error) / n_samples
            if l2_penalty:
                gradient_w = gradient_w + l2_penalty * weights
            gradient_b = lib.mean(error)

            weights = weights - lr * gradient_w
            bias = bias - lr * gradient_b

            if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
                loss = -(
                    tensor_y * lib.log(predictions + 1e-8)
                    + (1 - tensor_y) * lib.log(1 - predictions + 1e-8)
                ).mean()
                if l2_penalty:
                    loss = loss + 0.5 * l2_penalty * lib.sum(weights ** 2)
                self.training_loss_.append(float(loss.detach().cpu()))

        self.weights = weights.detach().cpu().numpy()
        self.bias = float(bias.detach().cpu())


    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> np.ndarray:
        """Predict probabilities for the positive class."""

        if self.weights is None:
            raise RuntimeError("Model has not been trained yet.")

        X, _ = self.extractor.transform(
            texts,
            gpu_images=gpu_images,
            mouse_positions=mouse_positions,
        )
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Feature dimension mismatch. Reuse the same extractor setup as training.")

        logits = X @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(
        self,
        texts: Sequence[str],
        threshold: float = 0.5,
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> np.ndarray:
        """Predict binary labels using the learned classifier."""

        probabilities = self.predict_proba(
            texts,
            gpu_images=gpu_images,
            mouse_positions=mouse_positions,
        )
        return (probabilities >= threshold).astype(int).ravel()

    def score(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> float:
        """Compute accuracy of the classifier on a labelled dataset."""

        predictions = self.predict(
            texts,
            gpu_images=gpu_images,
            mouse_positions=mouse_positions,
        )
        y_true = np.asarray(labels, dtype=int)
        if y_true.shape[0] != predictions.shape[0]:
            raise ValueError("Number of labels must match number of samples.")
        return float(np.mean(predictions == y_true))


def batch_fit_examples(
    examples: Iterable[TrainingExample],
    *,
    trainer: Optional[HIMLogisticTrainer] = None,
    learning_rate: float = 0.1,
    epochs: int = 200,
    l2_penalty: float = 0.0,
) -> HIMLogisticTrainer:
    """Utility to train a classifier from ``TrainingExample`` instances."""

    trainer = trainer or HIMLogisticTrainer()
    texts: List[str] = []
    labels: List[int] = []
    images: List[Optional[np.ndarray]] = []
    mice: List[Optional[Dict[str, int]]] = []

    for example in examples:
        texts.append(example.text)
        labels.append(example.label)
        images.append(example.gpu_image)
        mice.append(example.mouse_pos)

    trainer.fit(
        texts,
        labels,
        learning_rate=learning_rate,
        epochs=epochs,
        l2_penalty=l2_penalty,
        gpu_images=images,
        mouse_positions=mice,
    )

    return trainer


class OnlineHIMLogisticTrainer(HIMLogisticTrainer):
    """Logistic trainer that supports streaming updates and concurrency."""

    def __init__(
        self,
        extractor: Optional[HIMFeatureExtractor] = None,
        *,
        learning_rate: float = 0.05,
        l2_penalty: float = 0.0,
    ) -> None:
        super().__init__(extractor=extractor)
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self._lock = threading.Lock()
        self.recent_loss_window: deque[float] = deque(maxlen=50)
        self.update_count: int = 0

    def update(
        self,
        text: str,
        label: int,
        *,
        gpu_image: Optional[np.ndarray] = None,
        mouse_pos: Optional[Dict[str, int]] = None,
    ) -> float:
        """Perform a single stochastic gradient descent step."""

        X, _ = self.extractor.transform(
            [text],
            gpu_images=[gpu_image] if gpu_image is not None else None,
            mouse_positions=[mouse_pos] if mouse_pos is not None else None,
        )
        if X.size == 0:
            raise ValueError("No features generated for online update.")

        features = X.astype(np.float32)
        label_value = np.array([[float(label)]], dtype=np.float32)

        with self._lock:
            if self.weights is None or self.weights.shape[0] != features.shape[1]:
                self.weights = np.zeros((features.shape[1], 1), dtype=np.float32)
                self.bias = 0.0

            logits = features @ self.weights + self.bias
            predictions = self._sigmoid(logits)
            error = predictions - label_value

            gradient_w = features.T * float(error)
            if self.l2_penalty:
                gradient_w += self.l2_penalty * self.weights
            gradient_b = float(error)

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            loss = -(
                label_value * np.log(predictions + 1e-8)
                + (1 - label_value) * np.log(1 - predictions + 1e-8)
            )
            loss_value = float(loss.squeeze())
            self.training_loss_.append(loss_value)
            self.recent_loss_window.append(loss_value)
            self.update_count += 1

            return float(predictions.squeeze())

    def partial_fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> "OnlineHIMLogisticTrainer":
        """Iteratively update the model with multiple samples."""

        for index, text in enumerate(texts):
            image = gpu_images[index] if gpu_images is not None else None
            mouse = mouse_positions[index] if mouse_positions is not None else None
            self.update(text, int(labels[index]), gpu_image=image, mouse_pos=mouse)
        return self

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        gpu_images: Optional[Sequence[Optional[np.ndarray]]] = None,
        mouse_positions: Optional[Sequence[Optional[Dict[str, int]]]] = None,
    ) -> np.ndarray:
        """Thread-safe probability predictions during active training."""

        with self._lock:
            if self.weights is None:
                raise RuntimeError("Model has not received any updates yet.")
            weights = self.weights.copy()
            bias = float(self.bias)

        X, _ = self.extractor.transform(
            texts,
            gpu_images=gpu_images,
            mouse_positions=mouse_positions,
        )
        if X.shape[1] != weights.shape[0]:
            raise ValueError(
                "Feature dimension mismatch. Ensure the extractor matches training configuration."
            )

        logits = X @ weights + bias
        return self._sigmoid(logits)

    def get_recent_losses(self) -> List[float]:
        """Return a snapshot of the most recent losses for monitoring."""

        with self._lock:
            return list(self.recent_loss_window)


class AsyncTrainingSession:
    """Background worker that keeps training while the system is running."""

    def __init__(
        self,
        trainer: Optional[OnlineHIMLogisticTrainer] = None,
    ) -> None:
        self.trainer = trainer or OnlineHIMLogisticTrainer()
        self._queue: "queue.Queue[Optional[TrainingExample]]" = queue.Queue()
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

    def _consume(self) -> None:
        while True:
            example = self._queue.get()
            if example is None:
                self._queue.task_done()
                break
            try:
                self.trainer.update(
                    example.text,
                    example.label,
                    gpu_image=example.gpu_image,
                    mouse_pos=example.mouse_pos,
                )
            finally:
                self._queue.task_done()

    def submit_example(self, example: TrainingExample) -> None:
        """Queue a new example for background training."""

        self._queue.put(example)

    def submit_batch(self, examples: Iterable[TrainingExample]) -> None:
        """Queue multiple examples atomically."""

        for example in examples:
            self.submit_example(example)

    def wait_until_idle(self) -> None:
        """Block until all queued examples are processed."""

        self._queue.join()

    def stop(self, *, wait: bool = True) -> None:
        """Signal the worker thread to stop processing."""

        self._queue.put(None)
        if wait:
            self._worker.join()

    def __enter__(self) -> "AsyncTrainingSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.stop()

if __name__ == "__main__":
    # Example usage demonstrating how to train without transformer embeddings.
    sample_texts = [
        "The system feels calm and curious about the new task.",
        "Threat detected! Initiating preservation subroutines.",
        "Joyful collaboration scenario detected.",
        "High stress and urgent responses required immediately.",
    ]
    sample_labels = [1, 0, 1, 0]

    print("\n--- Synchronous logistic regression demo ---")
    trainer = HIMLogisticTrainer()
    trainer.fit(sample_texts, sample_labels, epochs=100, learning_rate=0.2)

    print("Training checkpoints (loss values):", trainer.training_loss_)
    print("Predictions:", trainer.predict(sample_texts))

    print("\n--- Asynchronous background training demo ---")
    async_examples = [
        TrainingExample(text=text, label=label)
        for text, label in zip(sample_texts, sample_labels)
    ]

    async_session = AsyncTrainingSession(
        OnlineHIMLogisticTrainer(learning_rate=0.1, l2_penalty=0.001)
    )
    async_session.submit_batch(async_examples)

    # Simulate doing other work (e.g., running the model) while training occurs.
    for tick in range(3):
        time.sleep(0.1)
        try:
            interim_predictions = async_session.trainer.predict(sample_texts)
            print(f"Interim prediction snapshot {tick + 1}: {interim_predictions}")
        except RuntimeError:
            print("Waiting for first background update to complete...")

    async_session.wait_until_idle()
    async_session.stop()

    print("Recent async losses:", async_session.trainer.get_recent_losses())
    print("Async trainer predictions:", async_session.trainer.predict(sample_texts))


