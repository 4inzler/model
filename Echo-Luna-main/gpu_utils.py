from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass(frozen=True)
class AccelerationState:
    """Describe the active numerical backend."""

    backend: str
    device: str
    using_gpu: bool


def _force_cpu() -> bool:
    """Check whether GPU acceleration is explicitly disabled."""

    return os.getenv("ECHO_LUNA_FORCE_CPU", "0") == "1"


@lru_cache(maxsize=4)
def _detect_acceleration(force_cpu: bool, prefer_gpu: bool) -> AccelerationState:
    """Detect the best available acceleration backend."""

    if force_cpu or torch is None:
        return AccelerationState(backend="numpy", device="cpu", using_gpu=False)

    if prefer_gpu:
        if torch.cuda.is_available():
            return AccelerationState(backend="torch", device="cuda", using_gpu=True)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return AccelerationState(backend="torch", device="mps", using_gpu=True)

    return AccelerationState(backend="torch", device="cpu", using_gpu=False)


def get_acceleration_state(prefer_gpu: bool = True) -> AccelerationState:
    """Return the cached acceleration state."""

    return _detect_acceleration(_force_cpu(), prefer_gpu)


def torch_available() -> bool:
    """Return True when PyTorch can be imported."""

    return torch is not None


def require_torch():  # type: ignore[return-value]
    """Return the torch module or raise a descriptive error."""

    if torch is None:
        raise RuntimeError("PyTorch is not available. Install torch to enable GPU acceleration.")
    return torch


def compute_image_stats(image: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation, preferring GPU acceleration."""

    state = get_acceleration_state()
    if state.backend == "torch":
        lib = require_torch()
        tensor = lib.as_tensor(image, dtype=lib.float32, device=state.device)
        mean = lib.mean(tensor)
        std = lib.std(tensor, unbiased=False)
        return float(mean.detach().cpu()), float(std.detach().cpu())

    return float(np.mean(image)), float(np.std(image))


def combine_weighted_arrays(arrays: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Combine arrays using the provided weights, with optional GPU support."""

    if not arrays:
        raise ValueError("No arrays provided for combination.")
    if len(arrays) != len(weights):
        raise ValueError("Number of arrays must match number of weights.")

    sanitized = [np.asarray(array, dtype=np.float32) for array in arrays]
    weights_array = np.asarray(weights, dtype=np.float32)
    total_weight = float(np.sum(weights_array))
    if total_weight == 0:
        raise ValueError("Sum of weights must be non-zero.")

    state = get_acceleration_state()
    if state.backend == "torch":
        lib = require_torch()
        stack = lib.stack([lib.as_tensor(array, dtype=lib.float32, device=state.device) for array in sanitized])
        weight_tensor = lib.as_tensor(weights_array, dtype=lib.float32, device=state.device)
        expand_shape = (1,) * (stack.ndim - 1)
        weight_tensor = weight_tensor.view(-1, *expand_shape)
        combined = lib.sum(stack * weight_tensor, dim=0) / total_weight
        return combined.detach().cpu().numpy().astype(np.float32)

    stack = np.stack(sanitized, axis=0)
    weight_tensor = weights_array.reshape((weights_array.shape[0],) + (1,) * (stack.ndim - 1))
    combined = np.sum(stack * weight_tensor, axis=0) / total_weight
    return combined.astype(np.float32)


def create_zipline_pattern(vector_data: np.ndarray) -> np.ndarray:
    """Create the zipline pattern using vectorised math, optionally on GPU."""

    array = np.asarray(vector_data, dtype=np.float32)
    if array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim == 1:
        array = array.reshape(1, -1)

    height, width = array.shape
    state = get_acceleration_state()

    if state.backend == "torch":
        lib = require_torch()
        tensor = lib.as_tensor(array, dtype=lib.float32, device=state.device)
        max_val = lib.max(lib.abs(tensor)) if tensor.numel() else lib.tensor(0.0, device=state.device)
        if float(max_val) > 1.0:
            tensor = lib.where(max_val > 0, tensor / max_val, tensor)
        tensor = lib.clamp(tensor, 0.0, 1.0)

        expanded = tensor.repeat_interleave(4, dim=0).repeat_interleave(8, dim=1)

        col_indices = lib.arange(width, device=state.device).repeat_interleave(8).unsqueeze(0)
        light_base = 0.3 + expanded * 0.4
        dark_base = 0.1 + expanded * 0.2
        base_pattern = lib.where(col_indices % 2 == 0, light_base, dark_base)

        row_mod = lib.arange(height * 4, device=state.device).unsqueeze(1) % 4
        col_mod = lib.arange(width * 8, device=state.device).unsqueeze(0) % 4

        dark_mask = ((col_mod == 0) | (col_mod == 3)) & ((row_mod == 0) | (row_mod == 3))
        light_mask = ((col_mod == 1) | (col_mod == 2)) & ((row_mod == 1) | (row_mod == 2))

        pattern = lib.where(dark_mask, base_pattern * 0.7, base_pattern)
        pattern = lib.where(light_mask, pattern * 1.3, pattern)
        pattern = lib.clamp(pattern, 0.0, 1.0)
        return pattern.detach().cpu().numpy().astype(np.float32)

    max_val = float(np.max(np.abs(array))) if array.size else 0.0
    if max_val > 1.0:
        array = np.where(max_val > 0, array / max_val, array)
    array = np.clip(array, 0.0, 1.0)

    expanded = np.repeat(np.repeat(array, 4, axis=0), 8, axis=1)

    col_indices = np.repeat(np.arange(width, dtype=np.int32), 8)
    light_base = 0.3 + expanded * 0.4
    dark_base = 0.1 + expanded * 0.2
    base_pattern = np.where(col_indices[np.newaxis, :] % 2 == 0, light_base, dark_base)

    row_mod = (np.arange(height * 4, dtype=np.int32)[:, None]) % 4
    col_mod = (np.arange(width * 8, dtype=np.int32)[None, :]) % 4

    dark_mask = ((col_mod == 0) | (col_mod == 3)) & ((row_mod == 0) | (row_mod == 3))
    light_mask = ((col_mod == 1) | (col_mod == 2)) & ((row_mod == 1) | (row_mod == 2))

    pattern = np.where(dark_mask, base_pattern * 0.7, base_pattern)
    pattern = np.where(light_mask, pattern * 1.3, pattern)
    return np.clip(pattern, 0.0, 1.0).astype(np.float32)
