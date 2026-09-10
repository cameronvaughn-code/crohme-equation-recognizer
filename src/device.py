"""
device.py

Single place that decides what hardware to run on, so every script
(train / evaluate / predict / visualize) picks the same accelerator.

Priority: CUDA (Linux/Colab GPU) -> MPS (Apple Silicon) -> CPU.
"""

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
