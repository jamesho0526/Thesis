from __future__ import annotations

import os

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Select the discrete DirectML GPU, then CUDA, then CPU."""
    if prefer_gpu:
        try:
            import torch_directml

            requested_id = int(os.environ.get("DIRECTML_DEVICE_ID", "1"))
            count = torch_directml.device_count()
            if requested_id < 0 or requested_id >= count:
                raise ValueError(
                    f"DIRECTML_DEVICE_ID={requested_id} is invalid; available IDs: 0-{count - 1}"
                )
            return torch_directml.device(requested_id)
        except (ImportError, RuntimeError):
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if str(device).startswith("privateuseone"):
        try:
            import torch_directml

            device_id = int(str(device).rsplit(":", 1)[1])
            return f"DirectML GPU {device_id}: {torch_directml.device_name(device_id).strip()}"
        except (ImportError, ValueError, RuntimeError):
            return "DirectML GPU"
    if device.type == "cuda":
        return f"CUDA GPU ({torch.cuda.get_device_name(device)})"
    return "CPU"
