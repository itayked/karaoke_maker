"""GPU detection and Whisper model-size auto-selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    cuda_available: bool
    name: str | None = None
    vram_mb: int | None = None


def probe_gpu() -> GpuInfo:
    """Returns CUDA availability + device-0 name + VRAM in MB.

    Imports torch lazily so health responses still work if torch fails to load
    (e.g. CUDA driver mismatch — useful info to surface to the user).
    """
    try:
        import torch
    except Exception:
        return GpuInfo(cuda_available=False)

    if not torch.cuda.is_available():
        return GpuInfo(cuda_available=False)

    try:
        props = torch.cuda.get_device_properties(0)
        return GpuInfo(
            cuda_available=True,
            name=props.name,
            vram_mb=int(props.total_memory // (1024 * 1024)),
        )
    except Exception:
        return GpuInfo(cuda_available=True)


def select_model_size(gpu: GpuInfo) -> str:
    """Auto-pick Whisper model size based on VRAM.

    Thresholds match the desktop-app brief: <4GB → small, 4–6GB → medium,
    >=6GB → large-v3. CPU-only falls back to small.
    """
    if not gpu.cuda_available or gpu.vram_mb is None:
        return "small"
    if gpu.vram_mb >= 6000:
        return "large-v3"
    if gpu.vram_mb >= 4000:
        return "medium"
    return "small"
