"""Where a post-processing step runs: on the GPU only when it has room.

Forge keeps the checkpoint in VRAM after generating, and with a large model
(or in low-VRAM mode) it fills nearly all of it. On Windows the NVIDIA driver
then quietly moves any further GPU allocation into system RAM ("sysmem
fallback"), which makes it hundreds of times slower: post-processing looks
stuck at 99-100 % for ever. So before a step goes to the GPU we ask how much
VRAM is really free. If it is not enough, Forge is asked to unload models to
make room, as it does itself before the VAE decode. If that is still not
enough, the step runs on the CPU: slower, but it finishes.
"""

from __future__ import annotations

import importlib
import time

import torch

MB = 1 << 20
MARGIN = 768 * MB          # headroom for the allocator, cuDNN workspaces, fragmentation


def _memory_manager():
    """Forge's model manager (newer and older layouts), or None elsewhere."""
    for name in ("backend.memory_management", "ldm_patched.modules.model_management"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def free_vram(device):
    """VRAM the next allocation can really use: free on the card, plus what
    PyTorch has reserved but is not using."""
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free + max(0, cached)


def make_room(need, device):
    mm = _memory_manager()
    if mm is not None and hasattr(mm, "free_memory"):
        try:
            mm.free_memory(need, device)
        except TypeError:
            mm.free_memory(need, device, [])
        except Exception as exc:  # noqa: BLE001 - a failed unload only means less room
            print(f"[post-processing] Could not ask Forge to free VRAM: {exc}")
    torch.cuda.empty_cache()


def image_need(image, buffers):
    """Bytes for `buffers` float32 RGB copies of a PIL image."""
    w, h = image.size
    return w * h * 3 * 4 * buffers


def pick(device, need, label):
    """`device` if the GPU has `need` bytes (after asking Forge to make room),
    else the CPU."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return device
    try:
        if free_vram(device) >= need + MARGIN:
            return device
        make_room(need + MARGIN, device)
        free = free_vram(device)
        if free >= need + MARGIN:
            print(f"[{label}] Made room on the GPU: {free / MB:.0f} MB free, about {need / MB:.0f} MB needed.")
            return device
        print(f"[{label}] Only {free / MB:.0f} MB of VRAM free, about {need / MB:.0f} MB needed: running on "
              f"the CPU so the GPU does not spill into system memory (slower, but it finishes).")
        return torch.device("cpu")
    except Exception as exc:  # noqa: BLE001 - an unknown backend keeps the old behaviour
        print(f"[{label}] Could not read free VRAM ({exc}); using {device}.")
        return device


def ram_available():
    """Free system RAM in bytes, or None if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


class Timer:
    """`with Timer(label, device):` prints how long a step took and where."""

    def __init__(self, label, device):
        self.label, self.device = label, device

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            print(f"[{self.label}] Done in {time.perf_counter() - self.t:.1f} s on {self.device}.")
        return False
