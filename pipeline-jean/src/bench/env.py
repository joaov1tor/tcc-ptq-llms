"""Environment record for each run (§6.2): hardware, drivers and library versions."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from importlib import metadata
from typing import Any

PACKAGES = ("torch", "transformers", "bitsandbytes", "datasets", "accelerate", "huggingface_hub", "safetensors")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_env(device: str) -> dict[str, Any]:
    import torch

    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or None,
        "device": device,
        "docker_image": os.environ.get("BENCH_IMAGE_TAG"),
        "packages": package_versions(),
        "torch_cuda": torch.version.cuda,
        "gpu": None,
        "gpu_memory_gib": None,
        "driver": None,
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_memory_gib"] = round(props.total_memory / 1024**3, 2)
        env["gpu_capability"] = f"{props.major}.{props.minor}"
        env["driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        env["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    elif device == "mps":
        env["gpu"] = "Apple MPS"
        mem = _run(["sysctl", "-n", "hw.memsize"])
        env["unified_memory_gib"] = round(int(mem) / 1024**3, 2) if mem else None
    return env