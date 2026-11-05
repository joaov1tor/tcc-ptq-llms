"""Cost metrics (Etapa 4): disk, peak memory, TTFT and decode throughput."""

from __future__ import annotations

import hashlib
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GIB = 1024**3
WEIGHT_GLOB = "*.safetensors"


# --- disk and hashes -------------------------------------------------------


def weight_files(path: Path) -> list[Path]:
    return sorted(path.glob(WEIGHT_GLOB))


def disk_size_gib(path: Path) -> float:
    # HF cache snapshots are symlinks to blobs; stat() follows them.
    return sum(f.stat().st_size for f in weight_files(path)) / GIB


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(path: Path) -> dict[str, str]:
    return {f.name: sha256_file(f) for f in weight_files(path)}


# --- memory -----------------------------------------------------------------


class MemoryTracker:
    """Peak memory for one cell (load + generation).

    CUDA: `max_memory_allocated` (what PyTorch allocated on the GPU — the "Pico de
    memória" of the table). MPS has no peak counter, so `current_allocated_memory`
    (tensors only; `driver_allocated_memory` also counts the allocator pool and hides
    the INT8 saving) is sampled at every `sample()` and misses transient peaks in
    between. CPU falls back to the process max RSS, which cannot be reset between
    cells and is only indicative.
    """

    def __init__(self, device: str):
        self.device = device
        self.peak_bytes = 0
        self.kind = {
            "cuda": "cuda_max_memory_allocated",
            "mps": "mps_current_allocated_sampled",
        }.get(device, "process_max_rss")

    def reset(self) -> None:
        import torch

        self.peak_bytes = 0
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def sample(self) -> None:
        import torch

        if self.device == "cuda":
            current = torch.cuda.max_memory_allocated()
        elif self.device == "mps":
            current = torch.mps.current_allocated_memory()
        else:
            current = max_rss_bytes()
        self.peak_bytes = max(self.peak_bytes, current)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / GIB


def max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return rss if sys.platform == "darwin" else rss * 1024


def synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --- generation timing ------------------------------------------------------


def tokens_per_second(n_new_tokens: int, t_first: float, t_end: float) -> float:
    """Decode throughput: tokens after the first one over the time after the first one."""
    decode_time = t_end - t_first
    if n_new_tokens <= 1 or decode_time <= 0:
        return float("nan")
    return (n_new_tokens - 1) / decode_time


def summarize(samples: list[float]) -> dict[str, Any]:
    """Median and IQR (§6.6). Raw samples are kept for the non-parametric tests."""
    clean = [s for s in samples if s == s]  # drop NaN
    if not clean:
        return {"median": None, "q1": None, "q3": None, "iqr": None, "n": 0, "samples": samples}
    if len(clean) == 1:
        q1 = q3 = clean[0]
    else:
        q1, _, q3 = statistics.quantiles(clean, n=4, method="inclusive")
    return {
        "median": statistics.median(clean),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "n": len(clean),
        "samples": samples,
    }


class _FirstTokenTimer:
    """Streamer hook: generate() calls put() once with the prompt, then once per new token.

    Each put() receives a tensor already copied to CPU, so the GPU is synchronized
    by the time the second call records the first-token timestamp.
    """

    def __init__(self):
        self.calls = 0
        self.t_first: float | None = None

    def put(self, value) -> None:
        self.calls += 1
        if self.calls == 2:
            self.t_first = time.perf_counter()

    def end(self) -> None:
        pass


def time_one_generation(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> dict[str, float]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    timer = _FirstTokenTimer()
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # fixed length so every sample decodes the same amount
            do_sample=False,
            pad_token_id=pad_id,
            streamer=timer,
        )
    synchronize(device)
    t_end = time.perf_counter()
    n_new = output.shape[1] - encoded["input_ids"].shape[1]
    t_first = timer.t_first if timer.t_first is not None else t_end
    return {
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "new_tokens": int(n_new),
        "ttft_ms": (t_first - t0) * 1000.0,
        "tokens_per_s": tokens_per_second(n_new, t_first, t_end),
        "total_s": t_end - t0,
    }


def measure_generation(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    warmup: int,
    reps: int,
    device: str,
    seed: int,
    memory: MemoryTracker | None = None,
    logger=None,
) -> dict[str, Any]:
    """Greedy decoding, batch 1. `warmup` runs are discarded; then `reps` passes over all prompts."""
    import torch

    torch.manual_seed(seed)
    for i in range(warmup):
        time_one_generation(model, tokenizer, prompts[i % len(prompts)], max_new_tokens, device)
    runs: list[dict[str, Any]] = []
    for rep in range(reps):
        for idx, prompt in enumerate(prompts):
            sample = time_one_generation(model, tokenizer, prompt, max_new_tokens, device)
            sample.update(rep=rep, prompt_index=idx)
            runs.append(sample)
            if memory is not None:
                memory.sample()
            if logger is not None:
                logger.info(
                    "gen rep=%s prompt=%s prompt_tokens=%s new_tokens=%s ttft_ms=%.1f tok_s=%.2f",
                    rep,
                    idx,
                    sample["prompt_tokens"],
                    sample["new_tokens"],
                    sample["ttft_ms"],
                    sample["tokens_per_s"],
                )
    return {
        "max_new_tokens": max_new_tokens,
        "warmup_discarded": warmup,
        "reps": reps,
        "n_prompts": len(prompts),
        "batch_size": 1,
        "decoding": "greedy",
        "seed": seed,
        "ttft_ms": summarize([r["ttft_ms"] for r in runs]),
        "tokens_per_s": summarize([r["tokens_per_s"] for r in runs]),
        "runs": runs,
    }


# --- thermal (§6.5) ---------------------------------------------------------


def gpu_thermal_snapshot() -> dict[str, str] | None:
    """Temperature and throttle reasons from nvidia-smi; None where it is not available."""
    query = "temperature.gpu,clocks.sm,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"query": query, "value": out}