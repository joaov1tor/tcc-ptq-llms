#!/usr/bin/env python3
"""CLI: FP16 × INT8 benchmark (Etapas 3, 4 e 5) and the summary table.

For each model and precision: load → functional smoke → TTFT/tokens/s → disk and
hashes → WikiText-2 PPL. One JSON per cell, `env.json`, `run.log` and
`summary.md`/`summary.csv` in the run directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.access import check_model_access, format_result  # noqa: E402
from bench.cost import (  # noqa: E402
    MemoryTracker,
    disk_size_gib,
    gpu_thermal_snapshot,
    measure_generation,
    sha256_files,
)
from bench.env import collect_env  # noqa: E402
from bench.report import write_summary  # noqa: E402
from ppl.evaluate import evaluate_gptq_audit, evaluate_perplexity  # noqa: E402
from ppl.generate import generate_once  # noqa: E402
from ppl.loading import (  # noqa: E402
    PRECISIONS,
    get_precision_spec,
    load_model,
    load_tokenizer,
    resolve_device,
    save_quantized,
)
from ppl.logging_utils import setup_logger, slug_model_id  # noqa: E402

PROMPTS_FILE = ROOT / "prompts" / "generation_v1.json"
SMOKE_PROMPT = "Explain in one sentence what perplexity measures in a language model."

# smoke: Mac, minutes. colab: CUDA path on a small model. full: reference bench (DGX Spark).
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": dict(
        models=["Qwen/Qwen2.5-0.5B"],
        precisions=["fp16", "int8"],  # INT4 would need CUDA; smoke validates the pipeline
        max_docs=2,
        max_length=128,
        n_prompts=1,
        max_new_tokens=16,
        warmup=1,
        reps=1,
        gptq_max_blocks=2,
    ),
    "colab": dict(
        models=["Qwen/Qwen2.5-1.5B"],
        precisions=["fp16", "int8", "int4-awq", "int4-gptq"],
        max_docs=8,
        max_length=2048,
        n_prompts=5,
        max_new_tokens=64,
        warmup=2,
        reps=5,
        gptq_max_blocks=None,
    ),
    "full": dict(
        models=["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen3-8B-Base"],
        precisions=["fp16", "int8", "int4-awq", "int4-gptq"],
        max_docs=None,
        max_length=2048,
        n_prompts=10,
        max_new_tokens=128,
        warmup=3,
        reps=10,
        gptq_max_blocks=None,
    ),
}

DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "tokenizer*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP16 × INT8 benchmark (PPL, disk, memory, TTFT, tokens/s)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--models", default=None, help="comma-separated; default comes from the profile")
    parser.add_argument("--precisions", default=None, help="comma-separated; default comes from the profile")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-dtype", default="float16", help="compute dtype of the FP16 baseline")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--all-docs", action="store_true", help="score the full test split")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-samples", type=int, default=128, help="C4 docs used for AWQ/GPTQ calibration")
    parser.add_argument("--calib-seed", type=int, default=0, help="seed for the C4 calibration shuffle")
    parser.add_argument("--requantize", action="store_true", help="re-run AWQ/GPTQ even if the artifact already exists")
    parser.add_argument("--no-audit-gptq", action="store_true", help="skip the GPTQ/AWQ-style PPL audit")
    parser.add_argument("--skip-access-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(PROFILES[args.profile])
    if args.models:
        config["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    for key in ("max_docs", "max_length", "max_new_tokens", "warmup", "reps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.all_docs:
        config["max_docs"] = None
    if args.precisions is not None:
        config["precisions"] = [p.strip() for p in args.precisions.split(",") if p.strip()]
    unknown = set(config["precisions"]) - set(PRECISIONS)
    if unknown:
        raise SystemExit(f"unknown precision(s): {sorted(unknown)}; use {PRECISIONS}")
    return config


def load_prompts(n_prompts: int) -> tuple[list[str], str]:
    raw = PROMPTS_FILE.read_bytes()
    prompts = json.loads(raw)["prompts"][:n_prompts]
    return prompts, hashlib.sha256(raw).hexdigest()


def download_snapshot(model_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=DOWNLOAD_PATTERNS,
        ignore_patterns=["original/*"],  # Llama 3 also ships the raw Meta checkpoint
    )
    return Path(path)


def free_memory(device: str) -> None:
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def run_cell(
    *,
    model_id: str,
    revision: str | None,
    snapshot: Path,
    precision: str,
    device: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    prompts: list[str],
    prompts_sha: str,
    logger,
) -> dict[str, Any]:
    import torch

    spec = get_precision_spec(precision)
    cell: dict[str, Any] = {
        "model_id": model_id,
        "model_revision": revision,
        "precision": precision,
        "profile": args.profile,
        "device": device,
        "quant_backend": spec.backend,
        "prompts_file": str(PROMPTS_FILE.relative_to(ROOT)),
        "prompts_sha256": prompts_sha,
        "status": "running",
    }
    if spec.requires_cuda and device != "cuda":
        cell["status"] = "skipped"
        cell["error"] = f"{precision} requires CUDA (>=8.0); device={device}"
        logger.info("skipping %s %s: %s", model_id, precision, cell["error"])
        return cell

    started = time.perf_counter()
    memory = MemoryTracker(device)
    cell["memory_kind"] = memory.kind
    cell["thermal_before"] = gpu_thermal_snapshot()

    # Pre-quantize AWQ/GPTQ with the calibration lock (C4, 128 docs, concat 2048).
    load_path = str(snapshot)
    if spec.quantizer is not None:
        quant_dir = args.artifacts_dir / f"{slug_model_id(model_id)}-{precision}"
        if args.requantize or not (quant_dir / "config.json").exists():
            logger.info("quantizing %s -> %s (calib samples=%d seed=%d)",
                        model_id, quant_dir, args.calib_samples, args.calib_seed)
            spec.quantizer(
                str(snapshot),
                str(quant_dir),
                nsamples=args.calib_samples,
                seed=args.calib_seed,
                logger=logger,
            )
        else:
            logger.info("reusing quantized artifact at %s", quant_dir)
        load_path = str(quant_dir)
        cell["calibration"] = {
            "dataset": "c4-en/train/00000-of-01024",
            "nsamples": args.calib_samples,
            "seed": args.calib_seed,
            "concat_size": 2048,
            "min_chars": 8000,
        }

    # Etapa 2/4: load (INT8 is quantized at load time) and track peak memory.
    memory.reset()
    t0 = time.perf_counter()
    model = load_model(
        load_path, precision=precision, device=device, dtype=getattr(torch, args.baseline_dtype)
    )
    tokenizer = load_tokenizer(load_path)
    memory.sample()
    cell["load_s"] = round(time.perf_counter() - t0, 3)
    cell["dtype"] = str(model.dtype).replace("torch.", "")
    logger.info("loaded %s %s in %.1fs dtype=%s", model_id, precision, cell["load_s"], cell["dtype"])

    # Etapa 3: functional smoke (one greedy completion).
    smoke = generate_once(model=model, tokenizer=tokenizer, prompt=SMOKE_PROMPT, max_new_tokens=32, device=device)
    cell["smoke"] = {"prompt": SMOKE_PROMPT, "output": smoke, "ok": bool(smoke.strip())}
    logger.info("smoke ok=%s output=%r", cell["smoke"]["ok"], smoke[:200])

    # Etapa 4: TTFT and decode throughput.
    cell["generation"] = measure_generation(
        model,
        tokenizer,
        prompts,
        max_new_tokens=config["max_new_tokens"],
        warmup=config["warmup"],
        reps=config["reps"],
        device=device,
        seed=args.seed,
        memory=memory,
        logger=logger,
    )
    cell["peak_memory_gib"] = round(memory.peak_gib, 4)
    logger.info(
        "cost ttft_ms_median=%.1f tok_s_median=%.2f peak_memory_gib=%.3f (%s)",
        cell["generation"]["ttft_ms"]["median"] or float("nan"),
        cell["generation"]["tokens_per_s"]["median"] or float("nan"),
        cell["peak_memory_gib"],
        memory.kind,
    )

    # Disk + hashes: the FP16 snapshot, or the saved quantized artifact (objetivo específico 1).
    if precision == "int8":
        weights_dir = save_quantized(model, tokenizer, args.artifacts_dir / f"{slug_model_id(model_id)}-int8")
    elif spec.quantizer is not None:
        weights_dir = Path(load_path)  # already produced by the pre-quantization step above
    else:
        weights_dir = snapshot
    cell["weights_dir"] = str(weights_dir)
    cell["disk_gib"] = round(disk_size_gib(weights_dir), 4)
    cell["weights_sha256"] = sha256_files(weights_dir)
    logger.info("disk_gib=%.3f files=%s dir=%s", cell["disk_gib"], len(cell["weights_sha256"]), weights_dir)

    # Etapa 5: PPL, same protocol and cut for every precision.
    memory.reset()
    cell["ppl"] = evaluate_perplexity(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        max_docs=config["max_docs"],
        max_length=config["max_length"],
        device=device,
        logger=logger,
        verbose=args.verbose,
    )
    memory.sample()
    cell["peak_memory_ppl_gib"] = round(memory.peak_gib, 4)
    if not args.no_audit_gptq:
        cell["gptq_audit"] = evaluate_gptq_audit(
            model=model, tokenizer=tokenizer, device=device, max_blocks=config["gptq_max_blocks"], logger=logger
        )

    cell["thermal_after"] = gpu_thermal_snapshot()
    cell["duration_s"] = round(time.perf_counter() - started, 3)
    cell["status"] = "ok"
    del model
    free_memory(device)
    return cell


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or ROOT / "results" / f"{stamp}_{args.profile}"
    logger = setup_logger(run_dir / "run.log", verbose=args.verbose, name="bench")
    device = resolve_device(args.device)
    logger.info("run_dir=%s profile=%s device=%s config=%s", run_dir, args.profile, device, config)

    revisions: dict[str, str | None] = {}
    for model_id in config["models"]:
        if args.skip_access_check:
            revisions[model_id] = None
            continue
        result = check_model_access(model_id)
        logger.info("access %s", format_result(result))
        if not result.ok:
            logger.error("abortando antes de baixar: %s", format_result(result))
            return 2
        revisions[model_id] = result.sha

    env = collect_env(device)
    write_json(run_dir / "env.json", env)
    logger.info("env gpu=%s packages=%s", env.get("gpu"), env["packages"])

    prompts, prompts_sha = load_prompts(config["n_prompts"])
    logger.info("prompts=%s sha256=%s", PROMPTS_FILE.name, prompts_sha)

    for model_id in config["models"]:
        snapshot = download_snapshot(model_id, revisions[model_id])
        logger.info("snapshot %s -> %s", model_id, snapshot)
        for precision in config["precisions"]:
            out_path = run_dir / f"{slug_model_id(model_id)}_{precision}.json"
            logger.info("=== %s %s ===", model_id, precision)
            try:
                cell = run_cell(
                    model_id=model_id,
                    revision=revisions[model_id],
                    snapshot=snapshot,
                    precision=precision,
                    device=device,
                    config=config,
                    args=args,
                    prompts=prompts,
                    prompts_sha=prompts_sha,
                    logger=logger,
                )
            except Exception as exc:  # OOM etc.: recorded as not viable here (§6.2), run continues
                logger.error("cell failed %s %s: %s\n%s", model_id, precision, exc, traceback.format_exc())
                cell = {
                    "model_id": model_id,
                    "model_revision": revisions[model_id],
                    "precision": precision,
                    "profile": args.profile,
                    "device": device,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
                free_memory(device)
            write_json(out_path, cell)
            logger.info("wrote %s status=%s", out_path, cell["status"])

    summary = write_summary(run_dir)
    logger.info("summary (%s):\n%s", run_dir / "summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())