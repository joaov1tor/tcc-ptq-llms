#!/usr/bin/env python3
"""CLI: WikiText-2 perplexity on a ready Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppl.evaluate import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_PATH,
    DEFAULT_MAX_DOCS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_SPLIT,
    evaluate_gptq_audit,
    evaluate_perplexity,
    resolve_model_revision,
)
from ppl.loading import load_model_and_tokenizer, resolve_device, resolve_dtype  # noqa: E402
from ppl.logging_utils import setup_logger, slug_model_id  # noqa: E402
from ppl.protocol import GPTQ_PROTOCOL_NAME, PROTOCOL_NAME  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harness-faithful WikiText-2 perplexity")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="model commit SHA (default: current main)")
    parser.add_argument("--dataset-revision", default=None, help="dataset commit SHA")
    parser.add_argument(
        "--protocol",
        choices=[PROTOCOL_NAME, GPTQ_PROTOCOL_NAME],
        default=PROTOCOL_NAME,
        help=f"{GPTQ_PROTOCOL_NAME} is an audit against GPTQ/AWQ tables, not the official metric",
    )
    parser.add_argument("--max-blocks", type=int, default=None, help=f"{GPTQ_PROTOCOL_NAME} only")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None, help="float32, float16, bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help="Ignore --max-docs and score the full split",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_docs = None if args.all_docs else args.max_docs
    output = args.output
    if output is None:
        suffix = "dryrun" if args.dry_run else "ppl"
        output = ROOT / "results" / f"{suffix}.json"
    if args.protocol == GPTQ_PROTOCOL_NAME:
        return run_gptq_audit(args, output)
    payload = evaluate_perplexity(
        model_id=args.model,
        model_revision=args.revision,
        dataset_path=args.dataset,
        dataset_revision=args.dataset_revision,
        dataset_config=args.config,
        split=args.split,
        max_docs=max_docs,
        max_length=args.max_length,
        device=args.device,
        dtype_name=args.dtype,
        dry_run=args.dry_run,
        output_path=output,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )
    print(json.dumps({k: payload[k] for k in (
        "protocol",
        "dry_run",
        "n_docs",
        "n_docs_skipped_empty",
        "n_pages_in_split",
        "n_words",
        "n_bytes",
        "word_perplexity",
        "byte_perplexity",
        "bits_per_byte",
        "token_perplexity",
        "device",
        "duration_s",
        "log_path",
    ) if k in payload}, indent=2))
    print(f"json={output}")
    return 0


def run_gptq_audit(args: argparse.Namespace, output: Path) -> int:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = setup_logger(args.log_dir / f"ppl_gptq_{stamp}_{slug_model_id(args.model)}.log", args.verbose)
    device = resolve_device(args.device)
    revision = resolve_model_revision(args.model, args.revision)
    model, tokenizer = load_model_and_tokenizer(args.model, device, resolve_dtype(device, args.dtype), revision)
    payload = evaluate_gptq_audit(
        model=model, tokenizer=tokenizer, device=device, max_blocks=args.max_blocks, logger=logger
    )
    payload.update(model_id=args.model, model_revision=revision, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
