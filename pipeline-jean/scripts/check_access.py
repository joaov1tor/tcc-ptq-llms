#!/usr/bin/env python3
"""CLI: check Hugging Face access to the benchmark models before downloading anything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.access import check_model_access, format_result  # noqa: E402

DEFAULT_MODELS = "meta-llama/Meta-Llama-3-8B,Qwen/Qwen3-8B-Base"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Hub access (Llama 3 is gated)")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated model ids")
    args = parser.parse_args()
    results = [check_model_access(m.strip()) for m in args.models.split(",") if m.strip()]
    for result in results:
        print(format_result(result))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())