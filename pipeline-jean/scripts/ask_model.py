#!/usr/bin/env python3
"""Qualitative smoke: ask a ready checkpoint one question. Not official PPL."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppl.evaluate import DEFAULT_MODEL  # noqa: E402
from ppl.generate import generate_once, load_for_generate  # noqa: E402
from ppl.logging_utils import setup_logger, slug_model_id  # noqa: E402

QUIT_TOKENS = {"/quit", "quit", "exit", ":q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy one-shot generation smoke")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.prompt and not args.interactive:
        print("passe --prompt ou --interactive", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log_dir / f"ask_{stamp}_{slug_model_id(args.model)}.log"
    logger = setup_logger(log_path, verbose=args.verbose)
    logger.info("loading model=%s (generation smoke, not official PPL)", args.model)
    model, tokenizer, device = load_for_generate(args.model, args.device, args.dtype)
    logger.info("ready device=%s", device)

    def run_one(prompt: str) -> None:
        logger.info("prompt_chars=%s", len(prompt))
        answer = generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        logger.info("answer_chars=%s", len(answer))
        print(answer)
        logger.info("answer=%s", answer)

    if args.prompt:
        run_one(args.prompt)
    if args.interactive:
        print("digite /quit para sair")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in QUIT_TOKENS:
                break
            if line:
                run_one(line)
    print(f"log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
