#!/usr/bin/env python3
"""CLI: rebuild summary.md / summary.csv from the JSONs of one or more runs.

Useful to merge cells measured in separate runs (e.g. one run per model on the DGX):
pass several --run-dir and an --out directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.report import load_cells, render_csv, render_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FP16 × INT8 summary table")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: the single --run-dir")
    args = parser.parse_args()
    cells = [cell for run_dir in args.run_dir for cell in load_cells(run_dir)]
    if not cells:
        print("nenhum JSON de célula encontrado", file=sys.stderr)
        return 1
    out = args.out or args.run_dir[0]
    out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(cells)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    (out / "summary.csv").write_text(render_csv(cells), encoding="utf-8")
    print(markdown)
    print(f"summary={out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())