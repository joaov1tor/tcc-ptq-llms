#!/usr/bin/env python3
"""Cross-GPU Mann-Whitney U with Holm-Bonferroni correction.

Same code (João's pipeline), two GPUs: compares TTFT and tokens/s pairwise
between a run on the reference DGX (GB10) and a replica on the Colab A100.
Writes a Markdown report next to the other Stage 6 outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.report import model_label, PRECISION_LABEL  # noqa: E402
from bench.stats import cliff_delta, holm_bonferroni, mann_whitney_u, rank_r_effect  # noqa: E402
from bench.stats_adapter import load_run_dir  # noqa: E402

METRICS = (("ttft_ms", "TTFT (ms)"), ("tokens_per_s", "Tokens/s"))
_PRECISION_ORDER = {"fp16": 0, "int8": 1, "int4-awq": 2, "int4-gptq": 3}


def _samples(cell, metric: str) -> list[float]:
    return cell.ttft_ms_samples if metric == "ttft_ms" else cell.tokens_per_s_samples


def _fmt_pt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<0,0001"
    return f"{p:.4g}".replace(".", ",")


def _key(cell) -> tuple[str, str]:
    return (cell.model_id, cell.precision)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-GPU MWU + Holm for the same pipeline")
    ap.add_argument("--left-dir", type=Path, required=True, help="JSONs do grupo A (ex.: DGX)")
    ap.add_argument("--right-dir", type=Path, required=True, help="JSONs do grupo B (ex.: A100)")
    ap.add_argument("--left-label", default="DGX")
    ap.add_argument("--right-label", default="A100")
    ap.add_argument("--source", choices=("auto", "ours", "drive"), default="drive")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    left_cells, _ = load_run_dir(args.left_dir, source=args.source)
    right_cells, _ = load_run_dir(args.right_dir, source=args.source)
    left = {_key(c): c for c in left_cells}
    right = {_key(c): c for c in right_cells}
    shared = sorted(set(left) & set(right), key=lambda k: (k[0], _PRECISION_ORDER.get(k[1], 99)))
    if not shared:
        raise SystemExit("nenhuma célula compartilhada entre os dois diretórios")

    # Collect raw results, then Holm across the whole set (16 comparisons for 2 models × 4 precisões × 2 métricas).
    raw: list[dict] = []
    for metric_key, _metric_label in METRICS:
        for key in shared:
            a = _samples(left[key], metric_key)
            b = _samples(right[key], metric_key)
            if len(a) < 5 or len(b) < 5:
                raw.append({"key": key, "metric": metric_key, "skipped": "n<5"})
                continue
            mwu = mann_whitney_u(a, b)
            cliff = cliff_delta(a, b)
            r = rank_r_effect(mwu["Z"], mwu["n"])
            raw.append({
                "key": key, "metric": metric_key,
                "mwu": mwu, "cliff": cliff, "r": r,
                "median_left": sorted(a)[len(a) // 2],
                "median_right": sorted(b)[len(b) // 2],
            })

    testable = [r for r in raw if "mwu" in r]
    holm_adj = holm_bonferroni([r["mwu"]["p"] for r in testable])
    for r, adj in zip(testable, holm_adj):
        r["p_holm"] = adj

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Comparação {args.left_label} × {args.right_label} — mesmo pipeline, duas GPUs")
    lines.append("")
    lines.append(f"- Esquerda ({args.left_label}): `{args.left_dir}`")
    lines.append(f"- Direita ({args.right_label}): `{args.right_dir}`")
    lines.append(f"- Teste: Mann-Whitney U bilateral, α = {args.alpha}, correção Holm-Bonferroni sobre o conjunto.")
    lines.append(f"- Comparações: {len(testable)} (dois modelos × {len({k[1] for k in shared})} precisões × 2 métricas).")
    lines.append("")

    n_reject = sum(1 for r in testable if r["p_holm"] < args.alpha)
    lines.append(f"**Resumo:** Holm rejeita H0 em {n_reject}/{len(testable)} comparações (α = {args.alpha:g}).")
    lines.append("")

    for metric_key, metric_label in METRICS:
        lines.append(f"## {metric_label}")
        lines.append("")
        lines.append(
            f"| Modelo | Precisão | Mediana {args.left_label} | Mediana {args.right_label} "
            f"| U | Z | p bruto | p (Holm) | r = \\|Z\\|/√N | Cliff's δ | efeito |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in raw:
            if r["metric"] != metric_key:
                continue
            model_id, prec = r["key"]
            prec_lbl = PRECISION_LABEL.get(prec, prec.upper())
            if "mwu" not in r:
                lines.append(f"| {model_label(model_id)} | {prec_lbl} | — | — | — | — | — | — | — | — | {r['skipped']} |")
                continue
            verdict = "rejeita H0" if r["p_holm"] < args.alpha else "não rejeita H0"
            lines.append(
                f"| {model_label(model_id)} | {prec_lbl} "
                f"| {_fmt_pt(r['median_left'], 2)} | {_fmt_pt(r['median_right'], 2)} "
                f"| {_fmt_pt(r['mwu']['U'], 1)} | {_fmt_pt(r['mwu']['Z'], 3)} "
                f"| {_fmt_p(r['mwu']['p'])} | {_fmt_p(r['p_holm'])} "
                f"| {_fmt_pt(r['r'], 3)} | {_fmt_pt(r['cliff']['delta'], 3)} ({r['cliff']['interp']}) | {verdict} |"
            )
        lines.append("")

    (out_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}/stats.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
