#!/usr/bin/env python3
"""Stage 6 of the TCC: median + IQR, Kruskal-Wallis, Mann-Whitney with Holm-Bonferroni,
Cliff's δ, r = |Z|/√N and Pareto-front for the trade-off matrix.

Runs locally (stdlib only, no scipy/numpy). Point it at a run dir with either
our pipeline's output or the Drive dump; detection is automatic.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.cost import summarize  # noqa: E402
from bench.report import model_label, relative_change_pct, within_threshold, PRECISION_LABEL  # noqa: E402
from bench.stats import (  # noqa: E402
    cliff_delta,
    holm_bonferroni,
    kruskal_wallis,
    mann_whitney_u,
    pareto_front,
    rank_r_effect,
)
from bench.stats_adapter import StatsCell, load_run_dir  # noqa: E402

BASELINE = "fp16"
METRICS_WITH_SAMPLES = ("ttft_ms", "tokens_per_s")
MIN_N_FOR_MWU = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 statistical analysis of the PTQ benchmark")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("auto", "ours", "drive"), default="auto")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/stats/")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _samples(cell: StatsCell, metric: str) -> list[float]:
    return cell.ttft_ms_samples if metric == "ttft_ms" else cell.tokens_per_s_samples


def _fmt_pt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def _fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:+.{decimals}f}%".replace(".", ",")


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<0,0001"
    return f"{p:.4f}".replace(".", ",")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_by_model(cells: list[StatsCell]) -> dict[str, dict[str, StatsCell]]:
    out: dict[str, dict[str, StatsCell]] = {}
    for c in cells:
        out.setdefault(c.model_id, {})[c.precision] = c
    return out


def _precision_label(precision: str) -> str:
    return PRECISION_LABEL.get(precision, precision.upper())


def build_summary_table(by_model: dict[str, dict[str, StatsCell]]) -> list[str]:
    lines = ["## Mediana e IQR por métrica, precisão e modelo", ""]
    for model_id, precisions in by_model.items():
        lines.append(f"### {model_label(model_id)}")
        lines.append("")
        lines.append("| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for metric_key, metric_label in (("ttft_ms", "TTFT (ms)"), ("tokens_per_s", "Tokens/s")):
            for prec in sorted(precisions, key=_precision_sort_key):
                cell = precisions[prec]
                samples = _samples(cell, metric_key)
                if not samples:
                    lines.append(f"| {metric_label} | {_precision_label(prec)} | — | — | — | — | 0 |")
                    continue
                stats = summarize(samples)
                lines.append(
                    f"| {metric_label} | {_precision_label(prec)} "
                    f"| {_fmt_pt(stats['median'], 2)} | {_fmt_pt(stats['q1'], 2)} "
                    f"| {_fmt_pt(stats['q3'], 2)} | {_fmt_pt(stats['iqr'], 2)} | {stats['n']} |"
                )
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            lines.append(
                f"| PPL (word) | {_precision_label(prec)} | {_fmt_pt(cell.ppl, 4)} | — | — | — | 1 |"
            )
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            lines.append(
                f"| Pico de memória (GiB) | {_precision_label(prec)} | {_fmt_pt(cell.peak_memory_gib, 2)} | — | — | — | 1 |"
            )
        lines.append("")
    return lines


_PRECISION_ORDER = {"fp16": 0, "int8": 1, "int4-awq": 2, "int4-gptq": 3}


def _precision_sort_key(prec: str) -> tuple[int, str]:
    return (_PRECISION_ORDER.get(prec, 99), prec)


def build_tests_block(by_model: dict[str, dict[str, StatsCell]], alpha: float) -> tuple[list[str], list[dict]]:
    lines = ["## Testes de hipótese", ""]
    csv_rows: list[dict] = []
    for model_id, precisions in by_model.items():
        lines.append(f"### {model_label(model_id)}")
        lines.append("")
        for metric_key, metric_label in (("ttft_ms", "TTFT (ms)"), ("tokens_per_s", "Tokens/s")):
            lines.append(f"**{metric_label}**")
            lines.append("")
            groups_for_kw: list[list[float]] = []
            group_names: list[str] = []
            for prec in sorted(precisions, key=_precision_sort_key):
                samples = _samples(precisions[prec], metric_key)
                if samples:
                    groups_for_kw.append(samples)
                    group_names.append(prec)
            if len(groups_for_kw) >= 2:
                kw = kruskal_wallis(groups_for_kw)
                verdict = "diferença significativa" if kw["p"] < alpha else "sem evidência de diferença"
                lines.append(
                    f"- Kruskal-Wallis (k={kw['k']}, df={kw['df']}, N={kw['N']}): "
                    f"H={_fmt_pt(kw['H'], 3)}, p={_fmt_p(kw['p'])} → {verdict} (α={alpha:g})."
                )
            else:
                lines.append(f"- Kruskal-Wallis: menos de 2 grupos com amostras — pulado.")
                kw = None

            # Pairwise MWU fp16 vs each quantized
            if BASELINE in precisions and _samples(precisions[BASELINE], metric_key):
                base_samples = _samples(precisions[BASELINE], metric_key)
                others = [p for p in sorted(precisions, key=_precision_sort_key) if p != BASELINE and _samples(precisions[p], metric_key)]
                raw_results = []
                for prec in others:
                    other_samples = _samples(precisions[prec], metric_key)
                    if len(base_samples) < MIN_N_FOR_MWU or len(other_samples) < MIN_N_FOR_MWU:
                        raw_results.append((prec, None, "n<5"))
                        continue
                    mwu = mann_whitney_u(base_samples, other_samples)
                    cliff = cliff_delta(base_samples, other_samples)
                    r = rank_r_effect(mwu["Z"], mwu["n"])
                    raw_results.append((prec, {"mwu": mwu, "cliff": cliff, "r": r}, None))

                raw_ps = [r[1]["mwu"]["p"] for r in raw_results if r[1] is not None]
                holm_adj = holm_bonferroni(raw_ps) if raw_ps else []
                holm_iter = iter(holm_adj)

                lines.append("")
                lines.append(f"- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α={alpha:g}):")
                lines.append("")
                lines.append("| Comparação | U | Z | p bruto | p (Holm) | r = \\|Z\\|/√N | Cliff's δ | efeito |")
                lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
                for prec, result, note in raw_results:
                    if result is None:
                        lines.append(f"| FP16 × {_precision_label(prec)} | — | — | — | — | — | — | {note} |")
                        csv_rows.append({
                            "model_id": model_id, "metric": metric_key, "precision": prec,
                            "mwu_U": "", "mwu_Z": "", "mwu_p_raw": "", "mwu_p_holm": "",
                            "r_effect": "", "cliff_delta": "", "cliff_interp": "", "note": note,
                            "kw_H": kw["H"] if kw else "", "kw_p": kw["p"] if kw else "",
                        })
                        continue
                    adj = next(holm_iter)
                    reject = adj < alpha
                    verdict = ("rejeita H0" if reject else "não rejeita H0")
                    lines.append(
                        f"| FP16 × {_precision_label(prec)} | {_fmt_pt(result['mwu']['U'], 1)} "
                        f"| {_fmt_pt(result['mwu']['Z'], 3)} | {_fmt_p(result['mwu']['p'])} "
                        f"| {_fmt_p(adj)} | {_fmt_pt(result['r'], 3)} "
                        f"| {_fmt_pt(result['cliff']['delta'], 3)} ({result['cliff']['interp']}) | {verdict} |"
                    )
                    csv_rows.append({
                        "model_id": model_id, "metric": metric_key, "precision": prec,
                        "mwu_U": result["mwu"]["U"], "mwu_Z": result["mwu"]["Z"],
                        "mwu_p_raw": result["mwu"]["p"], "mwu_p_holm": adj,
                        "r_effect": result["r"],
                        "cliff_delta": result["cliff"]["delta"], "cliff_interp": result["cliff"]["interp"],
                        "note": verdict,
                        "kw_H": kw["H"] if kw else "", "kw_p": kw["p"] if kw else "",
                    })
                lines.append("")
            else:
                lines.append("- Pairwise MWU: FP16 sem amostras disponíveis para servir de baseline.")
                lines.append("")
    return lines, csv_rows


def build_ppl_block(by_model: dict[str, dict[str, StatsCell]]) -> list[str]:
    lines = ["## Variação relativa de PPL vs FP16 (limiar de referência: 3%)", ""]
    for model_id, precisions in by_model.items():
        base = precisions.get(BASELINE)
        base_ppl = base.ppl if base else None
        lines.append(f"### {model_label(model_id)}")
        lines.append("")
        lines.append("| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |")
        lines.append("| --- | ---: | ---: | :-: |")
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            pct = relative_change_pct(base_ppl, cell.ppl)
            thr = within_threshold(pct) if pct is not None else None
            thr_cell = "—" if thr is None else ("✅" if thr else "❌")
            lines.append(
                f"| {_precision_label(prec)} | {_fmt_pt(cell.ppl, 4)} "
                f"| {_fmt_pct(pct, 1)} | {thr_cell} |"
            )
        lines.append("")
    return lines


def build_pareto_block(by_model: dict[str, dict[str, StatsCell]]) -> tuple[list[str], list[str]]:
    lines = ["## Matriz de trade-off e Pareto", ""]
    compact = ["# Trade-off (uma linha por modelo × precisão)", ""]
    compact.append("| Modelo | Precisão | PPL | Δ PPL | Mem (GiB) | Δ Mem | Tokens/s (med) | Δ Tok/s | Pareto |")
    compact.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: |")
    for model_id, precisions in by_model.items():
        base = precisions.get(BASELINE)
        base_ppl = base.ppl if base else None
        base_mem = base.peak_memory_gib if base else None
        base_tps = None
        if base and base.tokens_per_s_samples:
            base_tps = summarize(base.tokens_per_s_samples)["median"]

        items = []
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            tps_med = None
            if cell.tokens_per_s_samples:
                tps_med = summarize(cell.tokens_per_s_samples)["median"]
            if cell.ppl is None or cell.peak_memory_gib is None or tps_med is None:
                continue  # dominance only makes sense when the three axes are present
            items.append({
                "name": prec,
                "ppl": cell.ppl,
                "memory": cell.peak_memory_gib,
                "tps": tps_med,
            })
        pareto = pareto_front(items, key="name", minimize=("ppl", "memory"), maximize=("tps",))

        lines.append(f"### {model_label(model_id)}")
        lines.append("")
        lines.append("| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |")
        lines.append("| --- | ---: | ---: | ---: | :-: | --- |")
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            tps_med = summarize(cell.tokens_per_s_samples)["median"] if cell.tokens_per_s_samples else None
            info = pareto.get(prec)
            pareto_mark = "—"
            dominators = "—"
            if info is not None:
                pareto_mark = "✅" if info["pareto"] else "❌"
                dominators = ", ".join(_precision_label(p) for p in info["dominated_by"]) or "—"
            lines.append(
                f"| {_precision_label(prec)} | {_fmt_pt(cell.ppl, 4)} "
                f"| {_fmt_pt(cell.peak_memory_gib, 2)} | {_fmt_pt(tps_med, 2)} "
                f"| {pareto_mark} | {dominators} |"
            )
            compact.append(
                f"| {model_label(model_id)} | {_precision_label(prec)} "
                f"| {_fmt_pt(cell.ppl, 4)} | {_fmt_pct(relative_change_pct(base_ppl, cell.ppl), 1)} "
                f"| {_fmt_pt(cell.peak_memory_gib, 2)} | {_fmt_pct(relative_change_pct(base_mem, cell.peak_memory_gib), 0)} "
                f"| {_fmt_pt(tps_med, 2)} | {_fmt_pct(relative_change_pct(base_tps, tps_med), 0)} "
                f"| {pareto_mark} |"
            )
        lines.append("")
    return lines, compact


def build_csv(by_model: dict[str, dict[str, StatsCell]], test_rows: list[dict], source: str) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "source", "model_id", "metric", "precision", "median", "q1", "q3", "iqr", "n",
        "value", "pct_vs_fp16",
        "mwu_U", "mwu_Z", "mwu_p_raw", "mwu_p_holm", "r_effect", "cliff_delta", "cliff_interp",
        "kw_H", "kw_p", "note",
    ])
    for model_id, precisions in by_model.items():
        base = precisions.get(BASELINE)
        base_ppl = base.ppl if base else None
        base_mem = base.peak_memory_gib if base else None
        base_tps = None
        if base and base.tokens_per_s_samples:
            base_tps = summarize(base.tokens_per_s_samples)["median"]
        base_ttft = None
        if base and base.ttft_ms_samples:
            base_ttft = summarize(base.ttft_ms_samples)["median"]
        for prec in sorted(precisions, key=_precision_sort_key):
            cell = precisions[prec]
            for metric_key, baseline_value in (("ttft_ms", base_ttft), ("tokens_per_s", base_tps)):
                samples = _samples(cell, metric_key)
                if not samples:
                    continue
                s = summarize(samples)
                pct = relative_change_pct(baseline_value, s["median"]) if baseline_value is not None else None
                test = next(
                    (
                        r for r in test_rows
                        if r["model_id"] == model_id and r["metric"] == metric_key and r["precision"] == prec
                    ),
                    None,
                )
                writer.writerow([
                    source, model_id, metric_key, prec,
                    s["median"], s["q1"], s["q3"], s["iqr"], s["n"],
                    "", "" if pct is None else round(pct, 4),
                    test["mwu_U"] if test else "", test["mwu_Z"] if test else "",
                    test["mwu_p_raw"] if test else "", test["mwu_p_holm"] if test else "",
                    test["r_effect"] if test else "", test["cliff_delta"] if test else "",
                    test["cliff_interp"] if test else "",
                    test["kw_H"] if test else "", test["kw_p"] if test else "",
                    test["note"] if test else "",
                ])
            # PPL row (scalar)
            if cell.ppl is not None:
                writer.writerow([
                    source, model_id, "word_perplexity", prec,
                    "", "", "", "", 1,
                    cell.ppl,
                    "" if base_ppl is None else round(relative_change_pct(base_ppl, cell.ppl) or 0, 4),
                    "", "", "", "", "", "", "", "", "", "",
                ])
            if cell.peak_memory_gib is not None:
                writer.writerow([
                    source, model_id, "peak_memory_gib", prec,
                    "", "", "", "", 1,
                    cell.peak_memory_gib,
                    "" if base_mem is None else round(relative_change_pct(base_mem, cell.peak_memory_gib) or 0, 4),
                    "", "", "", "", "", "", "", "", "", "",
                ])
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.run_dir / "stats"
    _ensure_dir(out_dir)
    cells, source = load_run_dir(args.run_dir, source=args.source)
    if not cells:
        raise SystemExit(f"no usable cells in {args.run_dir}")
    by_model = group_by_model(cells)

    header = [
        f"# Etapa 6 — análise estatística ({source})",
        "",
        f"- Diretório: `{args.run_dir}`",
        f"- Precisões encontradas: {', '.join(sorted({c.precision for c in cells}, key=_precision_sort_key))}",
        f"- Modelos: {', '.join(by_model)}",
        f"- α = {args.alpha}",
        "",
    ]

    md_lines = header
    md_lines += build_summary_table(by_model)
    tests_lines, test_rows = build_tests_block(by_model, args.alpha)
    md_lines += tests_lines
    md_lines += build_ppl_block(by_model)
    pareto_lines, compact_lines = build_pareto_block(by_model)
    md_lines += pareto_lines

    md = "\n".join(md_lines) + "\n"
    (out_dir / "stats.md").write_text(md, encoding="utf-8")
    (out_dir / "trade_off.md").write_text("\n".join(compact_lines) + "\n", encoding="utf-8")
    (out_dir / "stats.csv").write_text(build_csv(by_model, test_rows, source), encoding="utf-8")

    (out_dir / "cells.json").write_text(
        json.dumps(
            [
                {
                    "model_id": c.model_id, "precision": c.precision, "source": c.source,
                    "ppl": c.ppl, "ppl_token": c.ppl_token,
                    "peak_memory_gib": c.peak_memory_gib, "disk_gib": c.disk_gib,
                    "n_ttft": len(c.ttft_ms_samples), "n_tps": len(c.tokens_per_s_samples),
                    "variants": c.variants,
                }
                for c in cells
            ],
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_dir}/stats.md")
    print(f"wrote {out_dir}/trade_off.md")
    print(f"wrote {out_dir}/stats.csv")
    print(f"wrote {out_dir}/cells.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
