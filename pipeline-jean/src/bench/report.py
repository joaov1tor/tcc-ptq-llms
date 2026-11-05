"""Summary table FP16 × (INT8, INT4-AWQ, INT4-GPTQ, …), in the layout used by the project report.

Columns are driven by whichever precisions actually show up in the cells, so a
smoke run that only measured FP16+INT8 keeps the old two-column look while a
full run with INT4 grows additional columns automatically.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PPL_THRESHOLD_PCT = 3.0  # §4: reference threshold of relative PPL increase vs FP16
VRAM_TARGET_GIB = 4.0  # §2: target around 4 GB of VRAM
BASELINE = "fp16"
COMPARED_ORDER: tuple[str, ...] = ("int8", "int4-awq", "int4-gptq")
PRECISION_LABEL: dict[str, str] = {
    "fp16": "FP16",
    "int8": "INT8",
    "int4-awq": "INT4-AWQ",
    "int4-gptq": "INT4-GPTQ",
}


def _label(precision: str) -> str:
    return PRECISION_LABEL.get(precision, precision.upper())


def _get(*keys: str) -> Callable[[dict[str, Any]], float | None]:
    def getter(cell: dict[str, Any]) -> float | None:
        value: Any = cell
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value if isinstance(value, (int, float)) else None

    return getter


@dataclass(frozen=True)
class Metric:
    label: str
    key: str
    get: Callable[[dict[str, Any]], float | None]
    decimals: int
    pct_decimals: int
    threshold: bool = False


METRICS = (
    Metric("PPL", "word_perplexity", _get("ppl", "word_perplexity"), 4, 1, threshold=True),
    Metric("PPL GPTQ (auditoria)", "gptq_token_perplexity", _get("gptq_audit", "token_perplexity"), 4, 1),
    Metric("Disco (GiB)", "disk_gib", _get("disk_gib"), 2, 0),
    Metric("Pico de memória (GiB)", "peak_memory_gib", _get("peak_memory_gib"), 2, 0),
    Metric("TTFT (ms)", "ttft_ms_median", _get("generation", "ttft_ms", "median"), 0, 0),
    Metric("Tokens/s", "tokens_per_s_median", _get("generation", "tokens_per_s", "median"), 2, 0),
)


def model_label(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def relative_change_pct(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0


def within_threshold(pct: float | None, threshold: float = PPL_THRESHOLD_PCT) -> bool | None:
    return None if pct is None else pct <= threshold


def fmt_number(value: float | None, decimals: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def fmt_pct(pct: float | None, decimals: int) -> str:
    if pct is None:
        return "—"
    return f"{pct:+.{decimals}f}%".replace(".", ",")


def load_cells(run_dir: Path) -> list[dict[str, Any]]:
    cells = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "env.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "precision" in data and "model_id" in data:
            cells.append(data)
    return cells


def _index(cells: list[dict[str, Any]]) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    models: list[str] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        if cell["model_id"] not in models:
            models.append(cell["model_id"])
        by_key[(cell["model_id"], cell["precision"])] = cell
    return models, by_key


def _cell_value(cell: dict[str, Any] | None, metric: Metric) -> float | None:
    if cell is None or cell.get("status") != "ok":
        return None
    return metric.get(cell)


def compared_precisions(cells: list[dict[str, Any]]) -> list[str]:
    """Return the precisions to compare against FP16, in canonical order, filtered by presence."""
    present = {cell["precision"] for cell in cells}
    return [p for p in COMPARED_ORDER if p in present]


def build_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models, by_key = _index(cells)
    compared = compared_precisions(cells)
    rows = []
    for metric in METRICS:
        if all(_cell_value(by_key.get((m, p)), metric) is None for m in models for p in (BASELINE, *compared)):
            continue  # e.g. the GPTQ audit was not run and no cell has gptq_token_perplexity
        for model_id in models:
            base_cell = by_key.get((model_id, BASELINE))
            base = _cell_value(base_cell, metric)
            row: dict[str, Any] = {
                "metric": metric,
                "model_id": model_id,
                "fp16": base,
                "fp16_status": (base_cell or {}).get("status", "missing"),
                "precisions": {},
            }
            for precision in compared:
                comp_cell = by_key.get((model_id, precision))
                comp = _cell_value(comp_cell, metric)
                pct = relative_change_pct(base, comp)
                row["precisions"][precision] = {
                    "value": comp,
                    "pct": pct,
                    "status": (comp_cell or {}).get("status", "missing"),
                    "within": within_threshold(pct) if metric.threshold else None,
                }
            rows.append(row)
    return rows


def _value_or_status(value: float | None, status: str, decimals: int) -> str:
    if value is None and status not in ("ok", "missing"):
        return status  # "failed" / "skipped": recorded as not viable in this environment (§6.2)
    return fmt_number(value, decimals)


def _variation_cell(metric: Metric, entry: dict[str, Any]) -> str:
    variation = fmt_pct(entry["pct"], metric.pct_decimals)
    if entry["within"] is not None:
        variation = f"**{variation}** {'✅' if entry['within'] else '❌'}"
    return variation


def render_markdown(cells: list[dict[str, Any]]) -> str:
    rows = build_rows(cells)
    compared = compared_precisions(cells)

    header_cols = ["Métrica", "Modelo", _label(BASELINE)]
    for precision in compared:
        header_cols.append(_label(precision))
        header_cols.append(f"Δ {_label(precision)}")
    alignments = ["---", "---"] + ["---:"] * (len(header_cols) - 2)
    lines = ["| " + " | ".join(header_cols) + " |", "| " + " | ".join(alignments) + " |"]

    previous = None
    for row in rows:
        metric: Metric = row["metric"]
        label = f"**{metric.label}**" if metric is not previous else ""
        previous = metric
        cells_row = [
            label,
            model_label(row["model_id"]),
            _value_or_status(row["fp16"], row["fp16_status"], metric.decimals),
        ]
        for precision in compared:
            entry = row["precisions"][precision]
            cells_row.append(_value_or_status(entry["value"], entry["status"], metric.decimals))
            cells_row.append(_variation_cell(metric, entry))
        lines.append("| " + " | ".join(cells_row) + " |")

    lines.append("")
    lines.extend(findings(rows, compared))
    lines.append("")
    lines.extend(footnotes(cells))
    return "\n".join(lines) + "\n"


def findings(rows: list[dict[str, Any]], compared: list[str]) -> list[str]:
    """Plain-language reading of the table, only from the measured numbers."""
    out = ["**O que isso diz:**"]
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(row["metric"].key, []).append(row)

    quality_rows = by_metric.get("word_perplexity", [])
    if quality_rows:
        for precision in compared:
            parts = []
            for r in quality_rows:
                entry = r["precisions"].get(precision, {})
                pct = entry.get("pct")
                if pct is None:
                    continue
                parts.append(
                    f"{model_label(r['model_id'])} {fmt_pct(pct, 1)} "
                    f"({'abaixo' if entry['within'] else 'acima'} do limiar de {PPL_THRESHOLD_PCT:g}%)"
                )
            if parts:
                out.append(f"- **Qualidade {_label(precision)}:** variação da PPL frente ao FP16: " + "; ".join(parts) + ".")

    memory_rows = by_metric.get("peak_memory_gib", [])
    for precision in compared:
        parts = []
        for r in memory_rows:
            entry = r["precisions"].get(precision, {})
            value = entry.get("value")
            if value is None:
                continue
            fits = value <= VRAM_TARGET_GIB
            parts.append(
                f"{model_label(r['model_id'])} {fmt_number(value, 2)} GiB "
                f"({fmt_pct(entry['pct'], 0)}; {'dentro' if fits else 'fora'} da meta de ~{VRAM_TARGET_GIB:g} GB)"
            )
        if parts:
            out.append(f"- **Memória {_label(precision)}:** " + "; ".join(parts) + ".")

    ttft = by_metric.get("ttft_ms_median", [])
    tps = by_metric.get("tokens_per_s_median", [])
    for precision in compared:
        ttft_map = {
            r["model_id"]: r["precisions"][precision].get("pct")
            for r in ttft
            if r["precisions"].get(precision, {}).get("pct") is not None
        }
        tps_map = {
            r["model_id"]: r["precisions"][precision].get("pct")
            for r in tps
            if r["precisions"].get(precision, {}).get("pct") is not None
        }
        if not (ttft_map or tps_map):
            continue
        parts = []
        for model_id in dict.fromkeys([*ttft_map, *tps_map]):
            bits = []
            if model_id in ttft_map:
                bits.append(f"TTFT {fmt_pct(ttft_map[model_id], 0)}")
            if model_id in tps_map:
                bits.append(f"tokens/s {fmt_pct(tps_map[model_id], 0)}")
            parts.append(f"{model_label(model_id)}: " + ", ".join(bits))
        out.append(
            f"- **Tempo {_label(precision)}:** " + "; ".join(parts)
            + " (mediana; TTFT menor é melhor, tokens/s maior é melhor)."
        )

    if len(out) == 1:
        out.append("- Ainda não há pares FP16/quantizado completos para comparar.")
    return out


def footnotes(cells: list[dict[str, Any]]) -> list[str]:
    out = ["**Condições do run:**"]
    seen = set()
    for cell in cells:
        ppl = cell.get("ppl") or {}
        gen = cell.get("generation") or {}
        note = (
            f"- {model_label(cell['model_id'])} {_label(cell['precision'])}: status={cell.get('status')}, "
            f"device={cell.get('device')}, dtype={cell.get('dtype')}, quant_backend={cell.get('quant_backend')}, "
            f"PPL={ppl.get('protocol')} ({ppl.get('n_docs')} docs, contexto {ppl.get('max_length')}), "
            f"geração={gen.get('n_prompts')} prompts × {gen.get('reps')} reps, {gen.get('max_new_tokens')} tokens, "
            f"warmup descartado={gen.get('warmup_discarded')}, memória={cell.get('memory_kind')}, "
            f"revisão={cell.get('model_revision')}"
        )
        if cell.get("calibration"):
            cal = cell["calibration"]
            note += (
                f", calibração={cal.get('dataset')} ({cal.get('nsamples')} docs, "
                f"seed={cal.get('seed')}, concat={cal.get('concat_size')})"
            )
        if cell.get("error"):
            note += f", erro={cell['error']}"
        out.append(note)
        seen.add(cell.get("profile"))
    if any(c.get("precision") == "int8" and c.get("device") != "cuda" for c in cells):
        out.append(
            "- INT8 fora de CUDA: o bitsandbytes usa um backend não otimizado em MPS/CPU; "
            "TTFT e tokens/s do INT8 aqui não representam o INT8 em GPU NVIDIA."
        )
    if any(c.get("precision", "").startswith("int4") and c.get("status") == "skipped" for c in cells):
        out.append(
            "- INT4 (AWQ/GPTQ via `gptqmodel`) exige CUDA compute capability ≥ 8.0; "
            "em MPS/CPU as células são registradas como `skipped`."
        )
    if any(c.get("memory_kind") != "cuda_max_memory_allocated" for c in cells if c.get("memory_kind")):
        out.append(
            "- Pico de memória fora de CUDA é amostrado (MPS) ou é o RSS do processo (CPU); "
            "não é o pico de VRAM."
        )
    if "smoke" in seen or "colab" in seen:
        out.append(
            "- Perfil de validação (smoke/colab): modelo e recorte reduzidos; "
            "os números não são comparáveis com a bancada de referência."
        )
    out.append(
        "- A PPL oficial é a `word_perplexity` do protocolo do harness; "
        "o valor absoluto não se compara às tabelas do GPTQ/AWQ (ver docs/wikitext2-versao-e-tokenizacao.md)."
    )
    return out


def render_csv(cells: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "model_id", "precision", "value", "baseline", "pct", "within_ppl_threshold"])
    for row in build_rows(cells):
        base = row["fp16"]
        for precision, entry in row["precisions"].items():
            writer.writerow(
                [
                    row["metric"].key,
                    row["model_id"],
                    precision,
                    "" if entry["value"] is None else entry["value"],
                    "" if base is None else base,
                    "" if entry["pct"] is None else round(entry["pct"], 4),
                    "" if entry["within"] is None else entry["within"],
                ]
            )
    return buffer.getvalue()


def write_summary(run_dir: Path) -> str:
    cells = load_cells(run_dir)
    markdown = render_markdown(cells)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (run_dir / "summary.csv").write_text(render_csv(cells), encoding="utf-8")
    return markdown
