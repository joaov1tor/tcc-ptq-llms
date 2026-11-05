"""Unify the two cell JSON shapes consumed by the Stage 6 analysis.

- Our pipeline (`scripts/run_benchmark.py`) writes one JSON per (model, precision)
  with `generation.ttft_ms.samples[]`, `generation.tokens_per_s.samples[]` and
  `ppl.word_perplexity`.
- The Drive dump splits each cell across one PPL file (`<slug>_<prec>.json`)
  and three cost files (`<slug>_<prec>_cost[_r2|_r3].json`) with top-level
  `ttft_ms_raw[]` and `tokens_per_s_raw[]` — the three cost files are three
  independent 10-rep runs that we concatenate (n=30 per group).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StatsCell:
    model_id: str
    precision: str  # fp16 | int8 | int4-awq | int4-gptq
    source: str  # "ours" | "drive"
    ppl: float | None = None  # word_perplexity (ours) or .ppl (drive)
    ppl_token: float | None = None  # gptq_audit.token_perplexity (ours) or .ppl (drive, which is already token-based)
    peak_memory_gib: float | None = None  # pico da fase de GERAÇÃO (cost); comparável entre ours e drive
    ppl_peak_memory_gib: float | None = None  # pico da fase de PPL (janela 2048); outra fase, não misturar
    disk_gib: float | None = None
    ttft_ms_samples: list[float] = field(default_factory=list)
    tokens_per_s_samples: list[float] = field(default_factory=list)
    status: str = "ok"
    variants: list[str] = field(default_factory=list)  # drive-only tags like "c2048", "sym"


_DRIVE_PRECISION = {
    "fp16": "fp16",
    "int8": "int8",
    "awq4": "int4-awq",
    "gptq4": "int4-gptq",
}
_DRIVE_MODEL_ALIASES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
}
_DRIVE_FILENAME_RE = re.compile(r"^(?P<model>[^_]+(?:-[^_]+)*)_(?P<precision>fp16|int8|awq4|gptq4)(?P<variants>(?:_[a-z0-9]+)*)?$")


def detect_source(run_dir: Path) -> str:
    """Drive dumps either have *_cost_r*.json or a top-level `ppl` field in a
    flat JSON matching the <alias>_<prec> naming; ours wraps PPL under `ppl.*`
    and names files like `<slug>_<prec>.json` with `precision` as a key.
    """
    for path in run_dir.glob("*_cost_r*.json"):
        return "drive"
    for path in sorted(run_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "precision" in data and "model_id" in data:
            return "ours"
        if isinstance(data.get("ppl"), (int, float)) and "quant" in data:
            return "drive"
    return "ours"


def _parse_drive_name(stem: str) -> tuple[str, str, list[str]] | None:
    """Extract (model_alias, precision_code, variants) from a Drive filename stem."""
    core = stem[:-5] if stem.endswith("_cost") else stem
    if core.startswith("smoke_"):
        return None
    m = re.match(r"^(?P<model>.+?)_(?P<precision>fp16|int8|awq4|gptq4)(?P<rest>.*)$", core)
    if not m:
        return None
    model_alias = m.group("model")
    precision_code = m.group("precision")
    rest = m.group("rest") or ""
    variants = [v for v in rest.strip("_").split("_") if v and not v.startswith("r") or (v.startswith("r") and not v[1:].isdigit())]
    return model_alias, precision_code, variants


def load_run_dir_ours(run_dir: Path) -> list[StatsCell]:
    """Read our per-cell JSONs; keep only cells with status=ok."""
    cells: list[StatsCell] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "env.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "precision" not in data or "model_id" not in data:
            continue
        if data.get("status") != "ok":
            continue
        gen = data.get("generation") or {}
        ppl = data.get("ppl") or {}
        audit = data.get("gptq_audit") or {}
        ttft = (gen.get("ttft_ms") or {}).get("samples") or []
        tps = (gen.get("tokens_per_s") or {}).get("samples") or []
        cells.append(
            StatsCell(
                model_id=data["model_id"],
                precision=data["precision"],
                source="ours",
                ppl=ppl.get("word_perplexity"),
                ppl_token=audit.get("token_perplexity") or ppl.get("token_perplexity"),
                peak_memory_gib=data.get("peak_memory_gib"),
                ppl_peak_memory_gib=data.get("peak_memory_ppl_gib"),
                disk_gib=data.get("disk_gib"),
                ttft_ms_samples=[float(x) for x in ttft],
                tokens_per_s_samples=[float(x) for x in tps],
                status=data["status"],
            )
        )
    return cells


def _canonical_drive_model(alias: str) -> str:
    return _DRIVE_MODEL_ALIASES.get(alias, alias)


def _drive_group_key(model_alias: str, precision_code: str, variants: list[str]) -> tuple[str, str, tuple[str, ...]]:
    return (_canonical_drive_model(model_alias), _DRIVE_PRECISION[precision_code], tuple(sorted(v for v in variants if v != "")))


def load_run_dir_drive(run_dir: Path, *, include_variants: bool = False) -> list[StatsCell]:
    """Reconstruct (model, precision) cells from the Drive layout.

    - PPL files (`<slug>_<prec>.json`, no `_cost` suffix) carry the single PPL.
    - Cost files (`<slug>_<prec>_cost*.json`) carry raw samples; we concatenate
      every `_cost`, `_cost_r2`, `_cost_r3` for the same (model, precision).
    - When `include_variants=False` (default), only files without variant tags
      (`_c2048`, `_sym`) are kept. This matches the Stage 6 primary comparison.
    """
    groups: dict[tuple[str, str, tuple[str, ...]], StatsCell] = {}
    cost_peaks: dict[tuple[str, str, tuple[str, ...]], list[float]] = {}
    for path in sorted(run_dir.glob("*.json")):
        stem = path.stem
        is_cost = "_cost" in stem
        parsed = _parse_drive_name(stem)
        if parsed is None:
            continue
        model_alias, precision_code, variants = parsed
        rep_tags = [v for v in variants if re.fullmatch(r"r\d+", v)]
        other_tags = [v for v in variants if not re.fullmatch(r"r\d+", v) and v != "cost"]
        if not include_variants and other_tags:
            continue
        key = _drive_group_key(model_alias, precision_code, other_tags)
        cell = groups.setdefault(
            key,
            StatsCell(
                model_id=key[0],
                precision=key[1],
                source="drive",
                variants=list(key[2]),
            ),
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        if is_cost:
            ttft = data.get("ttft_ms_raw") or []
            tps = data.get("tokens_per_s_raw") or []
            cell.ttft_ms_samples.extend(float(x) for x in ttft)
            cell.tokens_per_s_samples.extend(float(x) for x in tps)
            if "peak_mem_gb" in data:
                cost_peaks.setdefault(key, []).append(float(data["peak_mem_gb"]))
                cell.peak_memory_gib = statistics.median(cost_peaks[key])
            if cell.disk_gib is None and "disk_gb" in data:
                cell.disk_gib = float(data["disk_gb"])
        else:
            if cell.ppl is None and "ppl" in data:
                cell.ppl = float(data["ppl"])
                cell.ppl_token = float(data["ppl"])  # drive's .ppl is already token-based
            if cell.ppl_peak_memory_gib is None and "peak_mem_gb" in data:
                cell.ppl_peak_memory_gib = float(data["peak_mem_gb"])
        _ = rep_tags  # kept for debugging: three cost files is one n=30 group
    return list(groups.values())


def load_run_dir(run_dir: Path, *, source: str = "auto") -> tuple[list[StatsCell], str]:
    """Dispatch to the right loader; return (cells, source_used)."""
    resolved = detect_source(run_dir) if source == "auto" else source
    if resolved == "drive":
        return load_run_dir_drive(run_dir), "drive"
    return load_run_dir_ours(run_dir), "ours"
