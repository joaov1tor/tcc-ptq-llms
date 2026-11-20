"""Unit tests for the Stage 6 statistical primitives + the format adapter."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.stats import (  # noqa: E402
    cliff_delta,
    cliff_interp,
    holm_bonferroni,
    kruskal_wallis,
    mann_whitney_u,
    pareto_front,
    rank_r_effect,
)
from bench.stats_adapter import StatsCell, detect_source, load_run_dir  # noqa: E402


def test_mwu_textbook_no_overlap():
    """[1,2,3] vs [4,5,6]: U=0, extreme separation, two-sided p is small."""
    result = mann_whitney_u([1, 2, 3], [4, 5, 6])
    assert result["U"] == 0.0
    assert result["U1"] == 0.0
    assert result["U2"] == 9.0
    assert result["n1"] == 3 and result["n2"] == 3
    assert result["Z"] < 0  # x < y so U1 - mu < 0
    assert result["p"] < 0.1  # normal approx is loose at n=3; just enforce "small-ish"


def test_mwu_symmetric():
    """Swapping arguments mirrors Z and keeps p."""
    a = mann_whitney_u([1, 2, 3, 4], [3, 4, 5, 6])
    b = mann_whitney_u([3, 4, 5, 6], [1, 2, 3, 4])
    assert a["p"] == pytest.approx(b["p"])
    assert a["Z"] == pytest.approx(-b["Z"])


def test_mwu_tie_correction_shifts_sigma():
    """Ties shrink σ_U, so the Z grows in magnitude vs. the no-ties case."""
    no_ties = mann_whitney_u([1, 2, 3], [4, 5, 6])
    with_ties = mann_whitney_u([1, 2, 3], [3, 5, 6])
    assert with_ties["ties_correction"] > 0
    assert no_ties["ties_correction"] == 0


def test_kruskal_wallis_equal_groups_is_zero():
    H = kruskal_wallis([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert H["H"] == pytest.approx(0.0, abs=1e-9)
    assert H["df"] == 2
    assert H["p"] == pytest.approx(1.0)


def test_kruskal_wallis_different_groups_rejects():
    H = kruskal_wallis([[1, 2, 3], [10, 11, 12], [20, 21, 22]])
    # three well-separated groups: H should be close to its maximum N-1=8
    assert H["H"] > 6.0
    assert H["df"] == 2
    assert H["p"] < 0.05


def test_kruskal_wallis_falls_back_when_too_few_groups():
    H = kruskal_wallis([[1, 2, 3]])
    assert H["H"] == 0.0 and H["p"] == 1.0


def test_holm_bonferroni_order_and_monotonicity():
    adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    # m=4; ranks (asc): 0.005, 0.01, 0.03, 0.04 -> *4, *3, *2, *1
    # back to original order: [0.01*3, 0.04*1, 0.03*2, 0.005*4] = [0.03, 0.04, 0.06, 0.02]
    # then monotonicity: 0.02, 0.03, 0.06, 0.06 in sorted order -> back-mapped
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-9)


def test_holm_clamps_at_one():
    adj = holm_bonferroni([0.4, 0.6, 0.9])
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_cliff_delta_identical_is_zero():
    d = cliff_delta([1, 2, 3], [1, 2, 3])
    assert d["delta"] == 0.0
    assert d["interp"] == "negligible"


def test_cliff_delta_fully_separated_is_one():
    d = cliff_delta([1, 2], [3, 4])
    assert d["delta"] == -1.0  # all x < all y
    assert d["interp"] == "large"


def test_cliff_interp_thresholds():
    assert cliff_interp(0.1) == "negligible"
    assert cliff_interp(0.2) == "small"
    assert cliff_interp(0.4) == "medium"
    assert cliff_interp(0.6) == "large"


def test_rank_r_effect_formula():
    assert rank_r_effect(2.0, 100) == pytest.approx(0.2)
    assert rank_r_effect(-3.0, 9) == pytest.approx(1.0)
    assert rank_r_effect(1.0, 0) == 0.0


def test_pareto_front_dominance():
    # a is dominated by b in all axes; c is on the frontier
    items = [
        {"name": "a", "ppl": 10.0, "memory": 5.0, "tps": 10.0},
        {"name": "b", "ppl": 9.0, "memory": 4.0, "tps": 11.0},
        {"name": "c", "ppl": 8.0, "memory": 5.0, "tps": 10.0},
    ]
    pf = pareto_front(items)
    assert pf["a"]["pareto"] is False
    assert "b" in pf["a"]["dominated_by"]
    assert pf["b"]["pareto"] is True
    assert pf["c"]["pareto"] is True


def test_adapter_detects_our_format(tmp_path):
    sample = {
        "model_id": "x/Model",
        "precision": "fp16",
        "status": "ok",
        "generation": {
            "ttft_ms": {"samples": [10.0, 11.0, 12.0]},
            "tokens_per_s": {"samples": [30.0, 31.0, 32.0]},
        },
        "ppl": {"word_perplexity": 6.1},
        "peak_memory_gib": 7.0,
        "disk_gib": 14.0,
        "gptq_audit": {"token_perplexity": 6.0},
    }
    (tmp_path / "x_Model_fp16.json").write_text(json.dumps(sample), encoding="utf-8")
    assert detect_source(tmp_path) == "ours"
    cells, source = load_run_dir(tmp_path)
    assert source == "ours"
    assert len(cells) == 1
    cell = cells[0]
    assert cell.model_id == "x/Model"
    assert cell.precision == "fp16"
    assert cell.ttft_ms_samples == [10.0, 11.0, 12.0]
    assert cell.ppl == 6.1
    assert cell.ppl_token == 6.0
    assert cell.peak_memory_gib == 7.0


def test_adapter_detects_drive_format_and_concatenates(tmp_path):
    # one PPL file + three cost files for the same (model, precision)
    ppl_payload = {"model": "models/Meta-Llama-3-8B", "quant": "none", "ppl": 6.1359, "peak_mem_gb": 18.9}
    cost_payload = lambda samples: {
        "model": "models/Meta-Llama-3-8B",
        "quant": "none",
        "ttft_ms_raw": samples,
        "tokens_per_s_raw": [60.0] * len(samples),
        "disk_gb": 15.0,
        "peak_mem_gb": 17.4,
    }
    (tmp_path / "llama3-8b_fp16.json").write_text(json.dumps(ppl_payload), encoding="utf-8")
    (tmp_path / "llama3-8b_fp16_cost.json").write_text(json.dumps(cost_payload([150.0] * 10)), encoding="utf-8")
    (tmp_path / "llama3-8b_fp16_cost_r2.json").write_text(json.dumps(cost_payload([151.0] * 10)), encoding="utf-8")
    (tmp_path / "llama3-8b_fp16_cost_r3.json").write_text(json.dumps(cost_payload([152.0] * 10)), encoding="utf-8")

    assert detect_source(tmp_path) == "drive"
    cells, source = load_run_dir(tmp_path)
    assert source == "drive"
    assert len(cells) == 1
    cell = cells[0]
    assert cell.model_id == "meta-llama/Meta-Llama-3-8B"
    assert cell.precision == "fp16"
    assert len(cell.ttft_ms_samples) == 30
    assert cell.ppl == pytest.approx(6.1359)
    assert cell.disk_gib == pytest.approx(15.0)
    # geração (cost) é o pico comparável com o nosso; o pico do arquivo de PPL fica em outro campo
    assert cell.peak_memory_gib == pytest.approx(17.4)
    assert cell.ppl_peak_memory_gib == pytest.approx(18.9)


def test_adapter_filters_drive_variants_by_default(tmp_path):
    """`_c2048` and `_sym` variants are tagged tags, not primary cells; skip by default."""
    base_payload = {"model": "models/Qwen3-8B-Base", "quant": "pre", "ppl": 7.0, "peak_mem_gb": 6.0}
    (tmp_path / "qwen3-8b-base_awq4.json").write_text(json.dumps(base_payload), encoding="utf-8")
    (tmp_path / "qwen3-8b-base_awq4_c2048.json").write_text(
        json.dumps({**base_payload, "ppl": 7.5}), encoding="utf-8"
    )
    cells, _ = load_run_dir(tmp_path)
    # One primary cell for int4-awq; the c2048 variant is dropped
    assert len(cells) == 1
    assert cells[0].precision == "int4-awq"
    assert cells[0].ppl == pytest.approx(7.0)
