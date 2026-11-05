import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench.cost import summarize, tokens_per_second  # noqa: E402
from bench.report import (  # noqa: E402
    build_rows,
    compared_precisions,
    fmt_number,
    fmt_pct,
    load_cells,
    relative_change_pct,
    render_csv,
    render_markdown,
    within_threshold,
)
from ppl.loading import PRECISION_REGISTRY, PRECISIONS, get_precision_spec  # noqa: E402
from ppl.protocol import gptq_blocks, gptq_join_lines  # noqa: E402


def make_cell(model_id, precision, ppl, disk, peak, ttft, tps, status="ok"):
    return {
        "model_id": model_id,
        "precision": precision,
        "status": status,
        "device": "cuda",
        "memory_kind": "cuda_max_memory_allocated",
        "profile": "full",
        "ppl": {"word_perplexity": ppl, "protocol": "harness_wikitext_v2", "n_docs": 62, "max_length": 2048},
        "disk_gib": disk,
        "peak_memory_gib": peak,
        "generation": {"ttft_ms": {"median": ttft}, "tokens_per_s": {"median": tps}},
    }


# Numbers from the project's reference table (Llama 3 8B, FP16 vs INT8).
LLAMA = [
    make_cell("meta-llama/Meta-Llama-3-8B", "fp16", 6.1359, 14.96, 15.11, 150, 13.58),
    make_cell("meta-llama/Meta-Llama-3-8B", "int8", 6.2278, 8.46, 8.63, 169, 18.14),
]


def test_relative_change_and_threshold():
    pct = relative_change_pct(6.1359, 6.2278)
    assert pct == pytest.approx(1.4977, abs=1e-3)
    assert within_threshold(pct) is True
    assert within_threshold(3.0) is True
    assert within_threshold(3.01) is False
    assert relative_change_pct(None, 1.0) is None
    assert relative_change_pct(0.0, 1.0) is None


def test_pt_br_formatting():
    assert fmt_number(6.1359, 4) == "6,1359"
    assert fmt_number(None, 2) == "—"
    assert fmt_pct(1.4977, 1) == "+1,5%"
    assert fmt_pct(-43.4, 0) == "-43%"


def test_table_matches_reference_layout():
    md = render_markdown(LLAMA)
    assert "| Métrica | Modelo | FP16 | INT8 | Δ INT8 |" in md
    assert "| **PPL** | Meta-Llama-3-8B | 6,1359 | 6,2278 | **+1,5%** ✅ |" in md
    assert "| **Disco (GiB)** | Meta-Llama-3-8B | 14,96 | 8,46 | -43% |" in md
    assert "| **TTFT (ms)** | Meta-Llama-3-8B | 150 | 169 | +13% |" in md
    assert "| **Tokens/s** | Meta-Llama-3-8B | 13,58 | 18,14 | +34% |" in md
    assert "fora da meta de ~4 GB" in md
    # GPTQ audit was not run: its row is omitted instead of showing dashes.
    assert "auditoria" not in md


def test_failed_cell_is_reported_not_dropped():
    cells = [LLAMA[0], {"model_id": LLAMA[1]["model_id"], "precision": "int8", "status": "failed", "error": "OOM"}]
    rows = build_rows(cells)
    ppl = next(r for r in rows if r["metric"].key == "word_perplexity")
    entry = ppl["precisions"]["int8"]
    assert entry["value"] is None and entry["pct"] is None and entry["within"] is None
    md = render_markdown(cells)
    assert "| failed |" in md
    assert "erro=OOM" in md


def test_threshold_violation_marked():
    cells = [
        make_cell("x/Model", "fp16", 10.0, 1, 1, 1, 1),
        make_cell("x/Model", "int8", 10.5, 1, 1, 1, 1),
    ]
    assert "**+5,0%** ❌" in render_markdown(cells)


def test_csv_and_load_cells(tmp_path):
    for cell in LLAMA:
        (tmp_path / f"{cell['precision']}.json").write_text(json.dumps(cell))
    (tmp_path / "env.json").write_text(json.dumps({"gpu": "x"}))
    cells = load_cells(tmp_path)
    assert len(cells) == 2
    csv_text = render_csv(cells)
    assert csv_text.splitlines()[0] == "metric,model_id,precision,value,baseline,pct,within_ppl_threshold"
    assert "word_perplexity,meta-llama/Meta-Llama-3-8B,int8,6.2278,6.1359,1.4977,True" in csv_text


def test_dynamic_columns_with_int4():
    cells = [
        make_cell("x/Model", "fp16", 10.0, 10.0, 10.0, 100, 20.0),
        make_cell("x/Model", "int8", 10.1, 5.0, 5.0, 110, 25.0),
        make_cell("x/Model", "int4-awq", 10.4, 3.0, 3.0, 90, 30.0),
        make_cell("x/Model", "int4-gptq", 10.6, 3.0, 3.0, 95, 29.0),
    ]
    assert compared_precisions(cells) == ["int8", "int4-awq", "int4-gptq"]
    md = render_markdown(cells)
    header = "| Métrica | Modelo | FP16 | INT8 | Δ INT8 | INT4-AWQ | Δ INT4-AWQ | INT4-GPTQ | Δ INT4-GPTQ |"
    assert header in md
    # INT4-GPTQ exceeds +3% on PPL -> ❌; INT4-AWQ at +4% also ❌; INT8 at +1% -> ✅
    assert "**+4,0%** ❌" in md  # INT4-AWQ PPL
    assert "**+6,0%** ❌" in md  # INT4-GPTQ PPL
    assert "**+1,0%** ✅" in md  # INT8 PPL


def test_long_csv_has_a_row_per_precision():
    cells = [
        make_cell("x/Model", "fp16", 10.0, 10.0, 10.0, 100, 20.0),
        make_cell("x/Model", "int8", 10.1, 5.0, 5.0, 110, 25.0),
        make_cell("x/Model", "int4-awq", 10.4, 3.0, 3.0, 90, 30.0),
    ]
    csv_text = render_csv(cells)
    lines = csv_text.splitlines()
    assert lines[0] == "metric,model_id,precision,value,baseline,pct,within_ppl_threshold"
    # one row per (metric, precision); PPL has 2 precisions here, so 2 PPL rows
    ppl_rows = [line for line in lines if line.startswith("word_perplexity,")]
    assert len(ppl_rows) == 2
    assert any(",int8," in r for r in ppl_rows)
    assert any(",int4-awq," in r for r in ppl_rows)


def test_precision_registry_shape():
    assert PRECISIONS == ("fp16", "int8", "int4-awq", "int4-gptq")
    for name, spec in PRECISION_REGISTRY.items():
        assert callable(spec.loader), f"{name}: loader is not callable"
        assert spec.backend is None or isinstance(spec.backend, str)
        assert isinstance(spec.requires_cuda, bool)
    assert PRECISION_REGISTRY["fp16"].requires_cuda is False
    assert PRECISION_REGISTRY["int8"].requires_cuda is False
    assert PRECISION_REGISTRY["int4-awq"].requires_cuda is True
    assert PRECISION_REGISTRY["int4-gptq"].requires_cuda is True
    assert PRECISION_REGISTRY["int4-awq"].backend == "gptqmodel-awq"
    assert PRECISION_REGISTRY["int4-gptq"].backend == "gptqmodel-gptq"


def test_get_precision_spec_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_precision_spec("int2-vq")


def test_skipped_int4_cell_shows_up_in_footer():
    cells = [
        make_cell("x/Model", "fp16", 10.0, 10.0, 10.0, 100, 20.0),
        {
            "model_id": "x/Model",
            "precision": "int4-awq",
            "status": "skipped",
            "device": "mps",
            "quant_backend": "gptqmodel-awq",
            "error": "int4-awq requires CUDA (>=8.0); device=mps",
        },
    ]
    md = render_markdown(cells)
    assert "| skipped |" in md
    assert "exige CUDA" in md
    assert "quant_backend=gptqmodel-awq" in md


def test_tokens_per_second_excludes_first_token():
    # 11 tokens: the first arrives at t=0.5 s, the other 10 by t=1.5 s -> 10 tok/s.
    assert tokens_per_second(11, t_first=0.5, t_end=1.5) == pytest.approx(10.0)
    assert math.isnan(tokens_per_second(1, 0.5, 1.5))
    assert math.isnan(tokens_per_second(5, 1.0, 1.0))


def test_summarize_median_iqr():
    stats = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["median"] == 3.0
    assert stats["q1"] == 2.0 and stats["q3"] == 4.0 and stats["iqr"] == 2.0
    assert stats["n"] == 5
    single = summarize([7.0])
    assert single["median"] == 7.0 and single["iqr"] == 0.0
    assert summarize([float("nan")])["median"] is None


def test_gptq_blocks_drop_tail():
    blocks = gptq_blocks(list(range(10)), seqlen=4)
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert gptq_blocks([1, 2], seqlen=4) == []
    assert gptq_join_lines(["a", "", "b"]) == "a\n\n\n\nb"


def test_prompts_file_is_valid():
    data = json.loads((ROOT / "prompts" / "generation_v1.json").read_text(encoding="utf-8"))
    assert data["version"] == "generation_v1"
    assert len(data["prompts"]) == 10
    assert len(set(data["prompts"])) == 10


def test_score_windows_matches_full_log_softmax():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from ppl.evaluate import score_windows
    from ppl.protocol import rolling_windows

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(32, 32)

        def forward(self, ids):
            return type("Out", (), {"logits": self.emb(ids)})()

    model = FakeModel()
    tokens = [3, 7, 1, 9, 4, 4, 2, 8, 5]
    windows = rolling_windows(tokens, prefix_token=0, max_seq_len=4)
    got_ll, got_n = score_windows(model, windows, "cpu")

    expected = 0.0
    for context, continuation in windows:
        ids = torch.tensor([context + continuation])
        log_probs = F.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
        targets = ids[0, 1:]
        expected += log_probs[-len(continuation) :].gather(-1, targets[-len(continuation) :, None]).sum().item()
    assert got_n == len(tokens)
    assert got_ll == pytest.approx(expected, rel=1e-6)