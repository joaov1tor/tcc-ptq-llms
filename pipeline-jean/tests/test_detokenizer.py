import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppl.protocol import (  # noqa: E402
    aggregate_metrics,
    count_words_and_bytes,
    is_empty_page,
    make_disjoint_window,
    prepare_document,
    rolling_windows,
    wikitext_detokenizer,
)


def test_detokenizer_hyphen_and_punctuation():
    raw = "well @-@ known . Yes , it is : true"
    assert wikitext_detokenizer(raw) == "well-known. Yes, it is: true"


def test_detokenizer_headings_and_n_token():
    raw = "= = Heading = =\nThe  N  value"
    out = wikitext_detokenizer(raw)
    assert "== Heading ==" in out
    assert " 1 " in out


def test_detokenizer_contraction():
    assert wikitext_detokenizer("Wikipedia 's") == "Wikipedia's"


def test_word_count_uses_raw_text_not_detokenized():
    raw = "well @-@ known"
    n_words, n_bytes = count_words_and_bytes(raw)
    assert n_words == 3
    assert n_bytes == len(raw.encode("utf-8"))
    assert " @-@ " not in wikitext_detokenizer(raw)


def test_empty_page_skipped():
    assert is_empty_page("")
    assert is_empty_page("   \n")
    assert prepare_document("   ") is None
    assert prepare_document("Valjean") is not None


def test_rolling_windows_predict_each_token_once():
    tokens = [1, 2, 3, 4, 5]
    windows = rolling_windows(tokens, prefix_token=0, max_seq_len=3, context_len=1)
    scored = []
    for context, continuation in windows:
        scored.extend(continuation)
        assert context
        assert continuation
    assert scored == tokens


def test_disjoint_window_drops_overlapping_context():
    context, continuation = make_disjoint_window(([0, 1, 2], [1, 2, 3]))
    assert continuation == [1, 2, 3]
    assert context == [0]


def test_word_perplexity_formula():
    metrics = aggregate_metrics(sum_ll=-math.log(2) * 4, n_words=4, n_bytes=8, n_tokens=4)
    assert metrics["word_perplexity"] == pytest.approx(2.0)
    assert metrics["token_perplexity"] == pytest.approx(2.0)
    assert metrics["bits_per_byte"] == pytest.approx(0.5)
