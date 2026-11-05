"""Harness-faithful WikiText-2 detokenizer, rolling windows and PPL aggregation.

Detokenizer and word/byte counts follow:
https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/wikitext/preprocess_wikitext.py

Rolling windows follow:
https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/utils.py
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from typing import Any

PROTOCOL_NAME = "harness_wikitext_v2"

# Audit only (Etapa 1): the protocol behind the GPTQ/AWQ tables. Not the official metric.
# https://github.com/IST-DASLab/gptq/blob/main/datautils.py (get_wikitext2) + llama.py (llama_eval)
GPTQ_PROTOCOL_NAME = "gptq_wikitext2"
GPTQ_DATASET_PATH = "Salesforce/wikitext"
GPTQ_DATASET_CONFIG = "wikitext-2-raw-v1"
GPTQ_SEQLEN = 2048


def wikitext_detokenizer(page: str) -> str:
    """Official harness detokenizer. Input is the raw WikiText-2 `page`."""
    string = page
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def count_words_and_bytes(page: str) -> tuple[int, int]:
    """Counts are taken on the original page, before detokenization."""
    n_words = len(re.split(r"\s+", page))
    n_bytes = len(page.encode("utf-8"))
    return n_words, n_bytes


def is_empty_page(page: str) -> bool:
    return not page or not page.strip()


def get_rolling_token_windows(
    token_list: list[int],
    prefix_token: int,
    max_seq_len: int,
    context_len: int = 1,
) -> Iterator[tuple[list[int], list[int]]]:
    """Copy of lm_eval.utils.get_rolling_token_windows (harness v0.4.13)."""
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    first_seq_len = min(max_seq_len, len(token_list))
    yield [prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len]
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(
    pair: tuple[list[int], list[int]],
) -> tuple[list[int], list[int]]:
    """Copy of lm_eval.utils.make_disjoint_window (harness v0.4.13)."""
    context, continuation = pair
    return context[: len(context) - (len(continuation) - 1)], continuation


def rolling_windows(
    token_ids: list[int],
    prefix_token: int,
    max_seq_len: int,
    context_len: int = 1,
) -> list[tuple[list[int], list[int]]]:
    return [
        make_disjoint_window(pair)
        for pair in get_rolling_token_windows(
            token_list=token_ids,
            prefix_token=prefix_token,
            max_seq_len=max_seq_len,
            context_len=context_len,
        )
    ]


def gptq_join_lines(lines: Iterable[str]) -> str:
    """GPTQ/AWQ concatenate the line-level test split with blank lines, no detokenizer."""
    return "\n\n".join(lines)


def gptq_blocks(token_ids: list[int], seqlen: int = GPTQ_SEQLEN) -> list[list[int]]:
    """Non-overlapping blocks of `seqlen`; the tail shorter than `seqlen` is dropped."""
    n_blocks = len(token_ids) // seqlen
    return [token_ids[i * seqlen : (i + 1) * seqlen] for i in range(n_blocks)]


def word_perplexity(sum_ll: float, n_words: int) -> float:
    if n_words <= 0:
        return float("inf")
    return math.exp(-sum_ll / n_words)


def byte_perplexity(sum_ll: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        return float("inf")
    return math.exp(-sum_ll / n_bytes)


def bits_per_byte(sum_ll: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        return float("inf")
    return (-sum_ll / n_bytes) / math.log(2)


def token_perplexity(sum_ll: float, n_tokens: int) -> float:
    if n_tokens <= 0:
        return float("inf")
    return math.exp(-sum_ll / n_tokens)


def aggregate_metrics(
    sum_ll: float, n_words: int, n_bytes: int, n_tokens: int
) -> dict[str, float]:
    return {
        "word_perplexity": word_perplexity(sum_ll, n_words),
        "byte_perplexity": byte_perplexity(sum_ll, n_bytes),
        "bits_per_byte": bits_per_byte(sum_ll, n_bytes),
        "token_perplexity": token_perplexity(sum_ll, n_tokens),
        "sum_loglikelihood": sum_ll,
        "n_words": n_words,
        "n_bytes": n_bytes,
        "n_tokens_scored": n_tokens,
    }


def prepare_document(page: str) -> dict[str, Any] | None:
    """Return counts + detokenized text, or None if the page is empty."""
    if is_empty_page(page):
        return None
    n_words, n_bytes = count_words_and_bytes(page)
    return {
        "n_words": n_words,
        "n_bytes": n_bytes,
        "text": wikitext_detokenizer(page),
    }


def iter_prepared_pages(
    pages: Iterable[str], max_docs: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    skipped = 0
    for page in pages:
        doc = prepare_document(page)
        if doc is None:
            skipped += 1
            continue
        prepared.append(doc)
        if max_docs is not None and len(prepared) >= max_docs:
            break
    return prepared, skipped
