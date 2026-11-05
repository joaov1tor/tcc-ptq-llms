"""Score WikiText-2 with the harness protocol, on a loaded model or a ready HF checkpoint."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppl.loading import load_model_and_tokenizer, resolve_device, resolve_dtype
from ppl.logging_utils import setup_logger, slug_model_id
from ppl.protocol import (
    GPTQ_DATASET_CONFIG,
    GPTQ_DATASET_PATH,
    GPTQ_PROTOCOL_NAME,
    GPTQ_SEQLEN,
    PROTOCOL_NAME,
    aggregate_metrics,
    gptq_blocks,
    gptq_join_lines,
    iter_prepared_pages,
    rolling_windows,
)

DATASET_PATH = "EleutherAI/wikitext_document_level"
DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_DOCS = 4
# Official context lock (docs/wikitext2-versao-e-tokenizacao.md). Smaller values are smoke only.
DEFAULT_MAX_LENGTH = 2048


def resolve_model_revision(model_id: str, revision: str | None = None) -> str | None:
    """Commit SHA on the Hub, or None when offline / local path."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id, revision=revision).sha
    except Exception:
        return revision


def resolve_dataset_revision(dataset_path: str, revision: str | None = None) -> str | None:
    try:
        from huggingface_hub import dataset_info

        return dataset_info(dataset_path, revision=revision).sha
    except Exception:
        return revision


def prefix_token_id(tokenizer) -> int:
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def load_wikitext_pages(
    dataset_path: str,
    dataset_config: str,
    split: str,
    revision: str | None = None,
) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_path, dataset_config, split=split, revision=revision)
    field = "page" if "page" in ds.column_names else "text"
    return [row[field] or "" for row in ds]


def score_windows(model, windows: list[tuple[list[int], list[int]]], device: str) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    total_ll = 0.0
    n_tokens = 0
    for context, continuation in windows:
        if not continuation:
            continue
        input_ids = context + continuation
        tensor = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            logits = model(tensor).logits
        # Only the positions that predict the continuation, upcast to fp32 so the
        # fp16 rounding of log_softmax does not mix with the quantization effect.
        scored_logits = logits[0, -len(continuation) - 1 : -1, :].float()
        log_probs = F.log_softmax(scored_logits, dim=-1)
        tgt = tensor[0, -len(continuation) :]
        total_ll += log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        n_tokens += len(continuation)
    return total_ll, n_tokens


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def evaluate_perplexity(
    *,
    model_id: str = DEFAULT_MODEL,
    model=None,
    tokenizer=None,
    model_revision: str | None = None,
    dataset_path: str = DATASET_PATH,
    dataset_config: str = DATASET_CONFIG,
    dataset_revision: str | None = None,
    split: str = DEFAULT_SPLIT,
    max_docs: int | None = DEFAULT_MAX_DOCS,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "auto",
    dtype_name: str | None = None,
    dry_run: bool = False,
    output_path: Path | None = None,
    log_dir: Path | None = None,
    logger: logging.Logger | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Harness `wikitext` PPL. Pass `model`/`tokenizer` to reuse an already loaded
    (e.g. INT8) model; otherwise the FP16 checkpoint `model_id` is loaded here."""
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = None
    if logger is None:
        if log_dir is None:
            raise ValueError("pass log_dir or logger")
        log_path = log_dir / f"ppl_{stamp}_{slug_model_id(model_id)}.log"
        logger = setup_logger(log_path, verbose=verbose)
    device_resolved = "cpu" if dry_run else resolve_device(device)

    logger.info("protocol=%s", PROTOCOL_NAME)
    dataset_revision = resolve_dataset_revision(dataset_path, dataset_revision)
    logger.info(
        "dataset=%s config=%s revision=%s split=%s max_docs=%s max_length=%s",
        dataset_path,
        dataset_config,
        dataset_revision,
        split,
        max_docs,
        max_length,
    )

    pages = load_wikitext_pages(dataset_path, dataset_config, split, revision=dataset_revision)
    prepared, skipped = iter_prepared_pages(pages, max_docs=max_docs)
    n_words = sum(doc["n_words"] for doc in prepared)
    n_bytes = sum(doc["n_bytes"] for doc in prepared)
    logger.info(
        "loaded_pages=%s prepared=%s skipped_empty=%s n_words=%s n_bytes=%s",
        len(pages),
        len(prepared),
        skipped,
        n_words,
        n_bytes,
    )

    payload: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "dry_run": dry_run,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_name": None,
        "dataset_path": dataset_path,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split": split,
        "max_docs": max_docs,
        "max_length": max_length,
        "device": device_resolved,
        "dtype": None,
        "n_docs": len(prepared),
        "n_docs_skipped_empty": skipped,
        "n_pages_in_split": len(pages),
        "n_words": n_words,
        "n_bytes": n_bytes,
        "n_tokens_scored": 0,
        "word_perplexity": None,
        "byte_perplexity": None,
        "bits_per_byte": None,
        "token_perplexity": None,
        "sum_loglikelihood": None,
        "log_path": str(log_path) if log_path else None,
        "started_utc": stamp,
    }

    if dry_run:
        payload["duration_s"] = round(time.perf_counter() - started, 3)
        payload["note"] = "dry-run: dataset + detokenizer only; model not loaded"
        logger.info("dry-run complete duration_s=%.3f", payload["duration_s"])
        _write_json(output_path, payload)
        return payload

    if model is None:
        dtype = resolve_dtype(device_resolved, dtype_name)
        model_revision = resolve_model_revision(model_id, model_revision)
        payload["model_revision"] = model_revision
        logger.info(
            "loading model=%s revision=%s device=%s dtype=%s",
            model_id,
            model_revision,
            device_resolved,
            str(dtype).replace("torch.", ""),
        )
        model, tokenizer = load_model_and_tokenizer(model_id, device_resolved, dtype, revision=model_revision)
    payload["dtype"] = str(model.dtype).replace("torch.", "")
    prefix = prefix_token_id(tokenizer)
    payload["tokenizer_name"] = tokenizer.name_or_path
    payload["prefix_token_id"] = prefix

    import torch
    import transformers

    payload["torch_version"] = torch.__version__
    payload["transformers_version"] = transformers.__version__

    sum_ll = 0.0
    n_tokens = 0
    for idx, doc in enumerate(prepared, start=1):
        doc_t0 = time.perf_counter()
        token_ids = encode_text(tokenizer, doc["text"])
        windows = rolling_windows(token_ids, prefix_token=prefix, max_seq_len=max_length)
        doc_ll, doc_tokens = score_windows(model, windows, device_resolved)
        sum_ll += doc_ll
        n_tokens += doc_tokens
        logger.info(
            "doc=%s/%s n_tokens=%s n_windows=%s ll=%.4f elapsed_s=%.2f",
            idx,
            len(prepared),
            doc_tokens,
            len(windows),
            doc_ll,
            time.perf_counter() - doc_t0,
        )
        if verbose:
            logger.debug("doc=%s input_tokens=%s", idx, len(token_ids))

    metrics = aggregate_metrics(sum_ll, n_words, n_bytes, n_tokens)
    payload.update(metrics)
    payload["duration_s"] = round(time.perf_counter() - started, 3)
    logger.info(
        "done word_ppl=%.4f byte_ppl=%.4f bpb=%.4f token_ppl=%.4f duration_s=%.3f",
        metrics["word_perplexity"],
        metrics["byte_perplexity"],
        metrics["bits_per_byte"],
        metrics["token_perplexity"],
        payload["duration_s"],
    )
    _write_json(output_path, payload)
    return payload


def evaluate_gptq_audit(
    *,
    model,
    tokenizer,
    device: str,
    seqlen: int = GPTQ_SEQLEN,
    max_blocks: int | None = None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Token PPL in the GPTQ/AWQ protocol, only to cross-check published tables (Etapa 1)."""
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    started = time.perf_counter()
    revision = resolve_dataset_revision(GPTQ_DATASET_PATH)
    ds = load_dataset(GPTQ_DATASET_PATH, GPTQ_DATASET_CONFIG, split="test", revision=revision)
    # Same call as GPTQ's datautils: default add_special_tokens (BOS once, if the tokenizer has one).
    token_ids = tokenizer(gptq_join_lines(ds["text"]))["input_ids"]
    blocks = gptq_blocks(token_ids, seqlen)
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    logger.info(
        "audit protocol=%s dataset=%s revision=%s seqlen=%s n_blocks=%s",
        GPTQ_PROTOCOL_NAME,
        GPTQ_DATASET_PATH,
        revision,
        seqlen,
        len(blocks),
    )
    nll_sum = 0.0
    for block in blocks:
        tensor = torch.tensor([block], device=device)
        with torch.no_grad():
            logits = model(tensor).logits[0, :-1, :].float()
        loss = F.cross_entropy(logits, tensor[0, 1:])
        # GPTQ multiplies the mean loss by seqlen and divides by n_blocks * seqlen.
        nll_sum += loss.item() * seqlen
    ppl = math.exp(nll_sum / (len(blocks) * seqlen)) if blocks else float("inf")
    duration = round(time.perf_counter() - started, 3)
    logger.info("audit done token_ppl=%.4f duration_s=%.3f", ppl, duration)
    return {
        "protocol": GPTQ_PROTOCOL_NAME,
        "dataset_path": GPTQ_DATASET_PATH,
        "dataset_config": GPTQ_DATASET_CONFIG,
        "dataset_revision": revision,
        "seqlen": seqlen,
        "n_blocks": len(blocks),
        "token_perplexity": ppl,
        "duration_s": duration,
    }


def _write_json(output_path: Path | None, payload: dict[str, Any]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")