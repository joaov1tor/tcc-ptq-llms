"""INT4 AWQ and GPTQ quantization via gptqmodel, calibrated on C4 (same lock
as the private scripts: 128 docs with len >= 8000 chars, seed 0, from shard
en/c4-train.00000-of-01024.json.gz; blocks of 2048 tokens as in the paper).

Calibrating on C4 — not WikiText-2 — avoids leaking the evaluation corpus into
the quantization step. The same calibration list feeds AWQ and GPTQ so the two
algorithms are compared fairly.
"""

from __future__ import annotations

import logging
import random

CALIB_SHARD = "en/c4-train.00000-of-01024.json.gz"
CALIB_MIN_CHARS = 8000  # ~2000 tokens, enough to fill a 2048-token calibration block
CALIB_CONCAT = 2048


def _c4_calibration(nsamples: int = 128, seed: int = 0) -> list[str]:
    """128 long documents from one C4 shard, shuffled with `seed`."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", data_files=CALIB_SHARD, split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    out: list[str] = []
    for i in indices:
        text = ds[i]["text"]
        if len(text) >= CALIB_MIN_CHARS:
            out.append(text)
        if len(out) == nsamples:
            break
    if len(out) != nsamples:
        raise RuntimeError(
            f"C4 shard produced {len(out)}/{nsamples} docs with len>={CALIB_MIN_CHARS}; "
            "widen the shard or lower CALIB_MIN_CHARS."
        )
    return out


def _quantize(method: str, src_dir: str, dst_dir: str, *, nsamples: int, seed: int, logger: logging.Logger | None):
    from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore[import-not-found]

    log = logger or logging.getLogger("quantize")
    log.info("calibrating %s from C4 (%d docs, seed=%d)", method, nsamples, seed)
    calib = _c4_calibration(nsamples, seed)

    if method == "awq":
        cfg = QuantizeConfig(method="awq", bits=4, group_size=128, sym=False)
    elif method == "gptq":
        cfg = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'awq' or 'gptq'")

    log.info("loading FP16 weights from %s with %s", src_dir, cfg)
    model = GPTQModel.load(src_dir, cfg)
    log.info("quantizing (%s, bits=4, group=128, concat=%d, batch=1)", method, CALIB_CONCAT)
    model.quantize(calib, calibration_concat_size=CALIB_CONCAT, batch_size=1)
    log.info("saving quantized checkpoint to %s", dst_dir)
    model.save(dst_dir)


def quantize_awq(src_dir: str, dst_dir: str, *, nsamples: int = 128, seed: int = 0, logger: logging.Logger | None = None) -> None:
    """AWQ (zero-point, group 128, 4 bits), C4 calibration, 2048-token blocks."""
    _quantize("awq", src_dir, dst_dir, nsamples=nsamples, seed=seed, logger=logger)


def quantize_gptq(src_dir: str, dst_dir: str, *, nsamples: int = 128, seed: int = 0, logger: logging.Logger | None = None) -> None:
    """GPTQ (symmetric, no act-order, group 128, 4 bits), C4 calibration, 2048-token blocks."""
    _quantize("gptq", src_dir, dst_dir, nsamples=nsamples, seed=seed, logger=logger)
