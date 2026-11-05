"""Resolve device/dtype and load a checkpoint under a given precision.

Precisions live in a small registry so adding INT4 (AWQ, GPTQ, …) is one entry
and the rest of the pipeline (CLI validation, cell loop, artifact save) adapts
automatically. Loaders only consume paths and keep the heavy quantization step
in `src/ppl/quantize.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


class _Quantizer(Protocol):
    def __call__(self, src_dir: str, dst_dir: str, *, nsamples: int = 128, seed: int = 0) -> None: ...


_LoaderFn = Callable[..., Any]  # (model_path, *, device, dtype, revision) -> model


@dataclass(frozen=True)
class PrecisionSpec:
    backend: str | None
    loader: _LoaderFn
    quantizer: _Quantizer | None
    requires_cuda: bool


def resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, dtype_name: str | None):
    import torch

    if dtype_name:
        return getattr(torch, dtype_name)
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_tokenizer(model_path: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_fp16(model_path: str, *, device: str, dtype, revision: str | None = None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, revision=revision)
    model.to(device)
    return model


def _load_bnb_int8(model_path: str, *, device: str, dtype, revision: str | None = None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        revision=revision,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device,
    )


def _load_gptqmodel(model_path: str, *, device: str, dtype, revision: str | None = None):
    """Load a pre-quantized gptqmodel checkpoint (AWQ or GPTQ, same loader).

    gptqmodel só aceita strings de `device_map` do dicionário HF
    ('auto'/'balanced'/'balanced_low_0'/'sequential'); 'cuda' quebra. Como
    `requires_cuda=True` já garante que estamos em GPU, 'auto' coloca tudo no
    único dispositivo visível.
    """
    from gptqmodel import GPTQModel  # type: ignore[import-not-found]

    return GPTQModel.load(model_path, device_map="auto")


PRECISION_REGISTRY: dict[str, PrecisionSpec] = {
    "fp16": PrecisionSpec(backend=None, loader=_load_fp16, quantizer=None, requires_cuda=False),
    "int8": PrecisionSpec(backend="bitsandbytes", loader=_load_bnb_int8, quantizer=None, requires_cuda=False),
    "int4-awq": PrecisionSpec(backend="gptqmodel-awq", loader=_load_gptqmodel, quantizer=None, requires_cuda=True),
    "int4-gptq": PrecisionSpec(backend="gptqmodel-gptq", loader=_load_gptqmodel, quantizer=None, requires_cuda=True),
}

PRECISIONS = tuple(PRECISION_REGISTRY)


def _resolve_quantizer(precision: str) -> _Quantizer | None:
    """Lazy import so a Mac without gptqmodel still imports loading.py."""
    if precision == "int4-awq":
        from ppl.quantize import quantize_awq

        return quantize_awq
    if precision == "int4-gptq":
        from ppl.quantize import quantize_gptq

        return quantize_gptq
    return None


def get_precision_spec(precision: str) -> PrecisionSpec:
    spec = PRECISION_REGISTRY.get(precision)
    if spec is None:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if spec.quantizer is None:
        quantizer = _resolve_quantizer(precision)
        if quantizer is not None:
            spec = PrecisionSpec(spec.backend, spec.loader, quantizer, spec.requires_cuda)
    return spec


def load_model(
    model_path: str,
    *,
    precision: str = "fp16",
    device: str,
    dtype=None,
    revision: str | None = None,
):
    """Dispatch to the registered loader for `precision`.

    `dtype` is the compute dtype (float16 by default); loaders that keep part of
    the graph unquantized (embeddings, norms, lm_head) use it there.
    """
    import torch

    spec = get_precision_spec(precision)
    dtype = dtype or torch.float16
    model = spec.loader(model_path, device=device, dtype=dtype, revision=revision)
    model.eval()
    return model


def load_model_and_tokenizer(model_id: str, device: str, dtype, revision: str | None = None):
    model = load_model(model_id, precision="fp16", device=device, dtype=dtype, revision=revision)
    return model, load_tokenizer(model_id, revision=revision)


def save_quantized(model, tokenizer, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
