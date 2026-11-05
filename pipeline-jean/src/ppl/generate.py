"""Qualitative smoke: one greedy completion. Not an official metric."""

from __future__ import annotations

from ppl.loading import load_model_and_tokenizer, resolve_device, resolve_dtype


def format_prompt(prompt: str) -> str:
    return f"Question: {prompt.strip()}\nAnswer:"


def generate_once(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> str:
    import torch

    text = format_prompt(prompt)
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    generated = output[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_for_generate(model_id: str, device: str, dtype_name: str | None = None):
    device_resolved = resolve_device(device)
    dtype = resolve_dtype(device_resolved, dtype_name)
    model, tokenizer = load_model_and_tokenizer(model_id, device_resolved, dtype)
    return model, tokenizer, device_resolved
