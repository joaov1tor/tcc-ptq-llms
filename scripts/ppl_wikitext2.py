#!/usr/bin/env python3
"""PPL por token no WikiText-2 (wikitext-2-raw-v1, split test), estilo GPTQ/AWQ.

Protocolo: teste concatenado com "\n\n", tokenizer nativo do modelo, janelas sem
sobreposição de --ctx tokens (padrão 2048), PPL = exp(média do NLL por token).
Comparável com os valores publicados (Llama 3 8B ~6,1; Qwen3-8B-Base ~6,99).
NÃO é comparável com word_perplexity do lm-evaluation-harness.

Uso: python ppl_wikitext2.py --model <dir|repo> [--dtype float16] [--ctx 2048] [--out r.json]
"""
import argparse, json, math, platform, time
import torch, transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--quant", default="none", choices=["none", "int8", "pre"])  # pre = diretório já quantizado salvo em disco
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--limit", type=int, default=0, help="máx. de janelas (0 = todas); só para teste rápido")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="")
a = ap.parse_args()

torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
kw = dict(dtype=getattr(torch, a.dtype))
if a.quant in ("int8", "pre"):  # o modelo já nasce na GPU; int8 = LLM.int8() on-the-fly via bitsandbytes
    if a.quant == "int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    kw["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
if a.quant == "none":
    model = model.to(a.device)
model = model.eval()

# Teste do WikiText-2 unido por \n\n e tokenizado de uma vez; n = nº de janelas inteiras de ctx tokens (o resto final é descartado).
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
n = ids.numel() // a.ctx
if a.limit: n = min(n, a.limit)

if a.device == "cuda": torch.cuda.reset_peak_memory_stats()
# nll = soma do negativo do log da probabilidade do token correto, em todas as janelas.
nll, t0 = 0.0, time.time()
with torch.no_grad():
    for i in range(n):
        x = ids[:, i * a.ctx:(i + 1) * a.ctx].to(a.device)
        # O logit da posição i prevê o token i+1: por isso logits[:, :-1] contra x[:, 1:]. .float() evita erro numérico do fp16.
        logits = model(x).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
# Cada janela avalia ctx-1 tokens. PPL = exp(NLL médio por token).
ntok = n * (a.ctx - 1)  # o primeiro token de cada janela não tem alvo
ppl = math.exp(nll / ntok)
res = dict(model=a.model, dtype=a.dtype, quant=a.quant, ctx=a.ctx, windows=n, tokens_scored=ntok, ppl=round(ppl, 4),
           seconds=round(time.time() - t0, 1), seed=a.seed,
           peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device == "cuda" else None,
           gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else None,
           torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
           dataset="wikitext-2-raw-v1/test", ts=time.strftime("%Y-%m-%d %H:%M:%S"))
print(json.dumps(res, ensure_ascii=False, indent=1))
if a.out: json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
