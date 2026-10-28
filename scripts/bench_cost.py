#!/usr/bin/env python3
"""Custo de inferência: disco, memória (após carga e pico), TTFT e tokens/s.

Protocolo: batch 1, decodificação gulosa, mesmo prompt em todas as configurações
(primeiros --prompt-len tokens do split de VALIDAÇÃO do wikitext-2-raw-v1, para não usar o teste),
--warmup execuções descartadas, --reps medidas. Sem parar no EOS (min_new_tokens).
  TTFT     = tempo de generate(max_new_tokens=1)                      (prefill + 1º token)
  tokens/s = (N-1) / (t[generate(N)] - t[generate(1)])                (decode puro)
Reporta mediana e IQR; guarda os valores brutos (para Kruskal-Wallis / Mann-Whitney).
Memória (em GiB): peak_mem_gb = pico de torch.cuda.max_memory_allocated DURANTE a geração (pesos + KV + ativações, sem o transiente de carga); peak_reserved_gb = max_memory_reserved; (na DGX a memória é unificada; o nvidia-smi não separa).

Uso: python bench_cost.py --model <dir> [--dtype float16] [--out r.json]
"""
import argparse, glob, json, os, platform, subprocess, time
import numpy as np, torch, transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--quant", default="none", choices=["none", "int8", "pre"])  # pre = diretório já quantizado salvo em disco
ap.add_argument("--prompt-len", type=int, default=512)
ap.add_argument("--new-tokens", type=int, default=128)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
assert torch.cuda.is_available(), "medição de custo exige GPU"
torch.manual_seed(a.seed)
GiB = 2**30

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"erro: {e}"

def q(x): 
    x = np.array(x); return dict(median=round(float(np.median(x)), 4), q1=round(float(np.percentile(x, 25)), 4),
                                 q3=round(float(np.percentile(x, 75)), 4), iqr=round(float(np.percentile(x, 75) - np.percentile(x, 25)), 4))

# Disco = tamanho dos pesos salvos (um só formato, para não contar duas vezes).
files = glob.glob(os.path.join(a.model, "*.safetensors")) or glob.glob(os.path.join(a.model, "*.bin"))  # um só formato: evita contagem dupla
disk = sum(os.path.getsize(f) for f in files)
others = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
temp0 = sh("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader")

tok = AutoTokenizer.from_pretrained(a.model)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
kw = dict(dtype=getattr(torch, a.dtype))
if a.quant in ("int8", "pre"):  # o modelo já nasce na GPU; int8 = LLM.int8() on-the-fly via bitsandbytes
    if a.quant == "int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    kw["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
if a.quant == "none":
    model = model.to("cuda")
model = model.eval()
mem_load = torch.cuda.memory_allocated()

# Prompt de 512 tokens do split de VALIDAÇÃO (não do teste usado na perplexidade); o mesmo em todas as configurações.
val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
ids = tok("\n\n".join(val["text"]), return_tensors="pt").input_ids[:, :a.prompt_len].to("cuda")
assert ids.size(1) == a.prompt_len
mask = torch.ones_like(ids)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# synchronize() antes e depois: a GPU é assíncrona, sem isso o relógio pararia antes do fim do trabalho. Guloso, sem parar no EOS.
def gen(n):
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                   pad_token_id=pad, use_cache=True)
    torch.cuda.synchronize(); return time.perf_counter() - t

# Zera o pico depois da carga: o pico reportado = pesos + cache KV + ativações da geração (sem o transiente de carga).
torch.cuda.reset_peak_memory_stats()
ttft, tps, invalid_tps = [], [], 0
with torch.no_grad():
    for i in range(a.warmup + a.reps):
        # gen(1) = prefill + 1 token (TTFT); gen(N) - gen(1) cancela o prefill e deixa só o custo de decodificar N-1 tokens.
        t1 = gen(1); tn = gen(a.new_tokens)
        if i >= a.warmup:
            ttft.append(t1 * 1000)
            if tn > t1: tps.append((a.new_tokens - 1) / (tn - t1))
            else: invalid_tps += 1  # jitter: geração longa mais rápida que a curta; descartada e contada
peak = torch.cuda.max_memory_allocated(); peak_res = torch.cuda.max_memory_reserved()
temp1 = sh("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader")

res = dict(model=a.model, dtype=a.dtype, quant=a.quant, prompt_len=a.prompt_len, new_tokens=a.new_tokens, warmup=a.warmup, reps=a.reps, seed=a.seed,
           disk_gb=round(disk / GiB, 3), mem_after_load_gb=round(mem_load / GiB, 3), peak_mem_gb=round(peak / GiB, 3), peak_reserved_gb=round(peak_res / GiB, 3), tps_invalid_reps=invalid_tps,
           ttft_ms=q(ttft), tokens_per_s=q(tps), ttft_ms_raw=[round(x, 2) for x in ttft], tokens_per_s_raw=[round(x, 3) for x in tps],
           gpu=torch.cuda.get_device_name(0), gpu_other_procs_before=others, gpu_temp_clock_throttle_before=temp0, gpu_temp_clock_throttle_after=temp1,
           torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(), ts=time.strftime("%Y-%m-%d %H:%M:%S"))
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_raw")}, ensure_ascii=False, indent=1))
if a.out: json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
