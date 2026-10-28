"""[VARIANTE blocos 2048] INT4 GPTQ (gptqmodel), pesos 4 bits, grupo 128, calibração em C4 (NÃO em WikiText-2, para não vazar o teste).
Uso: python quantize_gptq_c2048.py <dir_fp16> <dir_saida> [--nsamples 128] [--seed 0]"""
import argparse, random
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--nsamples", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

# C4 en, 1 shard do train; documentos longos (>= 8000 caracteres, ~2000 tokens) para preencher o contexto de calibração
ds = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train")
idx = list(range(len(ds))); random.Random(a.seed).shuffle(idx)
calib = []
for i in idx:
    t = ds[i]["text"]
    if len(t) >= 8000:
        calib.append(t)
    if len(calib) == a.nsamples:
        break
assert len(calib) == a.nsamples, "shard do C4 sem documentos suficientes"
print("calibração: C4", len(calib), "docs, seed", a.seed)

cfg = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)
m = GPTQModel.load(a.src, cfg)
m.quantize(calib, calibration_concat_size=2048, batch_size=1)  # blocos de 2048 tokens, como no artigo do GPTQ
m.save(a.dst)
print("salvo em", a.dst)
