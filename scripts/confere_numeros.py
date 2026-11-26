# Recalcula os números do relatório a partir dos JSONs brutos (DGX e A100)
import json, statistics as st, sys, os
from scipy.stats import mannwhitneyu
R = os.path.expanduser("~/pos-genai/referencia")
DGX, A100 = f"{R}/resultados", f"{R}/pipeline-jean/resultados/dele/a100_replica"
M = {"llama3-8b": "Llama", "qwen3-8b-base": "Qwen"}; P = ["fp16", "int8", "gptq4", "awq4"]
def j(d, f): return json.load(open(f"{d}/{f}.json"))
def cost(d, m, p):
    rs = [j(d, f"{m}_{p}_cost" + s) for s in ("", "_r2", "_r3")]
    return rs
out = {}
for m, mn in M.items():
    base = {k: j(d, f"{m}_fp16")["ppl"] for k, d in (("dgx", DGX), ("a100", A100))}
    for p in P:
        row = {}
        for k, d in (("dgx", DGX), ("a100", A100)):
            ppl = j(d, f"{m}_{p}")["ppl"]; rs = cost(d, m, p)
            ttft = sum((r["ttft_ms_raw"] for r in rs), []); tps = sum((r["tokens_per_s_raw"] for r in rs), [])
            row[k] = dict(ppl=ppl, dppl=(ppl / base[k] - 1) * 100, peak=st.median(r["peak_mem_gb"] for r in rs),
                          disk=rs[0]["disk_gb"], ttft_pool=st.median(ttft), tps_pool=st.median(tps),
                          ttft_mm=st.median(r["ttft_ms"]["median"] for r in rs), tps_mm=st.median(r["tokens_per_s"]["median"] for r in rs),
                          ttft_raw=ttft, tps_raw=tps, n=len(ttft), procs=[r.get("gpu_other_procs_before", "") for r in rs])
        out[(mn, p)] = row
print(f"{'cel':12} {'PPLdgx':>7} {'Δdgx%':>6} {'PPLa100':>7} {'Δa100%':>6} {'|dPPL|':>6} {'pkD':>6} {'pkA':>6} {'TTFTmm':>7} {'TTFTpool':>8} {'TTFTa':>7} {'TPSmm':>6} {'TPSpool':>7} {'TPSa':>6} {'nD':>3} {'nA':>3}")
for (mn, p), r in out.items():
    d, a = r["dgx"], r["a100"]
    print(f"{mn+'-'+p:12} {d['ppl']:7.4f} {d['dppl']:6.2f} {a['ppl']:7.4f} {a['dppl']:6.2f} {abs(d['ppl']-a['ppl']):6.4f} {d['peak']:6.3f} {a['peak']:6.3f} {d['ttft_mm']:7.2f} {d['ttft_pool']:8.2f} {a['ttft_pool']:7.2f} {d['tps_mm']:6.2f} {d['tps_pool']:7.2f} {a['tps_pool']:6.2f} {d['n']:3} {a['n']:3}")
print("\nReduções de pico vs FP16 (DGX) e razões de vazão:")
for mn in M.values():
    f = out[(mn, "fp16")]
    for p in P[1:]:
        r = out[(mn, p)]
        print(f"{mn}-{p}: pico -{(1-r['dgx']['peak']/f['dgx']['peak'])*100:.2f}%  vazão DGX {r['dgx']['tps_mm']/f['dgx']['tps_mm']:.3f}x (pool {r['dgx']['tps_pool']/f['dgx']['tps_pool']:.3f}x)  A100 {r['a100']['tps_pool']/f['a100']['tps_pool']:.3f}x  TTFT DGX {(r['dgx']['ttft_mm']/f['dgx']['ttft_mm']-1)*100:+.1f}%  TTFT A100 {r['a100']['ttft_pool']/f['a100']['ttft_pool']:.2f}x")
print("\nMann-Whitney DGX x A100 (bilateral), min/max p bruto:")
ps = []
for (mn, p), r in out.items():
    for met in ("ttft_raw", "tps_raw"):
        u = mannwhitneyu(r["dgx"][met], r["a100"][met], alternative="two-sided"); ps.append((u.pvalue, u.statistic, mn, p, met))
def holm(pv):  # Holm-Bonferroni: p ajustados, monotônicos, limitados a 1
    o = sorted(range(len(pv)), key=pv.__getitem__); adj = [0.0] * len(pv); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(pv) - k) * pv[i])); adj[i] = run
    return adj
raw = [x[0] for x in ps]; adj = holm(raw)
print(f"p bruto: {min(raw):.2e} a {max(raw):.2e}; U de {min(x[1] for x in ps):.0f} a {max(x[1] for x in ps):.0f}")
print(f"Holm (m=16): {min(adj):.2e} a {max(adj):.2e}; Holm por métrica (m=8): {max(holm([x[0] for x in ps if x[4]=='ttft_raw']) + holm([x[0] for x in ps if x[4]=='tps_raw'])):.2e}")
