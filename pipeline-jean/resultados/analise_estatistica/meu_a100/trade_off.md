# Trade-off (uma linha por modelo × precisão)

| Modelo | Precisão | PPL | Δ PPL | Mem (GiB) | Δ Mem | Tokens/s (med) | Δ Tok/s | Pareto |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| Qwen3-8B-Base | FP16 | 10,9507 | +0,0% | 15,29 | +0% | 18,86 | +0% | ✅ |
| Qwen3-8B-Base | INT8 | 11,0593 | +1,0% | 8,95 | -41% | 5,28 | -72% | ✅ |
| Qwen3-8B-Base | INT4-AWQ | 13,2402 | +20,9% | 9,16 | -40% | 14,61 | -23% | ✅ |
| Qwen3-8B-Base | INT4-GPTQ | 11,4565 | +4,6% | 9,19 | -40% | 15,58 | -17% | ✅ |
| Meta-Llama-3-8B | FP16 | 7,7942 | +0,0% | 14,99 | +0% | 27,63 | +0% | ✅ |
| Meta-Llama-3-8B | INT8 | 7,9167 | +1,6% | 8,65 | -42% | 7,41 | -73% | ✅ |
| Meta-Llama-3-8B | INT4-AWQ | 8,6511 | +11,0% | 8,65 | -42% | 20,24 | -27% | ❌ |
| Meta-Llama-3-8B | INT4-GPTQ | 8,5778 | +10,1% | 8,28 | -45% | 21,40 | -23% | ✅ |
