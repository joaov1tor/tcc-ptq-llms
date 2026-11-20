# Trade-off (uma linha por modelo × precisão)

| Modelo | Precisão | PPL | Δ PPL | Mem (GiB) | Δ Mem | Tokens/s (med) | Δ Tok/s | Pareto |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| Meta-Llama-3-8B | FP16 | 6,1358 | +0,0% | 15,09 | +0% | 25,87 | +0% | ✅ |
| Meta-Llama-3-8B | INT8 | 6,2267 | +1,5% | 8,61 | -43% | 7,04 | -73% | ✅ |
| Meta-Llama-3-8B | INT4-AWQ | 6,7338 | +9,7% | 5,46 | -64% | 18,79 | -27% | ❌ |
| Meta-Llama-3-8B | INT4-GPTQ | 6,6235 | +7,9% | 5,44 | -64% | 19,61 | -24% | ✅ |
| Qwen3-8B-Base | FP16 | 6,9945 | +0,0% | 15,39 | +0% | 17,87 | +0% | ✅ |
| Qwen3-8B-Base | INT8 | 7,0269 | +0,5% | 9,20 | -40% | 5,01 | -72% | ✅ |
| Qwen3-8B-Base | INT4-AWQ | 7,9838 | +14,1% | 5,81 | -62% | 13,56 | -24% | ❌ |
| Qwen3-8B-Base | INT4-GPTQ | 7,2795 | +4,1% | 5,79 | -62% | 13,94 | -22% | ✅ |
