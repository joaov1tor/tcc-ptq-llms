# Trade-off (uma linha por modelo × precisão)

| Modelo | Precisão | PPL | Δ PPL | Mem (GiB) | Δ Mem | Tokens/s (med) | Δ Tok/s | Pareto |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| Meta-Llama-3-8B | FP16 | 6,1359 | +0,0% | 15,11 | +0% | 13,57 | +0% | ✅ |
| Meta-Llama-3-8B | INT8 | 6,2278 | +1,5% | 8,63 | -43% | 18,14 | +34% | ✅ |
| Meta-Llama-3-8B | INT4-AWQ | 6,7248 | +9,6% | 5,49 | -64% | 36,30 | +167% | ❌ |
| Meta-Llama-3-8B | INT4-GPTQ | 6,6754 | +8,8% | 5,46 | -64% | 36,51 | +169% | ✅ |
| Qwen3-8B-Base | FP16 | 6,9946 | +0,0% | 15,41 | +0% | 11,22 | +0% | ✅ |
| Qwen3-8B-Base | INT8 | 7,0299 | +0,5% | 9,22 | -40% | 15,21 | +36% | ✅ |
| Qwen3-8B-Base | INT4-AWQ | 7,9623 | +13,8% | 5,83 | -62% | 32,87 | +193% | ❌ |
| Qwen3-8B-Base | INT4-GPTQ | 7,3008 | +4,4% | 5,81 | -62% | 33,01 | +194% | ✅ |
