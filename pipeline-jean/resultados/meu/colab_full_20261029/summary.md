| Métrica | Modelo | FP16 | INT8 | Δ INT8 | INT4-AWQ | Δ INT4-AWQ | INT4-GPTQ | Δ INT4-GPTQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PPL** | Qwen3-8B-Base | 10,9507 | 11,0593 | **+1,0%** ✅ | 13,2402 | **+20,9%** ❌ | 11,4565 | **+4,6%** ❌ |
|  | Meta-Llama-3-8B | 7,7942 | 7,9167 | **+1,6%** ✅ | 8,6511 | **+11,0%** ❌ | 8,5778 | **+10,1%** ❌ |
| **PPL GPTQ (auditoria)** | Qwen3-8B-Base | 6,9945 | 7,0269 | +0,5% | 7,9941 | +14,3% | 7,2888 | +4,2% |
|  | Meta-Llama-3-8B | 6,1358 | 6,2267 | +1,5% | 6,7370 | +9,8% | 6,6275 | +8,0% |
| **Disco (GiB)** | Qwen3-8B-Base | 15,26 | 8,79 | -42% | 5,68 | -63% | 5,68 | -63% |
|  | Meta-Llama-3-8B | 14,96 | 8,46 | -43% | 5,33 | -64% | 5,34 | -64% |
| **Pico de memória (GiB)** | Qwen3-8B-Base | 15,29 | 8,95 | -41% | 9,16 | -40% | 9,19 | -40% |
|  | Meta-Llama-3-8B | 14,99 | 8,65 | -42% | 8,65 | -42% | 8,28 | -45% |
| **TTFT (ms)** | Qwen3-8B-Base | 57 | 236 | +316% | 72 | +26% | 68 | +20% |
|  | Meta-Llama-3-8B | 41 | 186 | +357% | 53 | +30% | 51 | +25% |
| **Tokens/s** | Qwen3-8B-Base | 18,86 | 5,28 | -72% | 14,61 | -23% | 15,58 | -17% |
|  | Meta-Llama-3-8B | 27,63 | 7,41 | -73% | 20,24 | -27% | 21,40 | -23% |

**O que isso diz:**
- **Qualidade INT8:** variação da PPL frente ao FP16: Qwen3-8B-Base +1,0% (abaixo do limiar de 3%); Meta-Llama-3-8B +1,6% (abaixo do limiar de 3%).
- **Qualidade INT4-AWQ:** variação da PPL frente ao FP16: Qwen3-8B-Base +20,9% (acima do limiar de 3%); Meta-Llama-3-8B +11,0% (acima do limiar de 3%).
- **Qualidade INT4-GPTQ:** variação da PPL frente ao FP16: Qwen3-8B-Base +4,6% (acima do limiar de 3%); Meta-Llama-3-8B +10,1% (acima do limiar de 3%).
- **Memória INT8:** Qwen3-8B-Base 8,95 GiB (-41%; fora da meta de ~4 GB); Meta-Llama-3-8B 8,65 GiB (-42%; fora da meta de ~4 GB).
- **Memória INT4-AWQ:** Qwen3-8B-Base 9,16 GiB (-40%; fora da meta de ~4 GB); Meta-Llama-3-8B 8,65 GiB (-42%; fora da meta de ~4 GB).
- **Memória INT4-GPTQ:** Qwen3-8B-Base 9,19 GiB (-40%; fora da meta de ~4 GB); Meta-Llama-3-8B 8,28 GiB (-45%; fora da meta de ~4 GB).
- **Tempo INT8:** Qwen3-8B-Base: TTFT +316%, tokens/s -72%; Meta-Llama-3-8B: TTFT +357%, tokens/s -73% (mediana; TTFT menor é melhor, tokens/s maior é melhor).
- **Tempo INT4-AWQ:** Qwen3-8B-Base: TTFT +26%, tokens/s -23%; Meta-Llama-3-8B: TTFT +30%, tokens/s -27% (mediana; TTFT menor é melhor, tokens/s maior é melhor).
- **Tempo INT4-GPTQ:** Qwen3-8B-Base: TTFT +20%, tokens/s -17%; Meta-Llama-3-8B: TTFT +25%, tokens/s -23% (mediana; TTFT menor é melhor, tokens/s maior é melhor).

**Condições do run:**
- Qwen3-8B-Base FP16: status=ok, device=cuda, dtype=float16, quant_backend=None, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=49e3418fbbbca6ecbdf9608b4d22e5a407081db4
- Qwen3-8B-Base INT4-AWQ: status=ok, device=cuda, dtype=bfloat16, quant_backend=gptqmodel-awq, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=49e3418fbbbca6ecbdf9608b4d22e5a407081db4, calibração=c4-en/train/00000-of-01024 (128 docs, seed=0, concat=2048)
- Qwen3-8B-Base INT4-GPTQ: status=ok, device=cuda, dtype=bfloat16, quant_backend=gptqmodel-gptq, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=49e3418fbbbca6ecbdf9608b4d22e5a407081db4, calibração=c4-en/train/00000-of-01024 (128 docs, seed=0, concat=2048)
- Qwen3-8B-Base INT8: status=ok, device=cuda, dtype=float16, quant_backend=bitsandbytes, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=49e3418fbbbca6ecbdf9608b4d22e5a407081db4
- Meta-Llama-3-8B FP16: status=ok, device=cuda, dtype=float16, quant_backend=None, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
- Meta-Llama-3-8B INT4-AWQ: status=ok, device=cuda, dtype=bfloat16, quant_backend=gptqmodel-awq, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920, calibração=c4-en/train/00000-of-01024 (128 docs, seed=0, concat=2048)
- Meta-Llama-3-8B INT4-GPTQ: status=ok, device=cuda, dtype=bfloat16, quant_backend=gptqmodel-gptq, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920, calibração=c4-en/train/00000-of-01024 (128 docs, seed=0, concat=2048)
- Meta-Llama-3-8B INT8: status=ok, device=cuda, dtype=float16, quant_backend=bitsandbytes, PPL=harness_wikitext_v2 (62 docs, contexto 2048), geração=10 prompts × 10 reps, 128 tokens, warmup descartado=3, memória=cuda_max_memory_allocated, revisão=8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
- A PPL oficial é a `word_perplexity` do protocolo do harness; o valor absoluto não se compara às tabelas do GPTQ/AWQ (ver docs/wikitext2-versao-e-tokenizacao.md).
