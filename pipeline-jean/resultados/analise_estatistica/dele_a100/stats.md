# Etapa 6 — análise estatística (drive)

- Diretório: `resultados/dele/a100_replica`
- Precisões encontradas: fp16, int8, int4-awq, int4-gptq
- Modelos: meta-llama/Meta-Llama-3-8B, Qwen/Qwen3-8B-Base
- α = 0.05

## Mediana e IQR por métrica, precisão e modelo

### Meta-Llama-3-8B

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 48,84 | 48,80 | 48,97 | 0,17 | 30 |
| TTFT (ms) | INT8 | 208,88 | 205,48 | 210,12 | 4,64 | 30 |
| TTFT (ms) | INT4-AWQ | 70,61 | 70,46 | 70,72 | 0,26 | 30 |
| TTFT (ms) | INT4-GPTQ | 69,70 | 69,64 | 69,77 | 0,13 | 30 |
| Tokens/s | FP16 | 25,87 | 25,68 | 26,20 | 0,52 | 30 |
| Tokens/s | INT8 | 7,04 | 7,01 | 7,15 | 0,14 | 30 |
| Tokens/s | INT4-AWQ | 18,79 | 18,68 | 18,93 | 0,26 | 30 |
| Tokens/s | INT4-GPTQ | 19,61 | 19,42 | 19,87 | 0,45 | 30 |
| PPL (word) | FP16 | 6,1358 | — | — | — | 1 |
| PPL (word) | INT8 | 6,2267 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 6,7338 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 6,6235 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 15,09 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 8,61 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 5,46 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 5,44 | — | — | — | 1 |

### Qwen3-8B-Base

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 62,06 | 60,98 | 63,92 | 2,94 | 30 |
| TTFT (ms) | INT8 | 261,51 | 257,65 | 263,71 | 6,06 | 30 |
| TTFT (ms) | INT4-AWQ | 77,85 | 77,50 | 78,75 | 1,25 | 30 |
| TTFT (ms) | INT4-GPTQ | 76,75 | 76,46 | 77,66 | 1,19 | 30 |
| Tokens/s | FP16 | 17,87 | 17,66 | 18,04 | 0,38 | 30 |
| Tokens/s | INT8 | 5,01 | 4,95 | 5,09 | 0,14 | 30 |
| Tokens/s | INT4-AWQ | 13,56 | 13,46 | 13,77 | 0,31 | 30 |
| Tokens/s | INT4-GPTQ | 13,94 | 13,81 | 14,10 | 0,29 | 30 |
| PPL (word) | FP16 | 6,9945 | — | — | — | 1 |
| PPL (word) | INT8 | 7,0269 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 7,9838 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 7,2795 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 15,39 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 9,20 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 5,81 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 5,79 | — | — | — | 1 |

## Testes de hipótese

### Meta-Llama-3-8B

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=120): H=108,871, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -6,647 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -6,648 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -6,648 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=120): H=110,937, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

### Qwen3-8B-Base

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=120): H=104,055, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=120): H=107,107, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

## Variação relativa de PPL vs FP16 (limiar de referência: 3%)

### Meta-Llama-3-8B

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 6,1358 | +0,0% | ✅ |
| INT8 | 6,2267 | +1,5% | ✅ |
| INT4-AWQ | 6,7338 | +9,7% | ❌ |
| INT4-GPTQ | 6,6235 | +7,9% | ❌ |

### Qwen3-8B-Base

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 6,9945 | +0,0% | ✅ |
| INT8 | 7,0269 | +0,5% | ✅ |
| INT4-AWQ | 7,9838 | +14,1% | ❌ |
| INT4-GPTQ | 7,2795 | +4,1% | ❌ |

## Matriz de trade-off e Pareto

### Meta-Llama-3-8B

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 6,1358 | 15,09 | 25,87 | ✅ | — |
| INT8 | 6,2267 | 8,61 | 7,04 | ✅ | — |
| INT4-AWQ | 6,7338 | 5,46 | 18,79 | ❌ | INT4-GPTQ |
| INT4-GPTQ | 6,6235 | 5,44 | 19,61 | ✅ | — |

### Qwen3-8B-Base

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 6,9945 | 15,39 | 17,87 | ✅ | — |
| INT8 | 7,0269 | 9,20 | 5,01 | ✅ | — |
| INT4-AWQ | 7,9838 | 5,81 | 13,56 | ❌ | INT4-GPTQ |
| INT4-GPTQ | 7,2795 | 5,79 | 13,94 | ✅ | — |

