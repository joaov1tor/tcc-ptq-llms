# Etapa 6 — análise estatística (ours)

- Diretório: `resultados/meu/colab_full_20261029`
- Precisões encontradas: fp16, int8, int4-awq, int4-gptq
- Modelos: Qwen/Qwen3-8B-Base, meta-llama/Meta-Llama-3-8B
- α = 0.05

## Mediana e IQR por métrica, precisão e modelo

### Qwen3-8B-Base

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 56,76 | 56,32 | 57,32 | 1,00 | 100 |
| TTFT (ms) | INT8 | 235,94 | 234,17 | 237,79 | 3,62 | 100 |
| TTFT (ms) | INT4-AWQ | 71,58 | 70,99 | 72,42 | 1,43 | 100 |
| TTFT (ms) | INT4-GPTQ | 68,00 | 67,60 | 68,66 | 1,06 | 100 |
| Tokens/s | FP16 | 18,86 | 18,77 | 18,93 | 0,16 | 100 |
| Tokens/s | INT8 | 5,28 | 5,21 | 5,34 | 0,12 | 100 |
| Tokens/s | INT4-AWQ | 14,61 | 14,57 | 14,66 | 0,09 | 100 |
| Tokens/s | INT4-GPTQ | 15,58 | 15,54 | 15,62 | 0,08 | 100 |
| PPL (word) | FP16 | 10,9507 | — | — | — | 1 |
| PPL (word) | INT8 | 11,0593 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 13,2402 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 11,4565 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 15,29 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 8,95 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 9,16 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 9,19 | — | — | — | 1 |

### Meta-Llama-3-8B

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 40,74 | 40,45 | 41,09 | 0,64 | 100 |
| TTFT (ms) | INT8 | 186,24 | 184,51 | 188,36 | 3,86 | 100 |
| TTFT (ms) | INT4-AWQ | 52,87 | 52,39 | 53,35 | 0,96 | 100 |
| TTFT (ms) | INT4-GPTQ | 50,90 | 50,34 | 51,48 | 1,14 | 100 |
| Tokens/s | FP16 | 27,63 | 27,47 | 27,73 | 0,26 | 100 |
| Tokens/s | INT8 | 7,41 | 7,35 | 7,48 | 0,14 | 100 |
| Tokens/s | INT4-AWQ | 20,24 | 20,08 | 20,35 | 0,27 | 100 |
| Tokens/s | INT4-GPTQ | 21,40 | 21,26 | 21,58 | 0,32 | 100 |
| PPL (word) | FP16 | 7,7942 | — | — | — | 1 |
| PPL (word) | INT8 | 7,9167 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 8,6511 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 8,5778 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 14,99 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 8,65 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 8,65 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 8,28 | — | — | — | 1 |

## Testes de hipótese

### Qwen3-8B-Base

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=400): H=372,540, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=400): H=374,065, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |

### Meta-Llama-3-8B

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=400): H=365,232, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -12,216 | <0,0001 | <0,0001 | 0,864 | -1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=400): H=373,975, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 12,216 | <0,0001 | <0,0001 | 0,864 | 1,000 (large) | rejeita H0 |

## Variação relativa de PPL vs FP16 (limiar de referência: 3%)

### Qwen3-8B-Base

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 10,9507 | +0,0% | ✅ |
| INT8 | 11,0593 | +1,0% | ✅ |
| INT4-AWQ | 13,2402 | +20,9% | ❌ |
| INT4-GPTQ | 11,4565 | +4,6% | ❌ |

### Meta-Llama-3-8B

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 7,7942 | +0,0% | ✅ |
| INT8 | 7,9167 | +1,6% | ✅ |
| INT4-AWQ | 8,6511 | +11,0% | ❌ |
| INT4-GPTQ | 8,5778 | +10,1% | ❌ |

## Matriz de trade-off e Pareto

### Qwen3-8B-Base

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 10,9507 | 15,29 | 18,86 | ✅ | — |
| INT8 | 11,0593 | 8,95 | 5,28 | ✅ | — |
| INT4-AWQ | 13,2402 | 9,16 | 14,61 | ✅ | — |
| INT4-GPTQ | 11,4565 | 9,19 | 15,58 | ✅ | — |

### Meta-Llama-3-8B

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 7,7942 | 14,99 | 27,63 | ✅ | — |
| INT8 | 7,9167 | 8,65 | 7,41 | ✅ | — |
| INT4-AWQ | 8,6511 | 8,65 | 20,24 | ❌ | INT4-GPTQ |
| INT4-GPTQ | 8,5778 | 8,28 | 21,40 | ✅ | — |

