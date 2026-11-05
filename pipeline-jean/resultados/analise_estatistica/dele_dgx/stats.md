# Etapa 6 — análise estatística (drive)

- Diretório: `resultados/dele/dgx`
- Precisões encontradas: fp16, int8, int4-awq, int4-gptq
- Modelos: meta-llama/Meta-Llama-3-8B, Qwen/Qwen3-8B-Base
- α = 0.05

## Mediana e IQR por métrica, precisão e modelo

### Meta-Llama-3-8B

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 140,50 | 137,88 | 149,50 | 11,62 | 30 |
| TTFT (ms) | INT8 | 168,91 | 168,19 | 169,11 | 0,93 | 30 |
| TTFT (ms) | INT4-AWQ | 122,30 | 121,86 | 122,81 | 0,95 | 30 |
| TTFT (ms) | INT4-GPTQ | 121,06 | 120,94 | 121,55 | 0,61 | 30 |
| Tokens/s | FP16 | 13,57 | 13,40 | 13,60 | 0,20 | 30 |
| Tokens/s | INT8 | 18,14 | 17,94 | 18,50 | 0,56 | 30 |
| Tokens/s | INT4-AWQ | 36,30 | 36,21 | 36,41 | 0,20 | 30 |
| Tokens/s | INT4-GPTQ | 36,51 | 36,31 | 36,53 | 0,22 | 30 |
| PPL (word) | FP16 | 6,1359 | — | — | — | 1 |
| PPL (word) | INT8 | 6,2278 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 6,7248 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 6,6754 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 15,11 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 8,63 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 5,49 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 5,46 | — | — | — | 1 |

### Qwen3-8B-Base

| Métrica | Precisão | Mediana | Q1 | Q3 | IQR | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | FP16 | 162,79 | 161,09 | 170,15 | 9,07 | 30 |
| TTFT (ms) | INT8 | 221,62 | 220,89 | 222,45 | 1,56 | 30 |
| TTFT (ms) | INT4-AWQ | 133,23 | 132,92 | 133,60 | 0,68 | 30 |
| TTFT (ms) | INT4-GPTQ | 131,49 | 131,17 | 131,76 | 0,60 | 30 |
| Tokens/s | FP16 | 11,22 | 11,15 | 11,43 | 0,27 | 30 |
| Tokens/s | INT8 | 15,21 | 15,06 | 15,25 | 0,19 | 30 |
| Tokens/s | INT4-AWQ | 32,87 | 32,75 | 33,16 | 0,41 | 30 |
| Tokens/s | INT4-GPTQ | 33,01 | 32,83 | 33,04 | 0,21 | 30 |
| PPL (word) | FP16 | 6,9946 | — | — | — | 1 |
| PPL (word) | INT8 | 7,0299 | — | — | — | 1 |
| PPL (word) | INT4-AWQ | 7,9623 | — | — | — | 1 |
| PPL (word) | INT4-GPTQ | 7,3008 | — | — | — | 1 |
| Pico de memória (GiB) | FP16 | 15,41 | — | — | — | 1 |
| Pico de memória (GiB) | INT8 | 9,22 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-AWQ | 5,83 | — | — | — | 1 |
| Pico de memória (GiB) | INT4-GPTQ | 5,81 | — | — | — | 1 |

## Testes de hipótese

### Meta-Llama-3-8B

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=120): H=105,922, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 30,0 | -6,202 | <0,0001 | <0,0001 | 0,801 | -0,933 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=120): H=104,069, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |

### Qwen3-8B-Base

**TTFT (ms)**

- Kruskal-Wallis (k=4, df=3, N=120): H=111,473, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

**Tokens/s**

- Kruskal-Wallis (k=4, df=3, N=120): H=100,495, p=<0,0001 → diferença significativa (α=0.05).

- Mann-Whitney FP16 × quantizado (Holm-Bonferroni, α=0.05):

| Comparação | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 × INT8 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-AWQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| FP16 × INT4-GPTQ | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |

## Variação relativa de PPL vs FP16 (limiar de referência: 3%)

### Meta-Llama-3-8B

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 6,1359 | +0,0% | ✅ |
| INT8 | 6,2278 | +1,5% | ✅ |
| INT4-AWQ | 6,7248 | +9,6% | ❌ |
| INT4-GPTQ | 6,6754 | +8,8% | ❌ |

### Qwen3-8B-Base

| Precisão | PPL (word) | Δ% vs FP16 | ≤ 3% |
| --- | ---: | ---: | :-: |
| FP16 | 6,9946 | +0,0% | ✅ |
| INT8 | 7,0299 | +0,5% | ✅ |
| INT4-AWQ | 7,9623 | +13,8% | ❌ |
| INT4-GPTQ | 7,3008 | +4,4% | ❌ |

## Matriz de trade-off e Pareto

### Meta-Llama-3-8B

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 6,1359 | 15,11 | 13,57 | ✅ | — |
| INT8 | 6,2278 | 8,63 | 18,14 | ✅ | — |
| INT4-AWQ | 6,7248 | 5,49 | 36,30 | ❌ | INT4-GPTQ |
| INT4-GPTQ | 6,6754 | 5,46 | 36,51 | ✅ | — |

### Qwen3-8B-Base

| Precisão | PPL | Mem (GiB) | Tokens/s (med) | Pareto | Dominada por |
| --- | ---: | ---: | ---: | :-: | --- |
| FP16 | 6,9946 | 15,41 | 11,22 | ✅ | — |
| INT8 | 7,0299 | 9,22 | 15,21 | ✅ | — |
| INT4-AWQ | 7,9623 | 5,83 | 32,87 | ❌ | INT4-GPTQ |
| INT4-GPTQ | 7,3008 | 5,81 | 33,01 | ✅ | — |

