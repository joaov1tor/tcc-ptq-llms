# Comparação DGX × A100 — mesmo pipeline, duas GPUs

- Esquerda (DGX): `resultados/dele/dgx`
- Direita (A100): `resultados/dele/a100_replica`
- Teste: Mann-Whitney U bilateral, α = 0.05, correção Holm-Bonferroni sobre o conjunto.
- Comparações: 16 (dois modelos × 4 precisões × 2 métricas).

**Resumo:** Holm rejeita H0 em 16/16 comparações (α = 0.05).

## TTFT (ms)

| Modelo | Precisão | Mediana DGX | Mediana A100 | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B-Base | FP16 | 162,83 | 62,17 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT8 | 221,76 | 261,52 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT4-AWQ | 133,24 | 77,96 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT4-GPTQ | 131,51 | 76,78 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | FP16 | 140,59 | 48,85 | 0,0 | 6,647 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT8 | 168,92 | 208,92 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT4-AWQ | 122,33 | 70,63 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT4-GPTQ | 121,09 | 69,70 | 0,0 | 6,647 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

## Tokens/s

| Modelo | Precisão | Mediana DGX | Mediana A100 | U | Z | p bruto | p (Holm) | r = \|Z\|/√N | Cliff's δ | efeito |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B-Base | FP16 | 11,22 | 17,87 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT8 | 15,21 | 5,01 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT4-AWQ | 32,87 | 13,58 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Qwen3-8B-Base | INT4-GPTQ | 33,01 | 13,94 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | FP16 | 13,57 | 25,91 | 0,0 | -6,646 | <0,0001 | <0,0001 | 0,858 | -1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT8 | 18,14 | 7,04 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT4-AWQ | 36,31 | 18,82 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |
| Meta-Llama-3-8B | INT4-GPTQ | 36,51 | 19,62 | 0,0 | 6,646 | <0,0001 | <0,0001 | 0,858 | 1,000 (large) | rejeita H0 |

