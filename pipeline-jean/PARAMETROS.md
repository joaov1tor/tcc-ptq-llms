# Parâmetros do experimento

Tabela única de parâmetros do pipeline do Jean. Os valores foram lidos do código
(`scripts/run_benchmark.py`, `src/ppl/*`, `src/bench/*`, `prompts/`,
`docs/wikitext2-versao-e-tokenizacao.md`) e dos `env.json` dos runs.

Cumpre o cartão **"Publicar scripts, parâmetros e dados brutos"** (Stage 4).

## Perfis de execução (`scripts/run_benchmark.py` — `PROFILES`)

| Perfil | Modelos | Precisões | n_prompts | reps | warmup | max_new_tokens | max_length (PPL) | max_docs | gptq_max_blocks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke` | Qwen2.5-0.5B | fp16, int8 | 1 | 1 | 1 | 16 | 128 | 2 | 2 |
| `colab` | Qwen2.5-1.5B | fp16, int8, int4-awq, int4-gptq | 5 | 5 | 2 | 64 | 2048 | 8 | — |
| `full` | Meta-Llama-3-8B, Qwen3-8B-Base | fp16, int8, int4-awq, int4-gptq | 10 | 10 | 3 | 128 | 2048 | todo o split | — |

O perfil `full` é o que gera `resultados/meu/colab_full_20261029/`. **n total por célula**:
`reps × n_prompts = 10 × 10 = 100` medidas de TTFT/tok/s. O pipeline do João faz
`3 rodadas × 10 repetições = 30` por célula. As duas convenções convivem.

## Seeds

| Parâmetro | Valor | Onde |
|---|---|---|
| `--seed` (geração/avaliação) | `0` | `scripts/run_benchmark.py:105` |
| `--calib-seed` (calibração C4) | `0` | `scripts/run_benchmark.py:107` |

## Calibração INT4 (AWQ e GPTQ) — `src/ppl/quantize.py`

Mesmo lock usado pelos scripts do João, para as curvas serem comparáveis.

| Parâmetro | Valor |
|---|---|
| Dataset | C4 (`en/c4-train.00000-of-01024.json.gz`) |
| Nº de documentos | `--calib-samples 128` (padrão) |
| Comprimento mínimo por doc | `CALIB_MIN_CHARS = 8000` caracteres (~2000 tokens) |
| Seed | `--calib-seed 0` |
| Tamanho do bloco | `calibration_concat_size = 2048` tokens |
| Batch da quantização | `1` |

## Quantização

| Precisão | Backend | Parâmetros |
|---|---|---|
| FP16 | `transformers.AutoModelForCausalLM` | `dtype=torch.float16` (`src/ppl/loading.py:_load_fp16`) |
| INT8 | `bitsandbytes` via Transformers | `BitsAndBytesConfig(load_in_8bit=True)`, `dtype=torch.float16`, `device_map=<device>` (`src/ppl/loading.py:_load_bnb_int8`) |
| INT4-AWQ | `gptqmodel` | `QuantizeConfig(method="awq", bits=4, group_size=128, sym=False)` (`src/ppl/quantize.py:50`) |
| INT4-GPTQ | `gptqmodel` | `QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)` (`src/ppl/quantize.py:52`) |

Os INT4 saem no disco como checkpoints prontos para o `GPTQModel.load(..., device_map="auto")`.

## Perplexidade (`src/ppl/evaluate.py`, `docs/wikitext2-versao-e-tokenizacao.md`)

| Parâmetro | Valor |
|---|---|
| Dataset | `EleutherAI/wikitext_document_level` / `wikitext-2-raw-v1` |
| Campo | `page` |
| Split | `test` |
| Métrica | `word_perplexity` (harness `lm-eval` 0.4.13, metadata 2.0) |
| Janela | `DEFAULT_MAX_LENGTH = 2048` tokens, sem sobreposição (`src/ppl/protocol.py:rolling_windows`) |
| Prefixo | `bos_token_id` do tokenizer (fallback: `eos_token_id`) |

Diferença do protocolo do João: o script dele (`scripts/ppl_wikitext2.py`) usa
`token_perplexity`, normalizando pela quantidade de tokens decodificados. Para PPL
ambos aceitam leituras compatíveis até 4 casas; os valores reportados aqui são o
`word_perplexity` do harness (pipeline do Jean) e o `token_perplexity` do ppl do João
nas tabelas lado a lado.

## Prompt de geração — `prompts/generation_v1.json`

| Parâmetro | Valor |
|---|---|
| Arquivo | `prompts/generation_v1.json` (SHA do conteúdo registrado em cada `env.json`) |
| Nº de prompts | `n_prompts` do perfil (10 no `full`) |
| Decodificação | gulosa (`do_sample=False`) |
| Comprimento fixo de saída | `max_new_tokens = min_new_tokens` (`src/bench/cost.py:167-168`) |

Amostras são descartadas: `warmup` iterações antes da coleta (`src/bench/cost.py:201-204`).

## Estatística — Etapa 6 (`src/bench/stats.py`, `scripts/analyze_stats.py`)

| Parâmetro | Valor |
|---|---|
| α | `0.05` (`scripts/analyze_stats.py:--alpha`, padrão) |
| Baseline | `fp16` (`BASELINE` em `analyze_stats.py:33`) |
| Limiar de referência da Δ PPL | `PPL_THRESHOLD_PCT = 3.0` (`src/bench/report.py:17`) |
| Teste global | Kruskal-Wallis entre as 4 precisões (por modelo e métrica) |
| Teste pareado | Mann-Whitney U bilateral com correção de empates |
| Correção para múltiplas comparações | Holm-Bonferroni |
| Tamanhos de efeito | Cliff's δ; `r = |Z|/√N` (critérios de Cohen: 0,1 / 0,3 / 0,5) |
| n mínimo por grupo para MWU | `MIN_N_FOR_MWU = 5` (`analyze_stats.py:35`) |
| Pareto | minimizar PPL e memória, maximizar tokens/s (`src/bench/stats.py:pareto_front`) |

Nenhum teste da Etapa 6 exige `scipy` ou `numpy`: tudo é `stdlib` (`math.erfc`,
Wilson-Hilferty para a sobrevida do χ²).

## Ambiente

Valores do `env.json` de cada run (os números abaixo são os do `meu/colab_full_20261029`):

| Chave | Valor |
|---|---|
| Python | `3.13.16` |
| torch | `2.11.0+cu130` |
| transformers | `5.20.2` |
| bitsandbytes | `0.50.2` |
| gptqmodel | (em `requirements-ppl.txt`) |
| GPU | `NVIDIA A100-SXM4-40GB` (40 GiB, driver 580.82.07, compute 8.0) |

As versões do DGX do João e da réplica A100 ficam em
`resultados/dele/dgx/*.json` e `resultados/dele/a100_replica/*.json` (campo `gpu`,
`torch`, `transformers`, `python` em cada arquivo).

## Referências cruzadas

- Dependências: `requirements-ppl.txt`
- Código: `src/bench/`, `src/ppl/`, `scripts/`
- Documentação longa do protocolo de PPL: `docs/wikitext2-versao-e-tokenizacao.md`
- Dados brutos: `resultados/meu/colab_full_20261029/`, `resultados/dele/dgx/`, `resultados/dele/a100_replica/`
- Análise estatística: `resultados/analise_estatistica/`
