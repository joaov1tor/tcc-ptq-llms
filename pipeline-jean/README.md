# `pipeline-jean/` — parte do Jean no benchmark PTQ 8B

Pasta nova no repo, isolada. Nada fora dela foi alterado. A junção com os
scripts e tabelas do João fica com o João; aqui ficam os arquivos para ele
decidir o que levar para o relatório.

Cumpre dois cartões do Trello (Stage 4):

1. **Publicar scripts, parâmetros e dados brutos** → `src/`, `scripts/`, `notebooks/`, `tests/`, `PARAMETROS.md`, `resultados/meu/`, `resultados/dele/`.
2. **Análise estatística e matriz de trade-off** → `resultados/analise_estatistica/` (mediana + IQR, Kruskal-Wallis, Mann-Whitney + Holm, Cliff's δ, Pareto, Δ PPL vs limiar de 3%).

## O que tem aqui

```
pipeline-jean/
├── README.md                — esta página
├── PARAMETROS.md            — tabela única de parâmetros (perfis, seeds, calibração, quantização, PPL, Etapa 6)
├── requirements-ppl.txt     — dependências
├── src/                     — pacote (bench + ppl): harness, stats, adaptador, geração, loading…
├── scripts/                 — CLIs (`run_benchmark.py`, `analyze_stats.py`, `compare_gpus.py`, `run_perplexity.py`…)
├── tests/                   — pytest (40 casos)
├── notebooks/               — `perplexity_colab.ipynb` e `colega_replica_colab.ipynb`
├── docs/                    — `wikitext2-versao-e-tokenizacao.md`
├── prompts/                 — `generation_v1.json`
├── resultados/
│   ├── dele/
│   │   ├── dgx/             — JSONs do DGX (cópia dos que já estão em `resultados/` na raiz, para a pasta ser self-contained)
│   │   └── a100_replica/    — JSONs do mesmo pipeline do João rodados numa A100 (Colab, 04/nov)
│   ├── meu/
│   │   └── colab_full_20261029/   — JSONs crus + `env.json` + `run.log` + `summary.md|csv`
│   └── analise_estatistica/
│       ├── dele_dgx/                — Etapa 6 sobre os dados do João (DGX)
│       ├── dele_a100/               — Etapa 6 sobre a réplica A100
│       ├── meu_a100/                — Etapa 6 sobre o full run do Jean
│       └── comparacao_dgx_vs_a100/  — Mann-Whitney DGX × A100 (16 comparações + Holm)
└── comparacao/
    └── dgx_vs_a100.html     — página de comparação (export do artefato do Jean)
```

## Quickstart

```bash
cd pipeline-jean
pip install -r requirements-ppl.txt

# Reproduzir a análise estatística do full run do Jean:
python scripts/analyze_stats.py \
    --source ours \
    --run-dir resultados/meu/colab_full_20261029 \
    --out-dir resultados/analise_estatistica/meu_a100

# Reproduzir a análise sobre os JSONs do DGX do João:
python scripts/analyze_stats.py \
    --source drive \
    --run-dir resultados/dele/dgx \
    --out-dir resultados/analise_estatistica/dele_dgx

# Comparar DGX × A100 para o mesmo pipeline do João:
python scripts/compare_gpus.py \
    --left-dir resultados/dele/dgx \
    --right-dir resultados/dele/a100_replica \
    --left-label DGX --right-label A100 \
    --source drive \
    --out-dir resultados/analise_estatistica/comparacao_dgx_vs_a100

# Testes:
pytest -q
```

## Protocolo: pipeline do Jean × pipeline do João

| | Jean | João |
|---|---|---|
| CLI | `scripts/run_benchmark.py` (profile `full`) | `scripts/ppl_wikitext2.py` + `scripts/bench_cost.py` |
| PPL | `word_perplexity` do harness (`lm-eval` 0.4.13, metadata 2.0) | `token_perplexity` |
| Janela | 2048 tokens, sem sobreposição | 2048 tokens, sem sobreposição |
| Dataset | `EleutherAI/wikitext_document_level / wikitext-2-raw-v1`, `test` | mesmo |
| Prompt de geração | `prompts/generation_v1.json` | prompts dele |
| n por célula (TTFT, tok/s) | `reps × n_prompts = 10 × 10 = 100` | `3 rodadas × 10 reps = 30` |
| Calibração INT4 | C4, 128 docs ≥ 8000 chars, seed 0, bloco 2048 | igual |
| Dtype compute | `float16` em FP16/INT8 e nas INT4 do harness | igual |

As duas convenções geram PPL/mem comparáveis entre si e tok/s/TTFT comparáveis
após agregar por mediana — o `scripts/analyze_stats.py` sabe ler os dois
formatos (`--source ours` ou `--source drive`).

## O que o pipeline do João diz nas duas GPUs

Mesmo código, duas máquinas: DGX Spark (GB10, medido pelo colega) × Colab
A100-SXM4-40GB (réplica em 04/nov). 16 comparações (2 modelos × 4 precisões × 2
métricas), MWU + Holm:

- **PPL idêntica**, maior diferença 0,78% (GPTQ Llama).
- **Memória idêntica**, diferença máxima 0,03 GiB.
- **Velocidade inverte**: no DGX o INT4 ganha do FP16 em 2,7–2,9×; na A100 o
  FP16 é o mais rápido. Nas 16 comparações, Holm rejeita H0 com
  `p_holm = 2,4×10⁻¹⁰` e `|δ| = 1,0` em todas.

Hipótese em aberto: DGX limitado por banda (~200 GB/s efetivos, perto do teto
de ~273 GB/s), A100 limitada por overhead de kernel (banda efetiva no INT4 em
83–112 GB/s, 5–7% do teto de ~1,5 TB/s). Teste proposto: repetir na A100 com
batch maior. Mais detalhe em `comparacao/dgx_vs_a100.html`.

## Meta de 4 GB — não atingida

A reunião do Gate 3 (28/10) pediu para documentar. Menores picos de memória
observados:

- **5,44 GiB** — Llama INT4-GPTQ na A100 (réplica).
- **5,46 GiB** — Llama INT4-GPTQ no DGX (código do João).
- 8,28 GiB — Llama INT4-GPTQ no pipeline do Jean (loader próprio, A100). Em
  investigação: a diferença de ~3 GiB para o pipeline do João vem de dtype de
  compute e/ou `device_map` (não testado).

Mesmo assim, INT4 economiza 62–64% sobre o FP16 (~15 GiB → ~5,5 GiB), o que é o
resultado para o §Resultados do relatório.

## Observações para a junção

- O `scripts/run_benchmark.py` do Jean escreve em `results/` (gitignored); o
  Drive guarda os JSONs crus. Os `resultados/meu/colab_full_20261029/` aqui são
  uma cópia desses JSONs para o PR ser self-contained.
- Os notebooks do Jean clonam o repo `jeanjonatas/tcc-ptq-ppl` com
  `GITHUB_TOKEN` (ambos repos privados). Depois do merge, apontar para
  `pipeline-jean/` do próprio repo é possível — fica a critério do João.
- Nada em `src/bench/stats.py` foi alterado em relação ao que já está no repo
  `jeanjonatas/tcc-ptq-ppl`; o script novo é `scripts/compare_gpus.py`, que só
  usa funções já existentes.
