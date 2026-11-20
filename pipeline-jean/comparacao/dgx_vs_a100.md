# DGX × A100 — mesmo pipeline, duas GPUs

Comparativo unificado entre as medições do colega no **DGX Spark (GB10)** e uma
réplica do mesmo código numa **Colab A100-SXM4-40GB**. Script, modelos,
precisões, seed e protocolo de PPL idênticos; só a GPU (e as versões de
software que vêm com cada ambiente) mudou.

Fonte dos dados: `resultados/dele/dgx/` (originais do colega) e
`resultados/dele/a100_replica/` (medido em 04/nov/2026, Colab Pro).

## O que é igual e o que é diferente

| | DGX Spark | Colab A100 |
|---|---|---|
| GPU | NVIDIA GB10 (LPDDR5X, ~273 GB/s) | NVIDIA A100-SXM4-40GB (HBM2e, ~1,5 TB/s) |
| torch / transformers / Python | 2.14 / 5.17 / 3.12 | 2.11 / 5.22 / 3.13 |
| Processos concorrentes na GPU | sim (registrado no diário dele) | não |

**Igual nas duas:** `ppl_wikitext2.py` (PPL token-level, janelas de 2048 sem
sobreposição), `bench_cost.py` (prompt de 512 tokens, 3 warmups, 10 repetições,
geração gulosa), 3 rodadas por célula = **n = 30**, seed 0, FP16 como dtype de
compute. Os modelos INT8/AWQ/GPTQ foram quantizados **de novo em cada máquina**
(mesma calibração C4, 128 docs, seed 0).

## Meta-Llama-3-8B

| Precisão | PPL DGX | PPL A100 | \|Δ PPL\| | Mem DGX (GiB) | Mem A100 (GiB) | TTFT DGX (ms) | TTFT A100 (ms) | Tok/s DGX | Tok/s A100 | A100/DGX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 6,1359 | 6,1358 | 0,00% | 15,11 | 15,09 | 140,5 | 48,8 | 13,57 | 25,87 | **1,91×** |
| INT8 | 6,2278 | 6,2267 | 0,02% | 8,63 | 8,61 | 168,9 | 208,9 | 18,14 | 7,04 | 0,39× |
| INT4-AWQ | 6,7248 | 6,7338 | 0,13% | 5,49 | 5,46 | 122,3 | 70,6 | 36,30 | 18,79 | 0,52× |
| INT4-GPTQ | 6,6754 | 6,6235 | 0,78% | 5,46 | **5,44** | 121,1 | 69,7 | 36,51 | 19,61 | 0,54× |

## Qwen3-8B-Base

| Precisão | PPL DGX | PPL A100 | \|Δ PPL\| | Mem DGX (GiB) | Mem A100 (GiB) | TTFT DGX (ms) | TTFT A100 (ms) | Tok/s DGX | Tok/s A100 | A100/DGX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 6,9946 | 6,9945 | 0,00% | 15,41 | 15,39 | 162,8 | 62,1 | 11,22 | 17,87 | **1,59×** |
| INT8 | 7,0299 | 7,0269 | 0,04% | 9,22 | 9,20 | 221,6 | 261,5 | 15,21 | 5,01 | 0,33× |
| INT4-AWQ | 7,9623 | 7,9838 | 0,27% | 5,83 | 5,81 | 133,2 | 77,9 | 32,87 | 13,56 | 0,41× |
| INT4-GPTQ | 7,3008 | 7,2795 | 0,29% | 5,81 | 5,79 | 131,5 | 76,8 | 33,01 | 13,94 | 0,42× |

## O que muda com a GPU — e o que não muda

- **PPL praticamente não muda.** A maior diferença entre as duas GPUs é 0,78%
  (GPTQ Llama); nas demais fica em 0,3% ou menos. Esse ruído vem de
  requantizar do zero em cada máquina (GPTQ e AWQ), não da avaliação. Quem
  decide "se o INT4 passa em 3%" é o método, não o hardware.
- **Memória praticamente não muda.** Diferença máxima de 0,03 GiB. INT4
  economiza 62–64% e INT8 40–43% sobre o FP16 nas duas GPUs. **Meta de 4 GB não
  é atingida**: menor pico 5,44 GiB (Llama INT4-GPTQ, A100) e 5,46 GiB (DGX).
- **Velocidade inverte.** No DGX, INT4 é 2,7–2,9× mais rápido que o FP16 e o
  INT8 ganha 34–36%. Na A100, o FP16 é o mais rápido e todas as quantizações
  perdem: INT8 cai 72–73%, INT4 cai 22–27%. O TTFT do FP16 também cai por 2,6–3×
  na A100 (prefill mais rápido).

## Análise estatística (Etapa 6)

Teste pareado **Mann-Whitney U** bilateral (duas amostras independentes de n =
30, nenhum pressuposto de normalidade), correção **Holm-Bonferroni** sobre as
16 comparações (2 modelos × 4 precisões × 2 métricas). Tamanhos de efeito:
**Cliff's δ** (ordinal, robusto a outliers) e `r = |Z|/√N`.

| Métrica | Comparações | U = 0 em | p (Holm) | Cliff's δ | r = \|Z\|/√N |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 8 | 8/8 | 2,4×10⁻¹⁰ | \|δ\| = 1,00 | ~0,86 |
| Tokens/s | 8 | 8/8 | 2,4×10⁻¹⁰ | \|δ\| = 1,00 | ~0,86 |

Leitura: `U = 0` significa que **nenhuma das 30 medidas do DGX se sobrepõe às
30 da A100**, para nenhuma célula. Holm rejeita H0 em todas as 16 comparações
com α = 0,05. O sinal do δ indica quem é maior: DGX ganha em tok/s para
INT8/INT4, A100 ganha em tok/s para FP16 e em TTFT para FP16/INT4.

Detalhes por célula em `resultados/analise_estatistica/comparacao_dgx_vs_a100/stats.md`.
Análises separadas de cada run estão em `resultados/analise_estatistica/{dele_dgx,dele_a100}/stats.md`
(Kruskal-Wallis entre precisões, MWU FP16 × quantizada com Holm, Pareto).

## Hipótese para a inversão de velocidade

> ⚠️ **Hipótese redigida por IA (Claude Opus).** Baseada nos números observados
> e em características conhecidas das duas GPUs. **Não foi validada
> experimentalmente**; é uma proposta para o relatório, não uma conclusão.

Na decodificação autoregressiva, cada token lê todos os pesos uma vez, então o
tempo por token é limitado ou pela banda de memória (ler os bytes) ou pelo
overhead de kernel (lançar e sincronizar as GEMMs/atenções). Estimando a banda
efetiva como `peso_na_GPU (GiB) × 1,0737 × tokens/s`:

| Precisão | Banda DGX (GB/s) | % de 273 GB/s | Banda A100 (GB/s) | % de 1,5 TB/s |
|---|---:|---:|---:|---:|
| Llama FP16 | 218 | ~80% | 415 | ~28% |
| Llama INT8 | 165 | ~60% | 64 | ~4% |
| Llama INT4-GPTQ | 208 | ~76% | 112 | ~7% |
| Qwen FP16 | 184 | ~67% | 293 | ~20% |
| Qwen INT4-GPTQ | 200 | ~73% | 85 | ~6% |

A leitura é que **o DGX está perto do teto de banda em todas as precisões**
(60–80% da LPDDR5X), então reduzir bytes com INT4 acelera, e o ganho (2,7–2,9×)
acompanha a razão de bytes (2,7–2,8×). Já **a A100 tem banda sobrando**: o INT4
usa menos de 7% do teto da HBM2e. Com banda ociosa, o gargalo passa a ser outra
coisa — a hipótese é **overhead de lançamento de kernel e de dequantização**,
que o INT4/INT8 pagam em cada camada e o FP16 não, o que explicaria o FP16
vencer na A100 e perder no DGX.

O INT8 do `bitsandbytes` (LLM.int8) tem ainda uma segunda fonte de custo: a
decomposição por outliers de ativação, que roda em cada passo e pesa mais na
A100 (banda efetiva de ~4% do teto) do que no DGX (~60%). Isso bate com o TTFT
do INT8 quadruplicar na A100 frente ao FP16.

**Caveats.** Cada GPU foi medida em uma sessão, com versões diferentes de
torch (2.14 × 2.11), transformers (5.17 × 5.22) e Python (3.12 × 3.13), e no
DGX havia outros processos na GPU durante parte das medições. O teste
estatístico separa as duas distribuições; não separa efeito de GPU de efeito
de versão.

**Como testar.** Repetir a geração na A100 com batch maior: se o INT4 passar a
ganhar do FP16 quando há mais trabalho por lançamento de kernel, a hipótese de
overhead fica mais forte. Para isolar as versões, rodar a mesma stack de
torch/transformers nas duas máquinas.

## Resumo para o relatório

- **Qualidade (PPL) é propriedade do método, não do hardware.** As duas GPUs
  dão PPL igual até a terceira casa. Só o INT8 passa no limiar de 3%.
- **Memória é a mesma nas duas GPUs.** A meta de 4 GB não é atingida; o menor
  pico é 5,44 GiB (Llama INT4-GPTQ na A100).
- **Velocidade depende da GPU.** Recomendação de precisão não é universal: no
  DGX INT4 é o melhor; na A100 o FP16 é o melhor. As 16 comparações MWU
  rejeitam H0 com `p_holm = 2,4×10⁻¹⁰` e `|δ| = 1,0` — a diferença é real.
- **Hipótese (IA, não testada):** DGX é bandwidth-bound, A100 é overhead-bound.
