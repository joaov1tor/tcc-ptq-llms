# Benchmark de quantização pós-treino para modelos de linguagem abertos em ambientes de recursos limitados

Repositório de dados e código do Trabalho de Conclusão de Curso da Especialização em Inteligência Artificial Generativa, Universidade Federal de Goiás (UFG) e Centro de Competência Embrapii em Tecnologias Imersivas (AKCIT), 2026.

**Autores:** Jean Jônatas de Jesus Borba, João Paulo Barros e João Vitor Ribeiro Santos
**Orientadores:** Hidelberg Oliveira Albuquerque, Jacson Rodrigues Barbosa e Sérgio Teixeira de Carvalho

Este repositório é a fonte dos números citados no relatório técnico, no resumo expandido e no banner do trabalho. A versão dos dados usada no texto é o commit [`d672735`](https://github.com/joaov1tor/tcc-ptq-llms/commit/d672735bee4fd9c44eb3776426e65221f8553995).

## Como citar

> BORBA, Jean Jônatas de Jesus; BARROS, João Paulo; SANTOS, João Vitor Ribeiro. **tcc-ptq-llms**: benchmark de quantização pós-treino (PTQ) de LLMs. [S. l.]: GitHub, 2026. Disponível em: https://github.com/joaov1tor/tcc-ptq-llms.

```bibtex
@misc{borba2026tccptq,
  author       = {Borba, Jean J{\^o}natas de Jesus and Barros, Jo{\~a}o Paulo and Santos, Jo{\~a}o Vitor Ribeiro},
  title        = {tcc-ptq-llms: benchmark de quantiza{\c{c}}{\~a}o p{\'o}s-treino (PTQ) de LLMs},
  year         = {2026},
  howpublished = {\url{https://github.com/joaov1tor/tcc-ptq-llms}},
  note         = {Trabalho de Conclus{\~a}o de Curso, Especializa{\c{c}}{\~a}o em IA Generativa, UFG/AKCIT}
}
```

## Resumo

Avaliamos os efeitos da quantização pós-treinamento (PTQ) sobre a qualidade preditiva e o custo de inferência de dois modelos abertos de cerca de 8 bilhões de parâmetros, Meta-Llama-3-8B e Qwen3-8B-Base. Foram comparadas quatro configurações: FP16 (linha de base), INT8 com bitsandbytes (LLM.int8()), INT4-GPTQ e INT4-AWQ.

- **Qualidade:** perplexidade por token no split de teste do WikiText-2 raw v1.
- **Custo:** pico de memória, tempo até o primeiro token (TTFT) e vazão em tokens/s.
- **Plataformas:** NVIDIA DGX Spark (bancada principal) e réplica do mesmo pipeline numa NVIDIA A100-SXM4-40GB.

Resultados:
- Só o INT8 manteve o aumento de perplexidade dentro do limite de 3%: +1,50% no Llama e +0,50% no Qwen.
- O INT4-GPTQ (+8,79% e +4,38%) foi melhor que o INT4-AWQ (+9,60% e +13,83%).
- O INT4 reduziu o pico de memória em 62% a 64%, mas nenhuma configuração ficou abaixo da meta de 4 GiB; o menor pico foi 5,46 GiB.
- Na DGX Spark, o INT4 teve 2,7 a 2,9 vezes a vazão do FP16; na A100, o FP16 foi o mais rápido.

A escolha da configuração depende da combinação de modelo, método, hardware e pilha de software.

## Pergunta de pesquisa

Até que nível de quantização pós-treinamento o Meta-Llama-3-8B e o Qwen3-8B-Base podem ser comprimidos para execução em ambientes com recursos limitados, mantendo o aumento relativo da perplexidade em até 3% sobre a linha de base FP16, e quais são os efeitos correspondentes sobre memória, latência e vazão nas plataformas avaliadas?

## Protocolo experimental

| Elemento | Configuração |
|---|---|
| Modelos | `meta-llama/Meta-Llama-3-8B` e `Qwen/Qwen3-8B-Base` (Hugging Face Hub) |
| Linha de base | FP16 |
| INT8 | bitsandbytes LLM.int8(), sem calibração |
| INT4-GPTQ | gptqmodel, 4 bits, grupo 128, simétrico, `desc_act=False`, kernel Marlin |
| INT4-AWQ | gptqmodel (método AWQ), 4 bits, grupo 128, assimétrico, kernel Marlin |
| Calibração (INT4) | C4 (`allenai/c4`, `en/c4-train.00000-of-01024`), 128 documentos inteiros com ≥ 8.000 caracteres (~387 mil tokens), semente 0. Sensibilidade: 128 blocos de 2.048 tokens |
| Qualidade | WikiText-2 raw v1, split test, concatenado com `"\n\n"`, tokenizador nativo, janelas de 2.048 tokens sem sobreposição; PPL = exp(média do NLL por token). Llama: 288.627 tokens em 141 janelas; Qwen: 298.862 tokens em 146 janelas |
| Custo | Prompt de 512 tokens (split de validação do WikiText-2, para não usar o teste); 128 tokens gerados; decodificação gulosa; batch 1; 3 aquecimentos descartados; 10 repetições por série; 3 séries por configuração (n = 30) |
| Métricas de custo | Tamanho em disco; pico de `torch.cuda.max_memory_allocated` durante a geração (pesos, KV cache e ativações); TTFT = tempo de `generate(max_new_tokens=1)`; tokens/s = (N − 1) / (t[generate(N)] − t[generate(1)]) |
| Estatística | Mediana e IQR; Kruskal-Wallis entre as quatro precisões; Mann-Whitney U com correção de Holm-Bonferroni; tamanho de efeito r = \|Z\|/√N e delta de Cliff |

Fora do escopo: fine-tuning, quantização do KV cache (mantido em FP16), batch maior que 1 e avaliação de qualidade além da perplexidade.

## Ambiente

| | DGX Spark (principal) | A100 (réplica) |
|---|---|---|
| GPU | NVIDIA GB10 Grace Blackwell, compute 12.1 | NVIDIA A100-SXM4-40GB, compute 8.0 |
| Memória | 121 GiB unificada LPDDR5X (~273 GB/s) | 40 GiB HBM2e (~1,5 TB/s) |
| Sistema | Ubuntu 24.04.4, driver 580.142, CUDA 13.0 | Google Colab Pro, driver 580.82.07 |
| Python / PyTorch / Transformers | 3.12.13 / 2.14.0+cu130 / 5.17.0 | 3.13 / 2.11 / 5.22 |
| Quantização | gptqmodel 7.6.0, bitsandbytes 0.50.2 (`logs/setup_quant.log`) | bitsandbytes 0.50.2 |

Na DGX Spark a memória é unificada entre CPU e GPU; os picos são "memória reportada" pelo PyTorch, não VRAM dedicada. A bancada principal era compartilhada com outros serviços; cada JSON de custo registra os processos concorrentes em `gpu_other_procs_before`.

## Resultados principais (DGX Spark)

| Modelo | Precisão | PPL | ΔPPL | Pico (GiB) | TTFT (ms) | Vazão (tok/s) |
|---|---|---:|---:|---:|---:|---:|
| Meta-Llama-3-8B | FP16 | 6,1359 | 0,00% | 15,11 | 140,32 | 13,58 |
| Meta-Llama-3-8B | INT8 | 6,2278 | +1,50% | 8,63 | 168,85 | 18,14 |
| Meta-Llama-3-8B | INT4-GPTQ | 6,6754 | +8,79% | 5,46 | 121,03 | 36,52 |
| Meta-Llama-3-8B | INT4-AWQ | 6,7248 | +9,60% | 5,49 | 122,00 | 36,37 |
| Qwen3-8B-Base | FP16 | 6,9946 | 0,00% | 15,41 | 162,79 | 11,22 |
| Qwen3-8B-Base | INT8 | 7,0299 | +0,50% | 9,22 | 221,67 | 15,21 |
| Qwen3-8B-Base | INT4-GPTQ | 7,3008 | +4,38% | 5,81 | 131,34 | 33,02 |
| Qwen3-8B-Base | INT4-AWQ | 7,9623 | +13,83% | 5,83 | 133,18 | 32,87 |

TTFT e vazão são a mediana das medianas das três séries (Tabela 3 do relatório). A comparação DGX × A100 (Tabela 4 e Apêndice A.8) usa a mediana das 30 observações juntas, por isso difere nos décimos. Os números podem ser recalculados com `python scripts/confere_numeros.py`.

## Onde está cada número do relatório

| Relatório | Arquivos |
|---|---|
| Tabela 1 (perplexidade, DGX) | `resultados/<modelo>_<precisao>.json` (campo `ppl`), com `<modelo>` = `llama3-8b` ou `qwen3-8b-base` e `<precisao>` = `fp16`, `int8`, `gptq4` ou `awq4` |
| Tabelas 2 e 3, Apêndices A.2 a A.4 (memória, TTFT e vazão, DGX) | `resultados/<modelo>_<precisao>_cost.json`, `..._cost_r2.json` e `..._cost_r3.json` (séries r1, r2 e r3; valores brutos em `ttft_ms_raw` e `tokens_per_s_raw`) |
| Tabela A5 (sensibilidade da calibração) | `resultados/<modelo>_{gptq4,awq4}_c2048.json` |
| A.5 (AWQ simétrico, Qwen) | `resultados/qwen3-8b-base_awq4_sym.json` e `..._sym_cost.json` |
| Smoke tests (excluídos da análise) | `resultados/smoke_qwen3_{gptq,int8}.json` |
| Tabela 4 (DGX × A100) | DGX: `pipeline-jean/resultados/dele/dgx/` (cópia de `resultados/`); A100: `pipeline-jean/resultados/dele/a100_replica/` (mesmos nomes de arquivo) |
| A.8 (Mann-Whitney DGX × A100) | `pipeline-jean/resultados/analise_estatistica/comparacao_dgx_vs_a100/` |
| Kruskal-Wallis, Mann-Whitney FP16 × quantizado, Pareto | `pipeline-jean/resultados/analise_estatistica/{dele_dgx,dele_a100}/` (`stats.md`, `stats.csv`, `trade_off.md`) |
| Consolidação em planilha | `resultados_benchmark_PTQ.xlsx` |
| Falhas e condições de medição | `DIARIO.md` e `logs/` (ex.: `logs/rerun_llama_fp16_r3.log`, série refeita após falta de memória) |

## Como reproduzir

Requer GPU NVIDIA com CUDA e acesso ao `meta-llama/Meta-Llama-3-8B` no Hugging Face (licença Meta Llama 3 Community). As versões usadas estão na seção Ambiente; a lista de dependências do pipeline de réplica está em `pipeline-jean/requirements-ppl.txt`.

```bash
# 1. Quantizar (a partir do diretório FP16 baixado do Hugging Face)
python scripts/quantize_int8.py models/Meta-Llama-3-8B models/Meta-Llama-3-8B-int8
python scripts/quantize_gptq.py models/Meta-Llama-3-8B models/Meta-Llama-3-8B-gptq4 --nsamples 128 --seed 0
python scripts/quantize_awq.py  models/Meta-Llama-3-8B models/Meta-Llama-3-8B-awq4  --nsamples 128 --seed 0

# 2. Perplexidade no WikiText-2 (FP16 e artefatos já quantizados)
python scripts/ppl_wikitext2.py --model models/Meta-Llama-3-8B --out resultados/llama3-8b_fp16.json
python scripts/ppl_wikitext2.py --model models/Meta-Llama-3-8B-gptq4 --quant pre --out resultados/llama3-8b_gptq4.json

# 3. Custo (repetir 3 vezes por configuração: _cost, _cost_r2, _cost_r3)
python scripts/bench_cost.py --model models/Meta-Llama-3-8B-gptq4 --quant pre --out resultados/llama3-8b_gptq4_cost.json

# 4. Estatística e comparação entre GPUs (ver pipeline-jean/README.md)
cd pipeline-jean && pip install -r requirements-ppl.txt && pytest
```

As filas usadas na DGX estão em `scripts/run_*.sh`; os notebooks da réplica na A100, em `pipeline-jean/notebooks/`.

## Estrutura do repositório

| Pasta | Conteúdo |
|---|---|
| `scripts/` | Quantização, perplexidade, custo, filas de execução e recálculo dos números |
| `resultados/` | 40 JSONs das medições na DGX Spark |
| `logs/` | Logs brutos das rodadas |
| `pipeline-jean/` | Pipeline de réplica na A100, análise estatística, 40 testes pytest e comparação DGX × A100 |
| `DIARIO.md` | Diário experimental, inclusive tentativas que falharam |
| `redacao/` | Rascunhos de Método e Resultados |
| `gate3/` | Roteiro da apresentação do Gate 3 |

## Limitações

- Dois modelos de uma mesma faixa de tamanho e duas plataformas; os resultados não se generalizam para outras famílias, GPUs ou versões de software.
- Qualidade medida só por perplexidade, que não equivale a qualidade de resposta em tarefas.
- A inversão de desempenho entre DGX Spark e A100 coincide com a troca de GPU, de versões de software e de sessão; a causa não foi isolada.
- Bancada principal compartilhada com outros serviços (registrado em cada JSON).
- Medições só com batch 1 e KV cache em FP16.

## Licenças

Os modelos avaliados seguem as próprias licenças: Meta Llama 3 Community License (Meta-Llama-3-8B) e Apache 2.0 (Qwen3-8B-Base). Os datasets são públicos: WikiText-2 (Merity et al., 2017) e C4 (Raffel et al., 2020).

## Agradecimentos

Ao Centro de Competência Embrapii em Tecnologias Imersivas (AKCIT) da UFG, pelo suporte durante a pesquisa, e à Data Agile e à TOTVS Brasil Central, pelo suporte ao hardware.

## Referências

- DETTMERS, T. et al. LLM.int8(): 8-bit matrix multiplication for transformers at scale. NeurIPS, 2022. https://arxiv.org/abs/2208.07339
- FRANTAR, E.; ASHKBOOS, S.; HOEFLER, T.; ALISTARH, D. GPTQ: accurate post-training quantization for generative pre-trained transformers. ICLR, 2023. https://arxiv.org/abs/2210.17323
- LIN, J. et al. AWQ: activation-aware weight quantization for on-device LLM compression and acceleration. Proceedings of Machine Learning and Systems, v. 6, p. 87-100, 2024.
- MERITY, S. et al. Pointer sentinel mixture models. ICLR, 2017. https://arxiv.org/abs/1609.07843
- RAFFEL, C. et al. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, v. 21, n. 140, 2020.
- YANG, A. et al. Qwen3 technical report. arXiv:2505.09388, 2025.
- ZHU, X. et al. A survey on model compression for large language models. TACL, v. 12, p. 1556-1577, 2024.
