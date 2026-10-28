# Diario de tentativas — TCC PTQ (inclui falhas)

## 2026-10-20 (DGX Spark, GB10, aarch64, CUDA 13, torch 2.14.0+cu130, transformers 5.17.0)
- venv .venv-quant criado: bitsandbytes 0.50.2, gptqmodel 7.6.0, llmcompressor 0.14.0 instalaram e importam (instalar nao prova que os kernels rodam).
- INT8 (bnb) Qwen3-8B-Base: smoke OK (8 janelas, PPL 6.0275, subconjunto, nao comparavel). Rodada completa: PPL 7.0299 vs FP16 6.9946 (+0.5%), disco 8.793 GiB, TTFT 221 ms (FP16 171), 15.25 tok/s (FP16 11.44).
- Condicao: outros processos usavam a GPU durante a medicao (whisper-server, pyannote, ollama, python3 de 4.9 GiB); constam no JSON (gpu_other_procs_before).
- Erro de processo: 1o envio do patch --quant falhou por SyntaxError (comentario dentro de chamada); nada rodou; corrigido.
- INT8 (bnb) rodada completa concluida 18:57. Llama 3 8B: PPL 6.1359 -> 6.2278 (+1.5%, 141 janelas); disco 14.958 -> 8.463 GiB; pico 15.108 -> 8.629 GiB; TTFT 150.3 -> 169.3 ms; 13.58 -> 18.14 tok/s.
- Qwen3-8B-Base: PPL 6.9946 -> 7.0299 (+0.5%, 146 janelas); disco 15.256 -> 8.793 GiB; pico 15.411 -> 9.225 GiB; TTFT 171.0 -> 221.3 ms (+29%); 11.44 -> 15.25 tok/s.
- Achado: INT8 piora TTFT nos dois modelos (custo de dequantizacao no prefill do bitsandbytes) e melhora tok/s (decodificacao limitada por banda).
- Log da rodada: sem erros reais; apenas avisos do PyTorch (enum KernelPreference/ScaleCalculationMode) e aviso de tokenizacao do texto inteiro (299078 > 131072 tokens) antes do janelamento, igual ao baseline FP16.
- Condicao de medicao: whisper-server, pyannote, ollama e um python3 (4.9 GiB) usavam a GPU durante as medicoes de custo (gpu_other_procs_before nos JSON). Ressalva para o texto final; considerar repetir em horario calmo.
- Pendente/nao medido: INT4 (GPTQ/AWQ), smoke test dos artefatos INT4, hardware de consumo (Avell, notebook JP), meta de 4 GB.

## 2026-10-20 — INT4 GPTQ (gptqmodel 7.6.0), Qwen3-8B-Base, smoke (16 amostras C4, G128, sim., desc_act=False)
- Quantizacao concluiu e salvou o artefato (models/smoke-qwen3-gptq na DGX).
- FALHA 1: carregar para PPL -> ImportError "Loading a GPTQ quantized model requires optimum". Causa: transformers exige o pacote optimum. Correcao: pip optimum 2.3.0.
- FALHA 2: ModuleNotFoundError Marlin torch.ops: "Ninja is required". O kernel Marlin e compilado por JIT (selecionado automaticamente). Correcao: instalar ninja e exportar CUDA_HOME=/usr/local/cuda e PATH com nvcc (CUDA 13.0 ja existe na DGX, fora do PATH).
- Resultado do smoke (8 janelas, NAO comparavel com FP16): PPL 6.3651, pico 8.58 GiB. Marlin JIT compila e roda na GB10.
- Decisao de fluxo (JV, 20/10): a DGX e maquina de execucao, nao guarda dados; resultados/logs sao puxados para o Avell/Drive apos cada etapa. DIARIO.md vive so aqui.

## 2026-10-20 — Calibracao C4 (registro exato)
- Fonte allenai/c4, en/c4-train.00000-of-01024.json.gz; seed 0; 128 documentos >= 8000 caracteres.
- gptqmodel (Qwen3-8B-Base, rodada completa): 386.925 tokens nao-padded; docs inteiros, ordenados por tamanho, sem concatenar/truncar em 2048 (calibration_concat_size vazio). Protocolo original do GPTQ: 128 segmentos de 2048 tokens (~262 mil tokens). DIFERENCA a declarar.
- Aviso da biblioteca: "Calibration dataset size should be more than 256. Current: 128." Mantido 128 (padrao do artigo).
- C4 so calibra; a metrica continua WikiText-2 test. Decisao de usar C4 e do assistente, nao do escopo aprovado pelo Berg; levar ao checkpoint de 30/10.

## 2026-10-20 — AWQ: escolha de ferramenta (registro)
- llmcompressor 0.14: AWQModifier antigo e apenas um adaptador (DeprecationWarning) para transform.AWQModifier + QuantizationModifier; saida em compressed-tensors, com risco de executar sem kernel INT4 no transformers (distorce custo). NAO usado; fica como alternativa documentada.
- Escolhido: gptqmodel method="awq", 4 bits, g128, assimetrico, calibracao C4 x128 (mesmo protocolo do GPTQ), mesmo carregador/kernels. Script quantize_awq.py (arquivo separado para nao alterar o GPTQ em execucao).
- run_awq.sh encadeado: espera run_gptq.sh terminar para nao sobrepor medicoes de custo na GPU.

## 2026-10-20 — Tempo de quantizacao (custo do processo, nao do modelo)
- GPTQ g128 (128 docs C4), Qwen3-8B-Base: ~17 min para quantizar na GB10 (19:26 -> ~19:43).
- AWQ (gptqmodel, method=awq) Qwen3-8B-Base: estimativa de ~1h23 so para quantizar (camada 15/35 aos 37 min); ~5x mais lento que o GPTQ nesta implementacao/plataforma. Dado de custo de producao do artefato a relatar.

## 2026-10-20 — AWQ (gptqmodel) Qwen3-8B-Base: PPL 7.9623 (+13.8% vs FP16 6.9946), 146 janelas
- Comparacao no mesmo modelo/protocolo: INT8 7.0299 (+0.5%), GPTQ 7.3008 (+4.4%), AWQ 7.9623 (+13.8%).
- Dentro da faixa publicada para INT4 no Qwen3 (+3% a +16%), mas no topo e ~3x pior que o GPTQ aqui. NAO ajustado para melhorar; registrado como medido.
- Hipoteses (NAO testadas): (1) porte do AWQ no gptqmodel difere do codigo dos autores; (2) calibracao com documentos inteiros de tamanho variavel em vez de blocos de 2048 tokens; (3) config assimetrica g128 e calibracao C4 em vez de Pile.
- Sensibilidade candidata: AWQ com blocos de 2048 (calibration_dataset_concat_size), AWQ simetrico, AWQ via llmcompressor.

## 2026-10-20 — Fila da madrugada (aprovada por JV: "ok")
- run_noite.sh na DGX, em espera: so inicia quando run_awq.sh terminar E existir o arquivo-sinal ~/tcc-ptq/NOITE_OK (criado por mim apos o /code-review, para que correcoes nao invalidem o que rodar de noite).
- Parte A: replicas r2 e r3 do custo das 8 configuracoes (FP16/INT8/GPTQ/AWQ x 2 modelos), com a GPU livre das nossas quantizacoes. Ressalva: um python3 que nao e nosso segue na GPU (registrado nos JSON).
- Parte B: sensibilidade da calibracao: GPTQ e AWQ com blocos de 2048 tokens (calibration_concat_size=2048), 90 documentos (~130 blocos, proximo dos 128 do artigo), so PPL.
- Correcao de registro: o parametro do quantize() e calibration_concat_size (eu havia citado calibration_dataset_concat_size, nome de metodo interno).
- Fuso: a DGX esta em UTC; meia-noite de Brasilia = 03:00 na DGX.

## 2026-10-20 — /code-review dos scripts (7 achados) e correcoes
Verificado contra codigo e dados antes de agir:
- Disco inflado por .bin+.safetensors: NAO se aplica (dirs FP16 so tem safetensors; 14.958 GiB e o esperado). Endurecido: usa so um formato.
- peak_mem_gb: e o pico DURANTE a geracao (pesos+KV+ativacoes), sem o transiente de carga; para artefatos salvos e o pico de servico correto. Esclarecido no docstring e adicionado peak_reserved_gb (max_memory_reserved, inclui cache do alocador). Medicoes r1 NAO tem esse campo; r2/r3 terao. Smoke: pico 5.811 igual ao anterior; reservado 5.924.
- tokens/s = (N-1)/(t_N - t_1) de duas chamadas generate(): limitacao metodologica (ruido nao se cancela; per-token timestamps seria mais robusto). IQR observado 0.01-0.17 tok/s mostra ruido pequeno. Adicionada protecao: rep com tn<=t1 e descartada e contada em tps_invalid_reps. Formula mantida para comparar com r1.
- Sensibilidade da calibracao mudava 2 variaveis (90 docs vs 128 e bloco 2048): CORRIGIDO para os MESMOS 128 docs (so muda o empacotamento em blocos de 2048; ~190 blocos, nao 128 como no artigo).
- Docstring copiado nos scripts AWQ ("python quantize_gptq.py") e comentario de 8000 caracteres: corrigidos. Adicionado assert de 128 docs (log das rodadas ja mostrava "128 docs").
- run_noite.sh reescrito (cd || exit, contador de falhas, nsamples 128); fila antiga derrubada antes de editar, pois editar script bash em execucao e perigoso.
- ACEITO sem mudar: (a) 4 scripts quantize_* quase iguais (mantidos como foram executados, para reprodutibilidade; consolidar ao publicar o repo); (b) --device vs device_map {"":0}: scripts exigem CUDA; (c) run_int8/gptq/awq.sh sem set -e: resultados conferidos manualmente, 0 Tracebacks nos logs.
- Nenhum numero ja medido muda. NOITE_OK criado; fila da madrugada iniciada.

## 2026-10-21 — Resultado da fila da madrugada (run_noite.sh, 23:07 UTC a 03:14 UTC)

- Terminou com `falhas=1`: Llama 3 8B FP16, réplica r3, falhou com `CUDA error: out of memory` em `cuDevicePrimaryCtxRetain` (23:33 UTC). Havia outros processos de terceiros na GPU (um `python3` com ~8 GiB, ollama, whisper). Reexecutada às 14:40 UTC, sem mais nada pesado concorrente: passou (`logs/rerun_llama_fp16_r3.log`). Registrar na metodologia que essa réplica foi medida em outro horário.
- Réplicas de custo (tok/s, mediana de cada rodada r1/r2/r3): Llama FP16 13,58/13,39/13,70; INT8 18,14/17,92/18,53; GPTQ 36,53/36,27/36,52; AWQ 36,28/36,37/36,38. Qwen3 FP16 11,44/11,22/11,15; INT8 15,25/15,05/15,21; GPTQ 33,02/32,82/33,05; AWQ 32,87/32,74/33,17. Variação entre rodadas pequena (<2%); a ordem entre as configurações não muda.
- Sensibilidade da calibração (blocos de 2048, 128 documentos, só PPL): Llama GPTQ 6,6847 (vs 6,6754) e AWQ 6,7338 (vs 6,7248); Qwen3 GPTQ 7,2995 (vs 7,3008) e AWQ 7,9463 (vs 7,9623). Diferença <0,3% em todos os casos: o protocolo de calibração do gptqmodel não explica o AWQ pior no Qwen3.
- Card "Redação: Método" movido para Em andamento, com comentário.

## 2026-10-21 — Teste adicional: AWQ simétrico no Qwen3-8B-Base

- Motivo: AWQ do Qwen3 (+13,8%) bem pior que o GPTQ (+4,4%); a calibração foi descartada como causa (diferença máx. 0,2%). Hipótese testada agora: o AWQ assimétrico (zero-point) do porte do gptqmodel.
- Mudança única em relação ao AWQ principal: `sym=True` (scripts/quantize_awq_sym.py); mesma calibração C4 x128 documentos inteiros, G128, 4 bits, seed 0. Fila em scripts/run_awq_sym.sh (quantiza, PPL, custo em série única). Iniciada 15:36 UTC; AWQ levou cerca de 1h+ antes. Resultado, seja qual for, entra na planilha e no guia.

## 2026-10-21 — Material de estudo e do checkpoint (sem GPU)

- `guia/perguntas_banca.md`: 16 perguntas prováveis da banca com resposta curta e limite honesto. Itens (CONFIRMAR) aguardam a conferência das referências.
- `berg/resumo_checkpoint.{html,docx,pdf}`: resumo de uma página para o checkpoint de 30/10 (resultados, negativos e 4 decisões pedidas ao orientador). Rascunho: ainda não enviado a ninguém.
- Conferência das referências marcadas CONFIRMAR: delegada a um agente de pesquisa; resultado ainda a aplicar ao rascunho do Método e ao guia.

## 2026-10-21 — AWQ simétrico no Qwen3-8B-Base e conferência das referências

- AWQ simétrico (sym=True, mesma calibração C4 x128): PPL 7,9620 contra 7,9623 do assimétrico (Δ +13,8% sobre o FP16 nos dois). Custo (série única): pico 5,839 GiB, 33,2 tok/s, TTFT 132,9 ms. A fila terminou 17:03 UTC sem falhas (~1h27 total). Verificado que a mudança foi aplicada: `zero_point: false` no config e `qzeros` constante no simétrico, 27.178 valores distintos no assimétrico (camada 0, down_proj). Conclusão: o zero-point NÃO explica o AWQ pior no Qwen3.
- Literatura: Zheng et al. (2025, arXiv 2505.02214), Tab. 3, Qwen3-8B-Base g128: FP16 6,99; GPTQ 7,22 (+3,3%); AWQ 8,11 (+16,0%). O padrão coincide com o nosso (GPTQ +4,4%; AWQ +13,8%). Nota: na tabela do mesmo artigo o modelo pós-treinado tem FP16 9,71; o 6,99 é do Base (confirmado, o nosso FP16 6,9946 é comparável).
- Llama 3 8B (Huang et al., 2024, arXiv 2404.14047, FP16 6,1): GPTQ g128 6,5 (+6,6%); AWQ g128 6,6 (+8,2%); nossos: +8,8% e +9,6%.
- Referências conferidas por agente de pesquisa (fontes: arXiv, ACL Anthology, ISTA, MLSys). Correções: arXiv 2404.14047 é "An empirical study of LLaMA3 quantization: from LLMs to MLLMs" (título errado no escopo/planilha antes); MARLIN é arXiv 2408.11743 e PPoPP 2025 (p. 239-251); Llama 3 herd: primeiro autor atual é Grattafiori (v1 listava Dubey); título MLSys do AWQ inclui "on-device". Páginas de Dettmers (NeurIPS) e Lin (MLSys) NÃO confirmadas. Os títulos/páginas vindos de leitor automático de HTML merecem conferência visual.
- Aplicado em: metodo_rascunho (refs, citação Grattafiori, faixa de degradação com fontes), guia, perguntas_banca, berg/resumo_checkpoint, planilha (aba Sensibilidade calibração).

## 2026-10-21 — Rascunho do capítulo 4 (Resultados)

- `redacao/resultados_rascunho.{html,docx,pdf}` (5 páginas): validação do FP16, PPL, custo (com variação entre as 3 séries), trade-off, sensibilidade (calibração e AWQ simétrico), resultados negativos e limites. Valores gerados por script a partir dos JSON; percentuais de memória e vazão calculados sobre o FP16 do mesmo modelo.
- Pendências marcadas em amarelo: estatística (Jean), hardware de consumo, Base ou Instruct, aval do C4, conferência visual das tabelas dos artigos de Huang e Zheng, hipótese do TTFT do INT8 (não testada).

## 2026-10-22 — Guia do código e sessão de professor

- `guia/codigo_explicado.md` (~2.000 palavras): visão de conjunto, cada script bloco a bloco com o porquê, variantes de sensibilidade, run_*.sh, leitura dos JSON, contas para refazer à mão, perguntas prováveis sobre o código, como reproduzir.
- Comentários curtos adicionados em `ppl_wikitext2.py`, `bench_cost.py`, `quantize_gptq.py` e `quantize_awq.py` (apenas linhas de comentário; verificado que a árvore sintática de cada arquivo é idêntica à original, ou seja, a lógica não mudou). Os arquivos que rodaram na DGX são os originais sem esses comentários; md5 dos originais: ver abaixo. Os `quantize_*_c2048`, `quantize_awq_sym` e os `run_*.sh` não foram alterados.
- Verificado (cmp e md5): `quantize_*_c2048.py` diferem dos principais (usam `calibration_concat_size=2048`) e os da DGX são idênticos aos locais. Nota: o filtro de saída do `diff` local imprimiu "Files are identical" para arquivos diferentes; usar `cmp`/`md5sum` para conferir.
- Achado de rastreabilidade: os 8 JSON de custo r1 foram gerados antes do review e não têm `peak_reserved_gb` nem `tps_invalid_reps`; r2 e r3 têm. O review não mudou nenhum valor medido.
- Skill `professor-ptq` criada em `/home/jv/pos-genai/.claude/skills/professor-ptq/SKILL.md` para sessões de estudo (tutor socrático, simulado de banca, registro em `guia/progresso_estudo.md`).

```
40f622b8ae03f9a41a8554525d5f7c45  ppl_wikitext2.py
fdaba4d76ed3bda441ee65ccdbee020e  bench_cost.py
6d46196be7e9a15bbfec36033fe2064f  quantize_gptq.py
25bf22e5828f0324aaf3f8e3be1997f8  quantize_awq.py
40c56c9ec25e216b8d0d8b94df567016  quantize_int8.py
```

## 22/10/2026 - Bancada real (seção 6.2 do projeto)
- DGX Spark (execução dos experimentos): GPU NVIDIA GB10 (driver 580.142), CPU Arm 20 núcleos (Cortex-X925 + Cortex-A725), 121 GiB de memória unificada (CPU e GPU compartilham), Ubuntu 24.04.4 LTS, kernel 6.17.0-1014-nvidia, aarch64. CUDA 13.0 (torch 2.14.0+cu130); nvcc não está no PATH do shell de login (o JIT do Marlin usa o CUDA_HOME configurado no run).
- Avell (guarda dos dados; hardware de consumo a validar): Intel Core i7-12700H, 20 threads, 30 GiB de RAM, GPU NVIDIA GeForce RTX 3060 Laptop com 6 GiB de VRAM. Ainda não usado para medir.
- Limite que vale para a defesa: na DGX a memória é unificada, então "pico de memória" não é VRAM de GPU dedicada; no Avell o 8B em FP16 (~15 GiB) não cabe nos 6 GiB de VRAM, só o INT4 (~5,5 GiB) cabe e no limite.

## 22/10/2026 - Páginas de Dettmers e Lin
- Páginas oficiais (proceedings.neurips.cc e proceedings.mlsys.org) não trazem intervalo de páginas; DBLP bloqueou o acesso (Anubis). Dettmers: DOI 10.52202/068431-2198; o título oficial no NeurIPS é "GPT3.int8()", no arXiv é "LLM.int8()". Páginas ficam como pendência (PDF final ou omitir, conforme a ABNT permite quando não há paginação).
