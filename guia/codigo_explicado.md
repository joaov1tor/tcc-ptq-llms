# Código explicado: o que cada script faz, bloco a bloco

Para quem vai defender: leia na ordem. Todos os scripts estão em `referencia/scripts/` e rodaram na DGX Spark (`~/tcc-ptq/`). Os números dos resultados vêm dos JSON em `referencia/resultados/`. Conceitos (por que PPL, o que é GPTQ etc.) estão em `glossario_decisoes.md`; aqui a pergunta é **"o que o código faz, linha a linha, e como eu provo que mede o que dizemos medir"**.

## 0. Visão de conjunto

```
modelo FP16 (Hugging Face)
   |-- quantize_int8.py  --> models/<m>-int8            (LLM.int8, sem calibração)
   |-- quantize_gptq.py  --> models/<m>-gptq4           (GPTQ 4 bits, calibra no C4)
   |-- quantize_awq.py   --> models/<m>-awq4            (AWQ 4 bits, calibra no C4)
cada artefato (e o FP16 puro):
   |-- ppl_wikitext2.py  --> results/<m>_<cfg>.json        QUALIDADE (perplexidade)
   |-- bench_cost.py     --> results/<m>_<cfg>_cost.json   CUSTO (disco, memória, TTFT, tokens/s)
run_*.sh: só encadeiam os passos acima, um modelo por vez, para nenhuma medição de custo concorrer com outra nossa.
```

Regra de ouro do desenho: **o mesmo `ppl_wikitext2.py` e o mesmo `bench_cost.py` medem todas as configurações** (flag `--quant none|int8|pre`). Se a diferença entre duas linhas da tabela é grande, não é porque o código de medição mudou.

## 1. `ppl_wikitext2.py` — qualidade (perplexidade por token)

| Linhas | O que faz | Por que assim / pergunta provável |
|---|---|---|
| 16-25 | Lê argumentos: `--model`, `--dtype` (padrão float16), `--quant`, `--ctx` (2048), `--seed`, `--out`. | `--quant none` carrega FP16 normal; `int8` aplica LLM.int8 na hora; `pre` carrega um diretório já quantizado e salvo (GPTQ/AWQ e INT8 salvo). |
| 28-38 | Carrega tokenizer e modelo. Em `int8`/`pre` o modelo já nasce na GPU 0 (`device_map={"":0}`); no FP16 carrega e depois `.to(device)`. `.eval()` desliga dropout. | Os checkpoints originais são bfloat16; carregamos em float16 (`dtype=float16`) para todos os casos. |
| 40-42 | Carrega o **teste** do WikiText-2 (`wikitext-2-raw-v1`), junta todas as linhas com `"\n\n"`, tokeniza de uma vez e calcula `n = nº de tokens // 2048` janelas. | É o protocolo do GPTQ/AWQ. A divisão inteira descarta o resto final (menos de 2048 tokens). Llama: 141 janelas; Qwen3: 146 (tokenizers diferentes dão contagens diferentes de tokens). |
| 45-54 | Para cada janela: roda o modelo, pega os logits, e soma a **entropia cruzada** entre o que o modelo previu e o token real seguinte. `logits[:, :-1]` contra `x[:, 1:]`: o logit da posição *i* prevê o token *i+1*. `.float()` converte para float32 antes da conta (evita erro numérico do float16). `reduction="sum"` soma o NLL total. `torch.no_grad()` não guarda gradiente. | Pergunta: "por que `-1`?" Porque o último logit não tem alvo e o primeiro token não tem previsão: cada janela contribui com **2047** tokens, não 2048. |
| 53-54 | `ntok = n * (ctx - 1)` e `ppl = exp(nll / ntok)`. | Perplexidade = exponencial da média do negativo do log da probabilidade correta. Llama: 141 × 2047 = **288.627** tokens avaliados; Qwen3: 146 × 2047 = **298.862**. |
| 55-62 | Monta o JSON com resultado e **condições** (GPU, versões, data, semente). | Reprodutibilidade: qualquer número da planilha aponta para um arquivo com o ambiente em que foi gerado. |

Conta de cabeça para a banca: se a perplexidade é 6,1359, o modelo "hesita" como se escolhesse entre ~6 palavras igualmente prováveis a cada token. Cada +1% no PPL é +1% de "hesitação" geométrica média por token.

**Armadilhas conhecidas (já tratadas):** o `peak_mem_gb` do JSON de PPL (ex. 17,44 GiB no Llama FP16) é o pico **dessa rodada** (pesos + ativações de uma janela de 2048 + logits em float32); **não** é a memória de inferência que reportamos. A memória do trabalho vem do `bench_cost.py`. Não confundir na defesa.

## 2. `bench_cost.py` — custo (disco, memória, TTFT, tokens/s)

| Linhas | O que faz | Por que assim / pergunta provável |
|---|---|---|
| 30 | `assert torch.cuda.is_available()`. | Medir custo em CPU não faz sentido para este trabalho. |
| 34-40 | `sh()` roda `nvidia-smi`; `q()` calcula **mediana, Q1, Q3 e IQR** de uma lista. | Mediana e IQR são robustos a picos de uma repetição (ex.: outro processo na GPU). |
| 42-43 | **Disco**: soma o tamanho dos `*.safetensors` do diretório (ou `*.bin` se não houver; um só formato, para não contar duas vezes). | Disco = tamanho do artefato salvo. O FP16 original já está em 16 bits, então 14,96 GiB é "pesos em 2 bytes". |
| 44-45 | Guarda **quem mais usa a GPU** (`gpu_other_procs_before`) e temperatura/clock antes e depois. | Honestidade experimental: a DGX é compartilhada (whisper, ollama, python de terceiros). Isso fica gravado em cada JSON. |
| 47-59 | Carrega o modelo como no PPL e registra `mem_after_load` = memória alocada logo após a carga (pesos). | `memory_allocated` é só o que o PyTorch alocou. Na DGX (memória unificada) o `nvidia-smi` não separa CPU de GPU, por isso usamos o contador do PyTorch. |
| 61-65 | O **prompt** são os primeiros **512 tokens do split de VALIDAÇÃO** do WikiText-2; `attention_mask` de uns; `pad` = pad_token ou eos. | Validação, não teste, para não reaproveitar os dados da perplexidade. O mesmo prompt entra em todas as configurações. |
| 67-71 | `gen(n)`: `cuda.synchronize()`, inicia o relógio (`perf_counter`), gera `n` tokens (**guloso**: `do_sample=False`; `min_new_tokens=n` impede parar cedo no EOS; `use_cache=True`), `synchronize()` de novo, devolve segundos. | O `synchronize` é essencial: a GPU é assíncrona; sem ele mediríamos só o envio do trabalho, não o término. |
| 73 | `reset_peak_memory_stats()` **depois** da carga e antes do laço. | Assim o pico exclui o transiente da carga do modelo, mas inclui os pesos (já alocados) + cache KV + ativações da geração. |
| 74-81 | Laço de `warmup + reps` (3 + 10): em cada passada mede `gen(1)` e `gen(128)`. As 3 primeiras são descartadas. **TTFT** = `gen(1)` em ms. **tokens/s** = `(128−1) / (gen(128) − gen(1))`. | O `gen(1)` inclui o *prefill* do prompt de 512 tokens + 1 token. Subtrair `gen(1)` de `gen(128)` cancela o prefill e deixa só o custo de gerar 127 tokens a mais ("decode puro"). Se por jitter `gen(128) ≤ gen(1)`, a passada é descartada e contada em `tps_invalid_reps` (0 em todas as séries r2 e r3; as 8 séries r1 foram medidas antes do campo existir). |
| 82 | Lê `max_memory_allocated` (pico) e `max_memory_reserved` (o que o alocador do PyTorch reservou). | Pico alocado é o número que usamos; o reservado fica registrado como informação adicional. |
| 85-91 | Escreve o JSON com mediana/IQR **e** os valores brutos (`*_raw`). | Os brutos são a entrada do teste estatístico do Jean (Kruskal-Wallis / Mann-Whitney). |

**Limitações que você deve declarar:** (a) batch 1, guloso, um prompt de 512 tokens e 128 novos tokens; outros cenários (batch maior, contexto longo) podem inverter conclusões. (b) A subtração de dois `generate` assume que o prefill é igual nas duas chamadas. (c) As primeiras séries (r1) rodaram com uma versão anterior do script, antes do `/code-review`; as correções (proteção do denominador, registro do `peak_reserved_gb`) não mudaram nenhum número (o teste de fumaça deu os mesmos valores), por isso os 8 JSON r1 não trazem os campos `peak_reserved_gb` e `tps_invalid_reps`.

## 3. Quantização

### `quantize_int8.py` (8 linhas)
Carrega o FP16 com `BitsAndBytesConfig(load_in_8bit=True)` e salva com `save_pretrained`. **Não usa dados de calibração**: o LLM.int8 decide em tempo de execução, por vetor, quais dimensões têm *outliers* e as mantém em FP16 (Dettmers et al., 2022). O artefato salvo é o que medimos (`--quant pre`).

### `quantize_gptq.py` e `quantize_awq.py` (30 linhas cada)
1. **Calibração (linhas 13-24):** abre 1 *shard* do C4 (`allenai/c4`, `en/c4-train.00000-of-01024`), embaralha os índices com semente 0 e pega os **128 primeiros documentos com ≥ 8000 caracteres**. `assert len(calib) == 128` garante que não saiu menos. Documentos inteiros, sem concatenar nem truncar (≈ 387 mil tokens no Qwen3).
2. **Configuração (linha 26):**
   - GPTQ: `QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)`.
   - AWQ: `QuantizeConfig(method="awq", bits=4, group_size=128, sym=False)`.
3. `GPTQModel.load(src, cfg)` → `m.quantize(calib, batch_size=1)` (roda o algoritmo camada a camada) → `m.save(dst)`.

Significado dos parâmetros: `bits=4` (16 níveis); `group_size=128` (uma escala a cada 128 pesos; menos erro que uma por linha, mais metadados); `desc_act=False` (não reordena colunas por importância; mantém o formato compatível com o kernel rápido); `sym=True` (zero fixo; no AWQ usamos `sym=False`, com *zero-point*, que é o uso habitual).

### Variantes de sensibilidade (mudam **uma** coisa cada)
| Arquivo | Diferença única | Para responder |
|---|---|---|
| `quantize_gptq_c2048.py`, `quantize_awq_c2048.py` | `m.quantize(calib, calibration_concat_size=2048, batch_size=1)`: os mesmos 128 documentos são empacotados em blocos de 2048 tokens (protocolo original do GPTQ). | O protocolo de calibração do gptqmodel (documentos inteiros) distorce o resultado? Resposta: PPL mudou no máximo 0,2%. |
| `quantize_awq_sym.py` | `sym=True` no AWQ. | O *zero-point* explica o AWQ pior no Qwen3? Resposta: 7,9620 contra 7,9623, não. |

## 4. Os `run_*.sh`
- `run_int8.sh`: para cada modelo, quantiza INT8 → PPL completo → custo.
- `run_awq.sh` / `run_gptq.sh`: o mesmo para os métodos de 4 bits (com `ninja` e `CUDA_HOME` para o JIT do Marlin).
- `run_noite.sh`: (A) réplicas r2 e r3 de custo das 8 configurações; (B) sensibilidade da calibração (blocos de 2048). Conta falhas em `FAIL` e imprime `ALLDONE-NOITE ... falhas=N`.
- `run_awq_sym.sh`: AWQ simétrico no Qwen3.
Por que sequencial: a medição de custo exige a GPU sem outra carga **nossa**; a DGX tem cargas de terceiros que não controlamos (registradas nos JSON).

## 5. Como ler os JSON
**Perplexidade (`<m>_<cfg>.json`):** `ppl` (resultado), `windows` (nº de janelas), `tokens_scored` (tokens avaliados = janelas × 2047), `ctx`, `seconds`, `peak_mem_gb` (pico da própria avaliação; **não** é o do custo), `dataset`, versões e `ts`.
**Custo (`<m>_<cfg>_cost[_r2|_r3].json`):** `disk_gb`; `mem_after_load_gb` (pesos); `peak_mem_gb` (pico na geração = número usado nas tabelas); `ttft_ms` e `tokens_per_s` com `median`, `q1`, `q3`, `iqr`; `*_raw` (as 10 repetições); `tps_invalid_reps` (ausente nos r1); `gpu_other_procs_before` (quem estava na GPU); temperatura e clock antes e depois. As séries são r1 (sem sufixo), r2 e r3.

## 6. Contas que a banca pode pedir para você refazer
- **Δ PPL:** `PPL_config / PPL_FP16 − 1`. Llama GPTQ: 6,6754 / 6,1359 − 1 = +8,8%.
- **PPL a partir do NLL:** `exp(NLL_total / (janelas × 2047))`.
- **Redução de memória:** `pico_config / pico_FP16 − 1`. Llama GPTQ: 5,46 / 15,11 − 1 = −63,9%.
- **Aceleração:** `tokens/s_config / tokens/s_FP16`. Llama GPTQ: 36,53 / 13,58 = 2,69×.
- **Por que 4 GiB não saiu:** 8 bilhões de pesos × 4 bits = 4 GB só de pesos, mais escalas, o *embedding* e o `lm_head` que seguem em 16 bits, mais cache KV e ativações: 5,46 GiB.

## 7. Perguntas prováveis sobre o código
1. **"Como sei que o tokens/s não inclui o prefill?"** Subtraímos o tempo de gerar 1 token do tempo de gerar 128 (linha 80). O prefill aparece nos dois e se cancela.
2. **"Por que `synchronize`?"** A GPU executa de forma assíncrona; sem ele o relógio pararia antes do fim do trabalho.
3. **"A perplexidade usa dados de calibração?"** Não: a calibração é no C4; a PPL é no teste do WikiText-2.
4. **"Por que o disco do INT8 é 8,5 GiB e não 7,5?"** Nem todo o modelo é quantizado: embedding e `lm_head` ficam em 16 bits; os pesos quantizados carregam metadados (escalas).
5. **"O que garante que as configurações são comparáveis?"** Mesmo script de medição, mesmo prompt, mesma semente, mesmo ambiente (versões nos JSON) e série repetida três vezes.
6. **"O que mudou entre r1 e r2/r3?"** Só o horário e o que mais estava na GPU; o código de medição é o mesmo funcionalmente (as correções do review não alteram valores).
7. **"Por que `desc_act=False`?"** Não reordena as colunas por importância (mais simples e sem custo extra na inferência); é a configuração usual para inferência rápida. A relação exata com o kernel Marlin está na seção 5 do `glossario_decisoes.md` (CONFIRMAR antes de afirmar na banca).

## 8. Como reproduzir (resumo)
```
cd ~/tcc-ptq            # na DGX
PY=.venv-quant/bin/python
$PY quantize_gptq.py models/Meta-Llama-3-8B models/Meta-Llama-3-8B-gptq4 --nsamples 128
$PY ppl_wikitext2.py --model models/Meta-Llama-3-8B-gptq4 --quant pre --out results/llama3-8b_gptq4.json
$PY bench_cost.py    --model models/Meta-Llama-3-8B-gptq4 --quant pre --out results/llama3-8b_gptq4_cost.json
```
(Para o GPTQ/AWQ é preciso `ninja` e `CUDA_HOME=/usr/local/cuda` no ambiente, por causa da compilação do kernel Marlin.)
