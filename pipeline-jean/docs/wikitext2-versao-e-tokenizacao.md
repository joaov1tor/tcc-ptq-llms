# WikiText-2: versão, tokenização e o que registrar

Documento de saída da tarefa do Berg. Serve para o ticket e para o §6.4 do TCC.

## TL;DR

- **Corpus:** WikiText-2 de Merity et al. (2017). Não existe `wikitext-2-raw-v2`.
- **Variante lockada:** `EleutherAI/wikitext_document_level` / `wikitext-2-raw-v1`, campo `page`, split `test`.
- **Harness:** tarefa `wikitext`, metadata `2.0`, `lm-eval` 0.4.13 (31 ago 2026). Continua no mesmo `dataset_name`.
- **Métrica oficial:** `word_perplexity` (e `byte_perplexity` / `bits_per_byte`). Normalização por palavras/bytes do texto **original**, antes da detokenização. Cada modelo usa o próprio tokenizer.

## Por que WikiText-2

WikiText-2 é o benchmark consolidado de language modeling (Merity et al., 2017). GPTQ (Frantar et al., 2023), AWQ (Lin et al., 2024) e o Language Model Evaluation Harness (Gao et al., 2024) reportam perplexidade nesse corpus. É público, sem PII e não exige dataset próprio — alinhado ao §6.4 e ao §6.8 do projeto.

WikiText-103 fica de fora. É o mesmo domínio, porém grande demais para o recorte de recursos limitados. A tarefa `wikitext` do harness é WikiText-2, não 103.

## Por que `wikitext-2-raw-v1` e não `wikitext-2-v1`

O Hugging Face ainda oferece duas variantes `v1`:

- `wikitext-2-v1` — arquivos `.tokens`. Troca OOV por `<unk>`. Isso é o vocabulário de 2016, não o do Llama 3 / Qwen3.
- `wikitext-2-raw-v1` — texto bruto, sem `<unk>`.

A literatura atual de avaliação de LLMs (Gao et al., 2024) mede perplexidade sobre o texto retokenizado pelo modelo sob teste. Usar `<unk>` de um vocabulário alheio viesa o tokenizer nativo. Por isso o lock é **raw**.

Não há release mais nova do texto do WikiText-2. O que mudou desde 2017 é o empacotamento (linha a linha vs documento), não o corpus.

## Por que `EleutherAI/wikitext_document_level` e não `Salesforce/wikitext`

O card `Salesforce/wikitext` lista 4.358 linhas no test. O harness **não** avalia assim.

A tarefa oficial aponta:

- `dataset_path: EleutherAI/wikitext_document_level`
- `dataset_name: wikitext-2-raw-v1`
- `output_type: loglikelihood_rolling`
- campo `page` (artigo inteiro)

Fonte: [wikitext.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/wikitext/wikitext.yaml) no harness 0.4.13.

O card HF do empacotamento document-level ainda copia os números linha a linha (4.358). Dry-run local em 4 out 2026: **62 páginas no split test**, 0 vazias — alinhado aos ~60 artigos do WikiText-2 original (Merity et al., 2017). O JSON registra `n_pages_in_split` e `n_docs`.

## Tokenização e detokenização

O raw de 2016 ainda carrega marcas do tokenizador Moses (` @-@ `, ` . `, `= = =`). Sem detokenizar, o BPE/SentencePiece moderno vê texto artificial.

Pipeline do harness (é o que o script clona):

1. Carrega `page`.
2. Conta palavras (`re.split(r"\s+", page)`) e bytes UTF-8 no texto **cru**. Comentário oficial: *“wikitext counts number of words in original doc before detokenization”* ([preprocess_wikitext.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/wikitext/preprocess_wikitext.py)).
3. Aplica `wikitext_detokenizer`: contrações, ` @-@ ` → `-`, pontuação, colchetes, `= =` → `==`, ` N ` → ` 1 `.
4. O modelo retokeniza com o **próprio** tokenizer. Não se usa o vocabulário do WikiText.
5. `loglikelihood_rolling`: um documento por vez; janelas rolantes; cada token previsto uma vez; prefixo BOS/EOS.
6. `word_perplexity = exp(-soma_NLL / soma_palavras)`.

Não treinamos tokenizer. Não usamos Moses, NLTK nem spaCy. Sem lowercase, sem `<unk>`, sem stopwords.

## Processamento adicional a registrar (Etapa 5)

- Split: **test**. Não validation, não train.
- Sem concatenar documentos.
- Sem recorte de documentos no experimento oficial (split completo, `--all-docs` / perfil `full`). No smoke (Mac/Colab): `--max-docs` e `--max-length` só para validar a lógica.
- **Truncamento de contexto registrado (§6.4):** janelas rolantes de `max_length = 2048` tokens, iguais para FP16 e INT8 e para os dois modelos. É menor que o contexto nativo do Llama 3 (8k) e do Qwen3, e foi escolhido para caber no custo da bancada; o harness usaria o máximo do modelo. A PPL depende desse valor, então ele entra no JSON (`max_length`) e não muda entre células.
- Logits convertidos para fp32 antes do `log_softmax`, para que o arredondamento em fp16 não se misture ao efeito da quantização.
- Revisões fixadas: o commit SHA do modelo e do dataset no Hub vai no JSON (`model_revision`, `dataset_revision`).
- Sem temperatura. Forward de log-probs, não sampling.
- Páginas vazias puladas e contadas em `n_docs_skipped_empty`.
- Calibração de PTQ (AWQ/GPTQ), quando houver, usa corpus distinto (C4). WikiText-2 não entra na calibração.

## O que esta escolha não permite

Comparar o PPL **absoluto** com as tabelas do AWQ e do GPTQ. Esses papers concatenam o test set e normalizam por **token**. O harness normaliza por **palavra** e avalia **por documento** (EleutherAI/lm-evaluation-harness#2301). Llama-2-7B dá ~5,47 no protocolo AWQ e outro número no harness. Os dois são válidos; não são o mesmo número.

O limiar de 3% do TCC vale somente como variação relativa contra o baseline FP16 medido neste protocolo.

`token_perplexity` entra no JSON como auditoria, não como métrica oficial.

Para a conferência a valores publicados da Etapa 1 existe um protocolo de auditoria separado, `gptq_wikitext2` (`scripts/run_perplexity.py --protocol gptq_wikitext2`; no `run_benchmark.py` o audit é default-on, use `--no-audit-gptq` para desligar): `Salesforce/wikitext` linha a linha, `"\n\n".join`, sem detokenizer, blocos de 2048 sem sobreposição, PPL por token, como no `datautils.py`/`llama_eval` do GPTQ. Ele só confere a bancada contra as tabelas publicadas; não substitui a métrica oficial.

Para INT4 (AWQ/GPTQ) o lock de PPL permanece idêntico: nada muda no corpus, na detokenização, nas janelas ou na normalização. INT4 só muda a **origem dos pesos** (pré-quantizados em C4 pelo `src/ppl/quantize.py`) e o backend de carga (`gptqmodel`). A `token_perplexity` sob o protocolo GPTQ passa a ser reportada sempre em paralelo porque é a métrica que os papers AWQ/GPTQ publicam — útil como leitura cruzada, mas o limiar de 3% do TCC continua medido na `word_perplexity`.

## Versões lockadas

| Item | Valor |
| --- | --- |
| Corpus | WikiText-2 (Merity et al., 2017) |
| Dataset HF | `EleutherAI/wikitext_document_level` |
| Config | `wikitext-2-raw-v1` |
| Campo | `page` |
| Split | `test` |
| Harness | `lm-eval` 0.4.13 |
| Task | `wikitext` / metadata `2.0` |
| Protocolo no JSON | `harness_wikitext_v2` |
| Métrica oficial | `word_perplexity` |
| Contexto (`max_length`) | 2048 |
| Auditoria | `gptq_wikitext2` (não oficial) |

## Referências

- MERITY, Stephen et al. Pointer sentinel mixture models. ICLR, 2017. https://arxiv.org/abs/1609.07843
- GAO, Leo et al. The Language Model Evaluation Harness. v0.4.3. Zenodo, jul. 2024. https://github.com/EleutherAI/lm-evaluation-harness
- EleutherAI. `lm_eval/tasks/wikitext/wikitext.yaml` e `preprocess_wikitext.py`, tag v0.4.13.
- EleutherAI/lm-evaluation-harness#2301. Diferença de PPL vs AWQ/GPTQ (normalização por palavra vs token; documento vs concatenação).
- FRANTAR et al. GPTQ. ICLR, 2023. https://arxiv.org/abs/2210.17323
- LIN et al. AWQ. MLSys, 2024. https://arxiv.org/abs/2306.00978
