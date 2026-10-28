# Guia vivo de defesa: o que usamos, por quê, e como explicar

Atualizado a cada etapa. Para cada item: **o que é**, **por que escolhemos**, **como dizer na banca**, **limitação honesta**.
Itens marcados (CONFIRMAR) são números de memória que preciso checar na fonte antes de citar.

---

## 1. Perplexidade (a métrica de qualidade)
- **O que é:** exp(média da perda por token). Mede o quanto o modelo "se surpreende" com um texto. Menor = melhor.
- **Como medimos:** WikiText-2 raw, split test, textos unidos por `\n\n`, tokenizado de uma vez, cortado em janelas de 2048 tokens sem sobreposição; perda de cada token previsto dado o anterior (`ppl = exp(NLL_total / tokens_avaliados)`).
- **Por quê:** é a métrica validada pelo Berg e a usada nos artigos de GPTQ e AWQ, então dá para conferir nossa baseline (Llama 3 8B: 6,14 vs ~6,1 publicado; Qwen3-8B: 6,99 vs 6,99).
- **Limitação honesta:** não é comparável com o `word_perplexity` do lm-eval-harness (outra unidade). Perplexidade baixa não garante que o modelo responda bem tarefas; mede só modelagem de linguagem.
- **Pergunta provável:** "PPL basta para dizer que o modelo quantizado é bom?" Resposta: não; é proxy de fidelidade ao modelo original. O escopo aprovado é esse; tarefas downstream ficam como trabalho futuro.

## 2. Quantização pós-treino (PTQ) e "weight-only"
- **O que é:** trocar os pesos de 16 bits por inteiros de 8 ou 4 bits depois do treino, sem treinar de novo. "Weight-only": só os pesos são quantizados; as ativações continuam em FP16.
- **Por quê:** o modelo ocupa menos memória e, na decodificação, lê menos bytes por token (gargalo é a banda de memória).
- **Fórmula base:** `w_q = round(w / s)` com escala `s`; reconstrução `w ≈ s · w_q`.
- **Limitação:** erro de arredondamento; o que cada método faz é decidir *como* arredondar para errar menos.

## 3. INT8 com LLM.int8() (bitsandbytes) (Dettmers et al., 2022)
- **O que é:** quantização por vetor em 8 bits + tratamento de *outliers*: poucas dimensões das ativações têm valores enormes (aparecem a partir de ~6,7 bilhões de parâmetros). Essas dimensões ficam em FP16; o resto vai em INT8.
- **Por que lento no TTFT:** o caminho de outliers e a quantização das ativações em tempo de execução custam no prefill (processamento do prompt). Medimos TTFT pior nos dois modelos.
- **Por que mais rápido na decodificação:** gerar 1 token lê todos os pesos; com metade dos bytes lê-se menos da memória (decodificação é *memory-bound*).
- **Nossos números:** PPL +1,5% (Llama 3) e +0,5% (Qwen3); memória −40%; tok/s +33%; TTFT +13% / +29%.
- **Sem calibração:** o INT8 do bitsandbytes não usa dados de calibração.

## 4. GPTQ (INT4) (Frantar et al., 2022)
- **O que é:** quantiza camada por camada, escolhendo o arredondamento que minimiza o erro da *saída* da camada, `||WX − ŴX||²`, e não o erro do peso isolado. Usa a Hessiana `H = 2XXᵀ` das entradas `X` da camada (por isso precisa de dados) e, depois de quantizar cada coluna, **compensa o erro nas colunas ainda não quantizadas** (herança do método OBS/OBQ).
- **Por que precisa de calibração:** `X` vem de dados reais passando pelo modelo.
- **Nossa configuração:** 4 bits, grupo 128, simétrico, `desc_act=False`, calibração C4 × 128 documentos (seed 0).
- **Pergunta provável:** "Por que não calibrar no WikiText-2?" Porque o teste é WikiText-2: calibrar nele deixaria o modelo "ver" o gabarito e inflaria o resultado. C4 evita o vazamento. (Decisão minha; levar ao Berg no checkpoint.)

## 4b. O dataset C4 (usado só na calibração)
- **O que é:** *Colossal Clean Crawled Corpus*, corpus de texto da web em inglês criado pelo Google para treinar o T5 (Raffel et al., 2020). Vem do Common Crawl (coleta de abril de 2019) filtrado por heurísticas: mantém linhas terminadas em pontuação, descarta páginas curtas, com palavras ofensivas e trechos duplicados, e mantém só páginas identificadas como inglês. Tamanho da variante `en`: ~365 milhões de documentos, ~156 bilhões de tokens (Dodge et al., 2021) (CONFIRMAR).
- **O que ele faz no nosso projeto:** fornece os textos que passam pelo modelo durante a quantização GPTQ (e, depois, AWQ) para estimar a Hessiana `H = 2XXᵀ` de cada camada. **Não entra na métrica**: a perplexidade continua medida no WikiText-2. Não há fine-tuning: os pesos não são treinados, só arredondados com ajuda desses dados.
- **Por que C4 e não WikiText-2:** calibrar no corpus de teste vazaria o gabarito para a quantização e embelezaria o resultado. O C4 é o corpus de calibração do artigo original do GPTQ (Frantar et al., 2022), então mantém comparabilidade com a literatura. Também é texto de domínio geral, não específico do teste.
- **Como foi usado (registro exato):**
  - Fonte: `allenai/c4`, arquivo `en/c4-train.00000-of-01024.json.gz` (primeiro de 1024 shards do split train).
  - Seleção: embaralhamento com seed 0, primeiros 128 documentos com pelo menos 8.000 caracteres.
  - Total processado pelo gptqmodel no Qwen3-8B-Base: 386.925 tokens, sem padding (≈3.000 tokens por documento, em média).
- **Diferenças em relação ao protocolo original (declarar na banca):**
  - O GPTQ original usa 128 segmentos aleatórios de **2048 tokens** (≈262 mil tokens). Nós passamos **documentos inteiros** de comprimento variável; o gptqmodel os ordena por tamanho e **não concatena nem trunca para 2048** por padrão (`calibration_concat_size` vazio).
  - A biblioteca avisa que o ideal seria mais de 256 amostras; usamos 128, como no artigo.
  - Consequência: o INT4 pode diferir um pouco do número publicado por causa disso. É uma fonte de variação a mencionar, não um erro.
- **Limitações do C4 para citar:** é texto web em inglês, com ruído e vieses (Dodge et al., 2021); o Llama 3 e o Qwen3 foram treinados em corpora maiores e multilíngues. Se a calibração em outro corpus (WikiText-2 de treino, Pile) mudar o resultado, isso é sensibilidade do método, e fica como análise opcional.
- **Pergunta provável da banca:** "Vocês não usaram dataset próprio?" Resposta: não treinamos nem ajustamos nada com dado novo; o C4 só serve para a quantização estimar quais pesos importam. O teste final continua em WikiText-2, que o modelo não viu na calibração.

## 4c. AWQ (INT4) (Lin et al., 2023)
- **O que é:** *Activation-aware Weight Quantization*. Ideia central: uma pequena fração dos pesos (cerca de 1%) é mais importante, e quem os identifica são as **ativações** (canais que recebem valores grandes), não os próprios pesos. Em vez de guardar esses pesos em maior precisão (o que complicaria o hardware), o AWQ **multiplica o canal por uma escala antes de quantizar** (e compensa na camada anterior), o que reduz o erro relativo daqueles pesos. A escala é buscada por grade (*grid search*) em dados de calibração. Não usa retropropagação nem reconstrução com Hessiana, ao contrário do GPTQ.
- **Diferença para o GPTQ (resumo para a banca):** GPTQ corrige o erro arredondando peso a peso com compensação (usa a Hessiana); AWQ protege canais importantes por escala (usa a magnitude das ativações). Ambos precisam de calibração, ambos são weight-only.
- **Como rodamos:** pelo `gptqmodel` (`method="awq"`), 4 bits, grupo 128, **assimétrico (zero-point)**, mesma calibração C4 x128 do GPTQ. Mesmo ambiente, mesmo carregador e mesmos kernels que o GPTQ, para que a diferença venha do algoritmo e não da infraestrutura.
- **Por que não o llmcompressor (que também instalamos):** na versão 0.14 o AWQ sai em formato `compressed-tensors`, e no `transformers` isso pode executar sem kernel INT4 dedicado, o que distorceria tempo e memória. Está registrado como alternativa não usada.
- **Resultado e sensibilidade da calibração:** no Qwen3 o AWQ ficou bem pior que o GPTQ (+13,8% contra +4,4% de PPL); no Llama 3 os dois ficaram próximos (+9,6% e +8,8%). Para checar se o culpado era o protocolo de calibração (documentos inteiros em vez de blocos de 2048), refizemos os dois com blocos de 2048: a PPL mudou no máximo 0,2% (Qwen3 AWQ 7,9623 → 7,9463). Logo a calibração não explica o resultado. Testamos também o AWQ **simétrico** (sem zero-point) no Qwen3: PPL 7,9620 contra 7,9623 do assimétrico, ou seja, o zero-point não é a causa (conferimos que os tensores `qzeros` realmente mudaram: constante no simétrico, milhares de valores distintos no assimétrico). **A literatura mostra o mesmo padrão:** Zheng et al. (2025, arXiv 2505.02214, Tabela 3) reportam para o Qwen3-8B-Base com grupo 128 PPL 7,22 (GPTQ) e 8,11 (AWQ) contra FP16 6,99, isto é +3,3% e +16,0%; nós medimos +4,4% e +13,8%. Então o AWQ pior que o GPTQ no Qwen3 parece ser comportamento do algoritmo nesse modelo, não defeito do nosso porte. Hipóteses de causa que não testamos: sensibilidade do Qwen3 à busca de escala do AWQ. Dizer na banca: "reproduz o padrão da literatura; causa do algoritmo não isolada por nós".
- **Limitações a declarar:** a implementação do AWQ no gptqmodel é um porte do algoritmo, não o código dos autores (CONFIRMAR comparando com o AutoAWQ, que está descontinuado); o AWQ original calibra com uma amostra do Pile, e nós usamos C4 para manter o mesmo protocolo do GPTQ.

## 5. Parâmetros do GPTQ que precisamos saber explicar
- **Group size 128:** uma escala (e um zero) a cada 128 pesos, em vez de uma por linha inteira. Mais grupos = menos erro, mais bytes de metadados (~0,16 a 0,25 bit por peso extra) (CONFIRMAR).
- **Simétrico vs assimétrico:** simétrico centra a faixa em zero (sem *zero-point*); assimétrico usa zero-point e cobre melhor faixas deslocadas, ao custo de mais metadado.
- **`desc_act` (act-order):** ordena as colunas pela importância (diagonal da Hessiana) antes de quantizar. Melhora a precisão, mas deixa os kernels mais lentos/complexos. Usamos `False` (padrão compatível com kernels rápidos). É um fator que poderia ser explorado.
- **Calibração com 128 amostras:** padrão da literatura; poucas amostras deixam a Hessiana ruidosa. Rodamos um smoke com 16 só para testar o pipeline.

## 6. Kernel Marlin (execução do INT4)
- **O que é:** kernel CUDA para multiplicar ativações FP16 por pesos INT4 de forma eficiente. O gptqmodel o escolhe sozinho.
- **O que nos deu trabalho (resultado a relatar):** é compilado na hora (JIT) e exigiu `ninja` e `nvcc` (CUDA 13.0) no PATH; sem eles o carregamento falhou. Também o `transformers` exigiu o pacote `optimum` para ler GPTQ.
- **Por que isso importa na tese:** o custo medido de INT4 depende da qualidade do kernel disponível na plataforma (aarch64, GB10, CUDA 13), não só do número de bits. Dizer isso na discussão.

## 7. Métricas de custo
- **Disco / memória após carga / pico:** em GiB, do artefato salvo. Na DGX a memória é unificada (CPU e GPU compartilham), então o `nvidia-smi` não separa; usamos `torch.cuda.max_memory_allocated`.
- **TTFT:** tempo até o primeiro token (prefill do prompt de 512 tokens) em ms, mediana de 10 repetições após 3 de aquecimento.
- **Tokens/s:** `(N−1)/(t_N − t_1)` na geração de 128 tokens, batch 1, guloso.
- **Estatística:** mediana e IQR (robustos a picos). Cada configuração tem 3 séries de 10 medições (r1, r2, r3; aba Réplicas da planilha). A variação entre séries ficou abaixo de 2% na mediana de tok/s e não altera a ordem entre as configurações; a comparação entre configurações é testada pelo Jean (Kruskal-Wallis / Mann-Whitney).
- **Ressalva:** outros processos (whisper, ollama, python) usavam a GPU durante algumas medições; estão registrados nos JSON. A r3 do Llama 3 FP16 falhou por falta de memória (GPU compartilhada) e foi refeita em outro horário; isso consta no diário e na planilha.

## 8. Hardware: DGX Spark (GB10)
- Arquitetura aarch64 (CPU Arm) + GPU Blackwell GB10, memória unificada de 128 GB (≈121 GiB visíveis). Largura de banda ≈ 273 GB/s (CONFIRMAR na ficha NVIDIA).
- **Consequência:** decodificar é limitado por banda de memória, o que explica por que menos bytes por peso acelera tok/s.
- **Limitação:** resultados de velocidade valem para essa plataforma; o Avell (GPU de consumo) e o notebook do João Paulo vão validar em outros hardwares.

## 9. Decisões de método que precisamos justificar na banca
| Decisão | Justificativa | Risco |
|---|---|---|
| Qwen3-8B **Base** (não Instruct) | PPL é medida de modelagem de linguagem, mais limpa no Base | Banca pode perguntar de Instruct; decisão em aberto |
| Contexto 2048, janelas sem sobreposição | Padrão GPTQ/AWQ, comparável com literatura | Sobreposição daria PPL menor |
| Calibração C4 | Evita vazamento do teste | Dataset extra além do escopo; confirmar com Berg |
| 1 modelo FP16 como baseline | Escopo do Berg | Nenhum |
| Limiar de 3% de PPL | Só referência de interpretação | Literatura (g128): Llama 3 8B GPTQ +6,6% e AWQ +8,2% (Huang et al., 2024); Qwen3-8B-Base GPTQ +3,3% e AWQ +16,0% (Zheng et al., 2025); conferido nas fontes em 21/10 |
| Meta de 4 GB de VRAM | Objetivo da proposta | 8B em INT4 ≈ 5,7–6 GB: provavelmente inatingível; tratamos como resultado |

## 10. Pendentes de entrar neste guia
- Análise do trade-off qualidade × custo (os números do INT4 já estão na planilha; falta a leitura para a banca).
- Teste estatístico (Jean) e como ler p-valor e tamanho de efeito.
- Validação em hardware de consumo.

## Referências-chave para ler
- Dettmers et al. 2022, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*.
- Frantar et al. 2022, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*.
- Lin et al. 2023, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*.
- Raffel et al. 2020, *Exploring the limits of transfer learning with a unified text-to-text transformer* (define o C4).
- Dodge et al. 2021, *Documenting large webtext corpora: a case study on the Colossal Clean Crawled Corpus* (descreve o conteúdo e os vieses do C4).
- Frantar et al. 2024, *Marlin* (kernel INT4); Merity et al. 2016, *WikiText*; Raffel et al. 2020, *C4*.


## Referências em ABNT (rascunho para o texto; conferir páginas/DOI antes de entregar)
ALLEN INSTITUTE FOR AI. **allenai/c4**: conjunto de dados. Hugging Face, [s.d.]. Disponível em: https://huggingface.co/datasets/allenai/c4. Acesso em: 20 out. 2026.

DODGE, Jesse *et al.* Documenting large webtext corpora: a case study on the Colossal Clean Crawled Corpus. *In*: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING, 2021, Punta Cana. **Proceedings** [...]. Stroudsburg: Association for Computational Linguistics, 2021. p. 1286-1305. DOI 10.18653/v1/2021.emnlp-main.98.

FRANTAR, Elias *et al.* GPTQ: accurate post-training quantization for generative pre-trained transformers. **arXiv**, 2210.17323, 2022. (publicado na ICLR 2023)

MERITY, Stephen *et al.* Pointer sentinel mixture models. **arXiv**, 1609.07843, 2016.

RAFFEL, Colin *et al.* Exploring the limits of transfer learning with a unified text-to-text transformer. **Journal of Machine Learning Research**, v. 21, n. 140, p. 1-67, 2020.

LIN, Ji *et al.* AWQ: activation-aware weight quantization for LLM compression and acceleration. **arXiv**, 2306.00978, 2023. (publicado na MLSys 2024)


HUANG, Wei *et al.* An empirical study of LLaMA3 quantization: from LLMs to MLLMs. **Visual Intelligence**, v. 2, art. 36, 2024. DOI 10.1007/s44267-024-00070-x. arXiv:2404.14047.

ZHENG, Xingyu *et al.* An empirical study of Qwen3 quantization. **arXiv**, 2505.02214, 2025.

FRANTAR, Elias *et al.* MARLIN: mixed-precision auto-regressive parallel inference on large language models. *In*: ACM SIGPLAN SYMPOSIUM ON PRINCIPLES AND PRACTICE OF PARALLEL PROGRAMMING (PPoPP), 30., 2025, Las Vegas. **Proceedings** [...]. New York: ACM, 2025. p. 239-251. DOI 10.1145/3710848.3710871. arXiv:2408.11743.

GRATTAFIORI, Aaron *et al.* The Llama 3 herd of models. **arXiv**, 2407.21783, 2024. (a v1 listava Dubey como primeiro autor; a versão vigente lista Grattafiori)
