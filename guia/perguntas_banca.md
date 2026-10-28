# Perguntas prováveis da banca e respostas de treino

Complementa `glossario_decisoes.md` (o porquê de cada escolha). Aqui o formato é pergunta, resposta curta (o que dizer em voz alta) e limite honesto. Números vêm de `resultados_benchmark_PTQ.xlsx`. As referências foram conferidas nas fontes em 21/10/2026.

## A. Escopo e desenho

**1. Qual o problema e a pergunta do trabalho?**
Quanto de qualidade (perplexidade) se perde, e quanto de custo (memória, disco, velocidade) se ganha, ao quantizar pesos de LLMs de 8 bilhões de parâmetros para INT8 e INT4 sem retreinar? Comparamos dois modelos (Llama 3 8B e Qwen3-8B) e três métodos contra o FP16.
*Limite:* dois modelos, uma tarefa de qualidade (perplexidade), uma máquina de medição.

**2. Por que só pós-treino (PTQ) e não QAT ou fine-tuning?**
Escopo validado pelo orientador. PTQ é o caminho de quem usa um modelo pronto, sem GPU de treino nem dados de treino. QAT e QLoRA ficam como trabalho futuro.

**3. Por que perplexidade como métrica de qualidade?**
É barata, determinística e é a métrica usada nos artigos de GPTQ e AWQ, então comparamos com a literatura. Mede modelagem de linguagem, não capacidade em tarefas.
*Limite:* perplexidade baixa não garante bom desempenho em raciocínio ou instruções; não medimos tarefas.

**4. Por que WikiText-2 e por que contexto 2048 sem sobreposição?**
Padrão da literatura de quantização (GPTQ, AWQ). Janelas sem sobreposição dão PPL maior que com sobreposição, então não comparamos com números de outros protocolos. Nosso FP16 (Llama 6,1359; Qwen 6,9946) bate com o publicado (~6,1 e 6,99).
*Limite:* o `word_perplexity` do lm-eval-harness é outra métrica e não é comparável; ainda não medimos.

**5. Por que o Qwen3 Base e não o Instruct?**
Perplexidade mede modelagem de linguagem, mais limpa no Base. O escopo diz "Qwen 3" sem especificar; levamos ao Berg.
*Limite:* decisão em aberto até a resposta do orientador.

## B. Métodos

**6. Qual a diferença entre INT8 (LLM.int8), GPTQ e AWQ?**
LLM.int8 quantiza em 8 bits e separa os poucos canais com valores extremos (outliers) em FP16; não precisa de calibração. GPTQ arredonda peso a peso e compensa o erro nos pesos seguintes usando a Hessiana, com dados de calibração. AWQ identifica os canais importantes pela magnitude das **ativações** e escala esses canais antes de quantizar. GPTQ e AWQ são INT4, grupo de 128, só nos pesos.

**7. Por que calibrar no C4 e não no WikiText-2?**
Calibrar no mesmo corpus que se avalia vaza informação e melhora a PPL artificialmente. O C4 é de domínio diferente. Foi escolha nossa fora do escopo aprovado e vai ao Berg.
*Limite:* o protocolo original do GPTQ usa 128 blocos de 2048 tokens; o nosso usa 128 documentos inteiros (~387 mil tokens). Testamos o protocolo original: a PPL mudou no máximo 0,2%.

**8. O que é o kernel Marlin e por que importa?**
É o kernel de multiplicação com pesos INT4 que roda na GPU. Sem um kernel dedicado, o INT4 pode até ocupar menos memória, mas ficar lento. Os dois métodos de 4 bits usam o mesmo carregador e o mesmo kernel, para a diferença vir do algoritmo.

## C. Resultados

**9. Qual o resultado principal?**
INT8 perde pouco (+1,5% Llama, +0,5% Qwen) e ganha memória (pico de 15,1 para 8,6 GiB no Llama). INT4 reduz o pico em cerca de 64% (15,1 para 5,5 GiB no Llama; 15,4 para 5,8 no Qwen) e cerca de 2,7× a 2,9× em tokens/s, com perda de +8,8% (GPTQ) e +9,6% (AWQ) no Llama e +4,4% (GPTQ) e +13,8% (AWQ) no Qwen.

**10. O INT8 ficou mais lento no primeiro token. Por quê?**
O TTFT do INT8 (169 ms Llama, 221 ms Qwen) é pior que o do FP16 (150 e 171 ms) porque o bitsandbytes dequantiza no prefill. Em geração (tokens/s) o INT8 ganha: 18,1 contra 13,6 no Llama. Resultado registrado como é.

**11. Atingiram a meta de 4 GB de VRAM?**
Não. O menor pico medido foi 5,46 GiB (Llama, GPTQ). Um modelo de 8 bilhões em INT4 já ocupa ~5 GiB só de pesos. Registramos como resultado negativo.
*Limite:* a medição é de `max_memory_allocated` na DGX (memória unificada); em placa de consumo pode diferir.

**12. Por que o AWQ foi tão pior no Qwen3 (+13,8%) que o GPTQ (+4,4%)?**
Descartamos dois defeitos nossos: a calibração (blocos de 2048 mudaram a PPL no máximo 0,2%) e o zero-point (AWQ simétrico deu 7,9620 contra 7,9623). E o padrão aparece na literatura: Zheng et al. (2025) reportam no Qwen3-8B-Base com grupo 128 GPTQ +3,3% e AWQ +16,0%. Dizer: "reproduz o padrão publicado; a causa dentro do algoritmo não foi isolada por nós".
*Não dizer:* que o AWQ é pior em geral. No Llama os dois ficam próximos (+8,8% e +9,6%).

**13. Passaram do limiar de 3% de degradação?**
O limiar de 3% é só referência de interpretação, não critério de aceitação. Nenhum INT4 ficou abaixo dele (o melhor foi +4,4%); a literatura (grupo 128) reporta +6,6% a +8,2% no Llama 3 8B (Huang et al., 2024) e +3,3% a +16,0% no Qwen3-8B-Base (Zheng et al., 2025); conferido nas fontes.

## D. Validade das medições

**14. Quão confiáveis são os tokens/s?**
Três séries de 10 repetições por configuração, com mediana e IQR. A variação entre séries ficou abaixo de 2% e a ordem entre as configurações não muda. Outros processos usavam a GPU em parte do tempo; isso está registrado nos JSON de cada medição. Uma réplica falhou por falta de memória e foi refeita em outro horário.
*Limite:* uma máquina só (DGX Spark); a comparação estatística formal é do Jean.

**15. O que vocês não conseguem afirmar?**
Que o resultado vale para outros tamanhos ou famílias de modelo, para tarefas além de perplexidade, para outras placas (o Avell e o notebook ainda não foram medidos) ou para outro corpus de calibração além do C4.

**16. O que faria com mais tempo?**
Medir em hardware de consumo, avaliar tarefas (ex.: benchmarks de raciocínio), testar o Instruct, isolar a causa do AWQ no Qwen3 dentro do algoritmo, medir `word_perplexity` com o harness e comparar com outras bibliotecas de AWQ.
