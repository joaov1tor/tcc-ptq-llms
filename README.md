# TCC — Quantização pós-treino (PTQ) de LLMs

Especialização GenAI (UFG/AKCIT). Compara FP16, INT8 (LLM.int8()), INT4 GPTQ e INT4 AWQ em Llama 3 8B e Qwen3-8B-Base: perplexidade no WikiText-2 e custo (disco, pico de memória, TTFT, tokens/s) na DGX Spark.

| Pasta | Conteúdo |
|---|---|
| `scripts/` | Quantização, perplexidade, custo e filas de execução (como rodaram na DGX) |
| `resultados/` | JSON de cada medição; consolidado em `resultados_benchmark_PTQ.xlsx` |
| `logs/` | Logs brutos das rodadas |
| `DIARIO.md` | Diário de tentativas, inclusive falhas e condições de medição |
| `redacao/` | Rascunhos de Método e Resultados |
| `guia/` | Guia de estudo, glossário de decisões, perguntas de banca |
| `gate3/` | Roteiro do Gate 3 |
| `reunioes/` | Resumo e transcrição das reuniões e aulas do TCC (índice no README da pasta) |
| `berg/` | Resumo de checkpoint (material interno) |
