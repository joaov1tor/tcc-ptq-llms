---
tipo: reuniao
data: 2026-08-31
hora_inicio: "18:02"
duracao_min: 25
origem: auto
assunto: "Apresentação de TCC - Quantização de Modelos"
natureza: pessoal
entidade: ""
ferramentas: ["nvidia-dgx-spark", "llm", "slm"]
tags: ["tcc", "ia", "quantização", "slm", "benchmark", "machine-learning"]
participantes: ["João Vitor", "Jean", "Jackson"]
---

Assunto: Apresentação de TCC - Quantização de Modelos

## 📌 Resumo

O estudante apresentou seu trabalho de conclusão de curso (TCC) focado na avaliação de técnicas de quantização em Small Language Models (SLMs). O estudo visa medir a redução de consumo de memória e a velocidade de inferência ao transitar entre os formatos FP16, FP8 e FP4. Foi detalhado o uso do dataset wiki-text2 e a infraestrutura de hardware disponível para os testes. O objetivo final é fornecer um benchmark técnico sobre o efeito da quantização pós-treino. A sessão foi encerrada com agradecimentos e a indicação de novos encontros para tirar dúvidas.

## ✅ Itens de ação

- (nenhum)

## 🗣️ Tópicos e decisões

- (nenhuma registrada)

## 📝 Transcrição completa

> [!quote]- Transcrição
> **Falante 1** [00:00]: Vou começar a
> **Falante 2** [00:00]: apresentar meu TCC aqui, filho. E aí?
> **Falante 1** [00:04]: Estava mexendo um pouco no material lá. Uhum, uhum. Eu dei uma re-revisada aqui. Só estou achando muito aqui as referências bibliográficas. Só que agora já tem cima da hora. Isso. Re-vê, dá uma diminuída que eu achei muito. Isso. Vamos ver o que dá para a gente implementar daquelas técnicas lá. Dá para a gente rodar. Olá, boa noite, tudo bom? É os três... Chamar João, algum responde. É os três Jotas aqui. Tudo bem. Os quatro, né? Porque você também é com Jota,
> **Falante 3** [00:35]: então... É isso, um gate.
> **Falante 4** [00:39]: 3G1. Isso. Já. Beleza. Isso aí. Está tranquilo. Vamos lá. Boa noite. Eu sou o João Vitor. Hoje, atualmente, eu estou como gestor do time de inovação aqui da
> **Falante 2** [00:53]: empresa que eu trabalho. E também sou coordenador... Estou como gerente de desenvolvimento de um time de fábrica de software aqui.
> **Falante 1** [00:58]: E o intuito... Sou da área de tecnologia também. Sou engenheiro eletricista. E o intuito aqui
> **Falante 4** [01:04]: do que
> **Falante 1** [01:05]: eu
> **Falante 4** [01:05]: escolhi esse tema aí do TCC é basicamente para ter modelos também, mas visando em modelos menores, pensando ali em SLMs também, não só LLMs, que consigam ser especialistas naquele segmento. E uma das frentes que é inicial é a quantização para depois fazer um fine-turning, que eu penso no futuro já conseguir incorporar isso para apoiar aqui dentro, que eu trabalho numa consultoria da TOTUS e nós provemos serviços para N segmentos de empresa diferente. Temos uma vasta quantidade de documentos, banco de conhecimento aqui para capacitar os modelos para isso. Vai lá, Jean. Não, você explicou bem a parte que você trouxe ali dos formatos, do que a gente espera entregar ali, dos modelos que nós vamos utilizar como base. E só desce um pouco aí, só para mostrar o que nós vamos entregar, o entregável. Aqui são as hipóteses, são três hipóteses que nós vamos avaliar, que é a quantização 28, reduz o consumo de memória aproximadamente 50%. Basicamente, trazendo as minhas palavras aqui para facilitar, as hipóteses que nós vamos trazer é pega do int 16 para o int 8, o int 16 a gente roda o dataset e pega o valor. Faz a quantização para int 8 e depois para int 4. Feito isso, rodando o mesmo dataset, a gente vai ver a diferença, a qualidade ali, a perplexidade entre cada um deles para ver se ela é inferior ou não, porque não adianta eu pegar um modelo de um determinado tamanho e reduzir ele demais, mas ele fica burro, ele não conseguir me responder em cima do que o outro modelo responde. E isso é o desperdício computacional. Basicamente, é isso que a gente quer responder com essas três hipóteses. Medindo a inferência do modelo, porque reduzindo a memória que ele vai consumir, em tese a inferência dele tem que ser aumentada, que é a parte de vazão de tokens por segundo, quanto que ele consegue responder, inferindo. Desce mais um pouco. O objetivo geral da pesquisa é a entrega de um benchmark. Aí está explicando todo o objetivo da pesquisa, que é avaliar o efeito da quantização do pós-treino, que é bem o que a gente comentou acima, e aqui ele está detalhando bastante sobre o processo. E o entregável é um benchmark entre o FP16, FP8 e FP4, sendo um relatório técnico de benchmark. Isso. É que nós vamos fazer o benchmark, né? É, nós escolhemos um dataset. Isso, isso, isso. Nós escolhemos um. É o wiki... Cadê o nome dele ali? De cabeça não lembro, não. É o wiki-text2. Isso, isso, isso, isso, isso. Alex, a parar. E só um adendo do porquê que a gente pegou isso aí e vai tentar fazer isso. Falando de recurso computacional, eu tenho um servidor meu aqui com 6GB de VRAM, que é o que a gente vai rodar depois. E eu tenho também aqui na empresa uma DGX Spark da NVIDIA. E aí eu... Nela a gente consegue. Eu já venho brincando algumas coisas, vim brincando um pouco com o Fine Toney também. Aí eu já tenho... A gente já tem ali o hardware para simular e ir testando a inferência, ver o que dá para fazer ali, validando a inferência. Isso. Isso. Não. No
> **Falante 1** [04:41]: estado da arte, não por conta do tamanho dos modelos. O modelo estado da arte acaba que o tamanho
> **Falante 5** [04:47]: dele, por ser maior, a gente pelo menos não pesquisou. Focou mais nesses menores para trazer,
> **Falante 6** [04:54]: porque eles estão bem longe do estado da arte. Bem longe do estado da arte, né? Beleza. Muito bom. Obrigado. Sim, dez minutos. O
> **Falante 1** [05:02]: quê? Beleza, beleza. Então tá bom.
> **Falante 4** [05:04]: Muito obrigado.
> **Falante 1** [05:05]: Na
> **Falante 4** [05:05]: próxima
> **Falante 1** [05:05]: a
> **Falante 4** [05:05]: gente... A gente deve se cruzar ainda, né? Porque ainda tem vários encontros, né? Então... Isso. É, mas a gente pode passar para você também para a tirada de dúvida, né? Isso. Entendi. Boa. Boa. Valeu, Jackson. Obrigadão. Um abraço. Tchau.
