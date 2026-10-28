---
tipo: reuniao
data: 2026-08-24
hora_inicio: "13:30"
duracao_min: 45
origem: auto
assunto: "[akcit] - Reunião berg gate 1"
natureza: gestao
entidade: ""
ferramentas: ["wikitext2"]
tags: ["pesquisa", "IA", "quantização", "cronograma", "dissertação"]
participantes: ["João Vitor", "Berne"]
---

Assunto: akcit - Reunião berg gate 1

## 📌 Resumo

A reunião tratou do progresso de um estudo acadêmico sobre quantização e compressão de modelos para dispositivos com recursos limitados, focando na comparação entre modelos comprimidos e bases em FP16. Foram discutidos a organização da documentação, o uso do dataset wikitext2 como benchmark, a estrutura da dissertação (relação entre introdução e revisão bibliográfica) e ajustes no cronograma de projeto. Como medida de qualidade, foi decidida a implementação de um checkpoint obrigatório antes dos gates para validação e formatação.

## ✅ Itens de ação

- [ ] Ajustar pontos pendentes no cronograma.
- [ ] Preencher os estágios do projeto.
- [ ] Implementar o novo fluxo de validação pré-gate.

## 🗣️ Tópicos e decisões

- Uso do wikitext2 como dataset de benchmark.
- Inclusão de uma etapa extra de verificação (checkpoint) antes dos gates para evitar retrabalho e garantir a qualidade da entrega.

## 📝 Transcrição completa

> [!quote]- Transcrição
> Bom dia, boa tarde, boa tarde, tudo bem? Bom dia, boa tarde
> , tudo bem? Bom dia, tudo bem? Sim, deixa eu só mudar o meu
>  positivo aqui de fone de ouvido. Sim, por favor, prazer. Be
> leza, já arrumei o meu aqui. Joia, contigo, né? Tá tranquilo
> . Isso, é o primeiro gate nosso. Ainda não. Eu não sei qual
>  o prazo, né? Porque era até semana passada, só que a gente
>  não conseguiu vaga em agendas. O ideal seria passar por
>  essa orientação, ajustar os pontos e
>  já ir pro próximo, né? Menos mal, porque pra gente não tava
>  claro que era nesse formato, ainda mais pelos outros mód
> ulos, né? Que você perdesse qualquer prazo dos outros mód
> ulos, você era desclassificado. As disciplinas, isso. Os
>  gates ali, né? Não, vamos pra orientação aqui, mais pé no
>  chão. Nos próximos, talvez, não podiam ir, mas nesse, não.
>  Vamos entender. Eu, João Vitor, sou engenheiro eletricista,
>  só que eu atuo como gerente de desenvolvimento. Sou
>  desenvolvedor há bastante tempo.
>  E pretendo utilizar o nosso estudo de caso aqui pra
>  compressão de modelos pra rodar em dispositivos mais,
>  dispositivos com menos recursos, isso, com mais limitações.
>  Porque eu tô numa área aqui gerenciando o time de inovação
>  também e isso é algo que aqui na empresa que eu trabalho,
>  que é uma empresa de software, isso vai ser bem utilizado
>  por conta de custos. Tá no mudo, João? Não, acho que faz
>  sentido compartilhar. Beleza. Segue com o outro. Já? Já
>  compartilhou.
>  Compartilha o outro. Esse daí é o inicial que a gente usou,
>  o outro já tá formatado na... tá na mesma pasta lá. Isso. O
>  que tá escrito preenchido é o início que a gente começou a
>  fazer e depois o outro é o que a gente foi transbordando as
>  informações. Porque esse aí tá com muita informação, o
>  outro já tá mais... e o outro tá no formato que tem que ser
>  a entrega também. Vai ali, ó. Tá vendo ali onde tá a docx?
>  Tem uma estrela e tem uma pastinha do lado. Mais pra dire
> ita sua. Ali, ó. Lá onde você tava mesmo. Tem uma pastinha
>  do lado lá, ó. Deixa eu te mandar o link, peraí. Mais fácil
> . Mandar o link do... Aí, ó. Clica.
>  Entra lá mesmo. Clica nessa... nesse ícone aí. Esse. Isso
>  aí. Esse aí vai te enviar pro drive. É o do outro lado. É o
>  do outro. Deixa eu ver. Muda aí a... o seu lá em cima lá.
>  Tá como ícone. Muda pra lista. Fica mais fácil. Não. Canto
>  direito. Aí. O de baixo. O de baixo. Preenchido é o que
>  tava aberto. Aí. Esse aí. Esse aí já tá no formato que a
>  gente tem que fazer a entrega. Sim. Quer começar, João?
>  Como que você quer fazer? Isso. Isso. Isso. Isso como
>  modelo de partida. Não, é isso mesmo. Como modelo de part
> ida aí com...
>  Com... Com o wikitext2 como dataset de... De benchmark pra
>  execução. Isso. Não. É bem isso mesmo. Ótimo.
> Isso, isso. Descer não precisa rodar não, a gente pega o
>  benchmark já que deve estar lá, que costuma ficar lá na
>  Hang Face e aí, isso. Aí pode ajustar essa parte aí. Não,
>  aí é o cronograma já. Parabéns, demais. Tem não, tem não,
>  tem não. Isso, aí descreve o cronograma, tem lá os estágios
>  4 e 5, né, que a gente precisa preencher eles também e tir
> ando esses estágios aí, os de cima nós já preenchemos.
>  Brinca. Testes e coisas que a gente vai fazendo durante a
>  pesquisa e foi aprendendo durante o curso como um todo,
>  entendeu? O que a gente espera, o que a gente espera só
>  antes do João falar, para contextualizar melhor, a gente
>  espera usar os benchmarks existentes com datasets conhe
> cidos para executar. O nosso trabalho de pesquisa é a quant
> ização, a compressão de modelo com algumas técnicas também
>  são conhecidas para tentar
>  chegar num resultado diferente do que o benchmark atrás e
>  diminuindo mais a quebra, a quebra não, desculpa, o tamanho
>  do modelo em si. O nosso objetivo é esse, não é montar um
>  dataset para poder fazer o fine tuning ou algo do gênero,
>  entendeu, Berg? Isso, tá bom. Isso. A gente pretende usar
>  os mesmos parâmetros, os mesmos datasets com o modelo quant
> izado, que é o modelo comprimido. A gente vai usar a técnica
>  para comprimir o modelo, vai pegar esse modelo comprimido e
>  testar ele com o modelo base. E aí nós vamos fazer a comp
> aração dos benchmarks para mostrar se a quantização para um
>  modelo de um exemplo de 8 bilhões de parâmetros, ele rodar
>  quantizado com o tamanho ou um espaço, né, de VRAM ali, de
>  4 GB, de 2 GB, se o nível de entrega dele tá próximo, tá
>  aceitável com os mesmos, com o mesmo dataset executado e os
>  mesmos parâmetros, se isso tá
>  não aceitável comparado com o FP16. Pegar a mesma técnica e
>  rodar um exemplo, um Queen 3, tá, beleza, beleza, beleza,
>  beleza, entendido.
>  Isso.
>  Por que? Não, eu estou hoje a ver com o nosso.
>  Ai, mas eu quero dar todos os que.
>  Então depois você vai arrumar. Não é para misturar.
>  Beleza.
>  Bom, pode tocar, eu só tava confirmando, eu só tava concord
> ando.
>  Machado.
>  Berg, essa revisão bibliográfica aqui, ela é similar à
>  introdução do TCC?
>  Beleza, entendido.
>  Igual o que você acabou de comentar, que dela a gente vai
>  destrinchar para o demais,
>  que aí da introdução a gente faz o descritivo, né, a
>  dissertação em si.
>  .
>  .
> Beleza, beleza.
>  Beleza, vamos correr agora para a próxima.
>  A gente já está com isso tudo revisado.
>  E aí, o que vocês acham?
>  Talvez, acho que ir aí com a sua orientação, Berne.
>  Interessante a gente fazer um outro checkpoint antes do
>  gate para poder validar.
>  Isso aí para o gate com uma formatação melhor,
>  meio que quase que um de acordo do que já tentar adicar um
>  gate.
>  E quando o gate é reprovado, a gente tem que fazer tudo de
>  novo.
>  Como que funciona? Beleza, beleza então.
>  Beleza, isso. Beleza então. Beleza, muito obrigado.
>  Obrigado, Berne. Valeu, pessoal. Tchau, tchau. Um abraço.
