# Relatório técnico do primeiro treinamento autônomo — Neuroevolution Racing

## 1. Objetivo deste documento

Este documento registra o primeiro experimento longo em que os agentes do projeto **Neuroevolution Racing** foram deixados aprender a dirigir por evolução, sem uma trajetória de referência programada diretamente no controlador e sem backpropagation.

A intenção é preservar não apenas os melhores resultados, mas também:

- decisões de projeto que funcionaram;
- hipóteses iniciais que se mostraram falsas ou incompletas;
- comportamentos emergentes inesperados;
- limitações do fitness;
- indícios de generalização;
- possíveis platôs e ótimos locais;
- limitações da observação fornecida à rede;
- erros de implementação ou de desenho experimental encontrados durante o processo;
- perguntas que ficaram abertas para experimentos futuros.

Este texto deve ser tratado como um **relatório técnico exploratório da primeira execução**, e não como evidência estatística definitiva. Houve uma execução longa, com uma única linhagem evolutiva principal e uma seed determinística. Para um artigo, os resultados mais importantes deverão posteriormente ser repetidos com múltiplas seeds e protocolos de benchmark fixos.

---

## 2. Arquitetura do agente

A política de controle usada neste experimento é uma MLP feed-forward com arquitetura:

$$\huge
6 \rightarrow 8 \rightarrow 2
$$

As duas camadas densas utilizam `tanh`.

A rede possui 74 parâmetros treináveis:

$$\huge
(6 \cdot 8 + 8) + (8 \cdot 2 + 2) = 74
$$

O controlador não aprende por gradiente. Todos os pesos e biases são codificados diretamente em um `Genome`, e o algoritmo genético é responsável por criar novas combinações desses parâmetros.

### 2.1 Entradas

O contrato externo da rede possui seis entradas:

1. sensor de distância à esquerda, aproximadamente `+60°`;
2. sensor de distância à esquerda, aproximadamente `+30°`;
3. sensor frontal, `0°`;
4. sensor de distância à direita, aproximadamente `-30°`;
5. sensor de distância à direita, aproximadamente `-60°`;
6. velocidade normalizada.

Portanto, a rede não recebe diretamente:

- posição absoluta na pista;
- coordenadas do centro da pista;
- índice do segmento atual;
- direção ideal da curva;
- distância até um waypoint;
- identificação da pista;
- trajetória de referência;
- ação desejada fornecida por um controlador externo.

O progresso na pista é usado pelo ambiente para avaliação, mas não é dado como observação privilegiada à MLP.

### 2.2 Saídas

As duas saídas representam:

1. direção/steering;
2. aceleração.

Como `tanh` é utilizada na saída, ambas são naturalmente limitadas aproximadamente ao intervalo `[-1, 1]`.

### 2.3 Consequência importante

A rede é uma política **reativa e sem memória**. Sua decisão é uma função apenas do vetor sensorial atual:

$$\huge
a_t = \pi(o_t)
$$

Não existe um estado interno temporal:

$$\huge
h_t
$$

como haveria em uma RNN, GRU ou LSTM. Essa característica será importante na discussão sobre possível “cegueira” ou aliasing perceptual.

---

## 3. Algoritmo evolutivo

O treinamento é feito por neuroevolução. O algoritmo genético não conhece carros, pistas ou redes neurais: ele manipula apenas vetores de `f32`.

A relação conceitual é:

```text
Genome
  ↓
parâmetros da MLP
  ↓
MLP
  ↓
controle do carro
  ↓
episódio
  ↓
fitness
  ↓
seleção/crossover/mutação
  ↓
novo Genome
```

Na configuração utilizada ao longo do desenvolvimento, a população foi de 500 indivíduos e a evolução utilizou seleção por torneio, elitismo, crossover uniforme e mutação gaussiana.

A principal vantagem dessa separação foi permitir alterar quase completamente a lógica de avaliação sem modificar a implementação do algoritmo genético ou da MLP.

---

## 4. Evolução da função de avaliação

A parte que mais mudou durante o experimento não foi a rede neural nem o algoritmo genético. Foi a definição de **“dirigir bem”**.

Essa foi uma das principais lições do projeto.

---

## 5. Primeira tentativa: distância máxima em uma janela fixa

A primeira função de fitness relevante era essencialmente:

$$\huge
F = \max_t d(t)
$$

onde `d(t)` era o progresso ao longo da pista durante uma janela fixa de aproximadamente 20 segundos.

O resultado foi inicialmente animador. A população claramente evoluía:

- o melhor indivíduo avançava cada vez mais;
- a média da população aumentava fortemente;
- indivíduos aleatórios da geração zero já produziam alguns comportamentos parcialmente úteis;
- a seleção conseguia preservar e combinar políticas melhores.

Porém, depois de poucas dezenas de gerações, surgiu um comportamento inesperado.

### 5.1 Comportamento emergente indesejado

Os agentes **não aprenderam a evitar a parede**.

Em vez disso, aprenderam uma política aproximadamente equivalente a:

1. acelerar;
2. fazer a curva de maneira agressiva;
3. atingir a parede;
4. continuar acelerando e esterçando;
5. eventualmente se desprender da parede;
6. continuar avançando.

A física permitia recuperação após contato com a parede. Como o fitness recompensava apenas a maior distância alcançada, uma colisão não era necessariamente ruim.

Isso é um caso claro de **especificação incompleta do objetivo**.

O algoritmo não “trapaceou” uma regra. Ele otimizou corretamente a regra que recebeu.

A intenção humana era:

> percorrer a pista de forma competente.

O objetivo matemático fornecido era:

> aumentar o máximo progresso dentro da janela.

Esses dois objetivos não eram equivalentes.

### 5.2 Primeira presunção falsa

**Presunção:** se distância for recompensada, evitar paredes emergirá naturalmente porque colisões atrapalham o progresso.

**Resultado:** falso.

A recuperação física era barata o suficiente para que bater pudesse fazer parte de uma política competitiva.

Essa observação motivou a criação de episódios individuais e encerramento por colisão.

---

## 6. Segunda fase: episódios formais

A avaliação foi reorganizada para que cada carro possuísse um episódio independente.

Um episódio passou a terminar por:

- `Completed`: completou a pista;
- `Collision`: colisão translacional real com a parede;
- `Stalled`: ausência de progresso significativo por tempo demais;
- `Timeout`: limite máximo de segurança.

O timeout deixou de representar o objetivo e passou a existir apenas para impedir episódios infinitos.

Essa mudança teve um efeito qualitativo importante: uma colisão deixou de ser uma etapa recuperável da estratégia e passou a encerrar a avaliação daquele indivíduo.

---

## 7. Fitness multiobjetivo simples

O score de um episódio passou a usar três informações principais:

- progresso normalizado;
- velocidade útil de progresso;
- estado de conclusão/colisão.

A forma utilizada durante a execução longa foi aproximadamente:

$$\huge
F =
1.0 \cdot p
+
0.4 \cdot v
+
0.45 \cdot C
-
0.08 \cdot K
$$

onde:

- \(p\) é o progresso normalizado;
- \(v\) é a velocidade útil normalizada;
- \(C=1\) quando o episódio é concluído;
- \(K=1\) quando termina por colisão.

O progresso passou a ser normalizado a partir do ponto real de spawn:

$$\huge
p =
\frac{d_{\max} - d_0}
{L - d_0}
$$

onde:

- $\large d_{\max}$ é o maior progresso atingido;
- $\large d_0$ é o progresso inicial;
- $\large L$ é o comprimento total da pista.

Isso evita atribuir fitness inicial gratuito simplesmente porque o carro não começa matematicamente em `track_distance = 0`.

### 7.1 Garantia de conclusão

O `completion_bonus` foi escolhido de modo a ser maior do que todo o peso de velocidade:

$$\huge
0.45 > 0.40
$$

Assim, na configuração válida, um episódio concluído deve superar qualquer episódio não concluído no limite teórico da fórmula, preservando conclusão como objetivo hierarquicamente importante.

---

## 8. Velocidade útil em vez de velocidade física

Uma decisão importante foi **não recompensar diretamente o módulo da velocidade física do carro**.

Um veículo pode possuir velocidade elevada e ainda estar:

- apontando para a parede;
- preso;
- indo para trás;
- se movimentando sem ganhar progresso na pista.

Foi usado, em vez disso, o ganho de melhor progresso por tempo:

$$\huge
v_{\text{útil}}
=
\frac{d_{\max} - d_0}{t}
$$

Esse valor mede quão rapidamente o indivíduo transforma tempo de simulação em avanço efetivo pela pista.

A ideia foi correta e os resultados posteriores mostram que ela fornece um sinal evolutivo útil. Entretanto, a normalização dessa velocidade criou um novo problema, discutido mais adiante.

---

## 9. Segunda presunção falsa: stall muito permissivo

A primeira configuração de stall relevante utilizava aproximadamente:

```text
significant_progress_epsilon = 6 u
stall_timeout = 2.5 s
```

Isso significa que um carro podia evitar `Stalled` fazendo apenas:

$$\huge
\frac{6}{2.5}=2.4\ u/s
$$

de progresso significativo.

Na prática, esse limiar era extremamente permissivo.

### 9.1 Comportamento emergente

Depois que colisões passaram a ser caras, a população começou a favorecer políticas excessivamente conservadoras:

- pouco risco;
- pouca aceleração;
- velocidade baixa;
- poucos stalls;
- grande número de timeouts.

A população havia aprendido outra solução válida para o objetivo:

> se bater é ruim, reduza a agressividade até quase eliminar a colisão.

### 9.2 Correção

A regra de stall foi endurecida para:

```text
significant_progress_epsilon = 60 u
stall_timeout = 2.0 s
```

Além disso, o peso da velocidade foi elevado de `0.20` para `0.40`.

Essa alteração teve um efeito forte: a velocidade útil dos campeões praticamente dobrou ao longo do treinamento.

---

## 10. Protocolo multi-pista

O treinamento deixou de usar uma única pista por geração.

O conjunto de treino utilizado é formado por:

- Interlagos;
- Red Bull Ring;
- Catalunya;
- Silverstone;
- Monza.

A cada geração são sorteadas três pistas de treino.

A regra importante é:

> todos os 500 indivíduos daquela geração são avaliados exatamente nas mesmas três pistas.

Isso evita que diferenças de fitness sejam causadas apenas pela diferença de dificuldade de pistas sorteadas individualmente.

O fitness final de um indivíduo é a média dos scores obtidos nessas três pistas.

---

## 11. Validação held-out

As pistas de validação foram mantidas fora do fitness:

- Spa;
- Suzuka;
- Monaco.

Depois que a geração termina as pistas de treino, o campeão é avaliado em uma pista de validação.

Esse score:

- é salvo;
- aparece nas métricas;
- pode ser visualizado;
- **não participa da seleção**;
- **não altera o fitness**;
- **não participa do crossover ou mutação**.

Essa separação foi uma decisão metodológica importante, porque permite observar algum grau de generalização sem transformar as pistas de validação em pistas de treino disfarçadas.

---

## 12. Checkpoints

Durante a execução foram preservados 139 checkpoints entre as gerações 10 e 3200.

Cada checkpoint contém informações suficientes para reconstruir a MLP e também metadados experimentais, incluindo:

- geração;
- arquitetura;
- ativações;
- genome;
- fitness do campeão;
- fitness médio da população;
- velocidade útil média;
- taxa de conclusão;
- motivos de término;
- pistas de treino usadas;
- pista de validação;
- score de validação;
- progresso de validação;
- velocidade útil em validação;
- parâmetros da função de avaliação.

Isso transformou a execução longa em uma sequência de estados históricos reaproveitáveis, permitindo posteriormente comparar redes antigas e novas sem refazer o treinamento.

---

# 13. Resultados quantitativos gerais

## 13.1 Checkpoints

No primeiro checkpoint salvo, geração 10:

| Métrica | Geração 10 |
|---|---:|
| Fitness do campeão | 0,685 |
| Fitness médio da população | 0,224 |
| Velocidade útil média do campeão | 144,0 u/s |
| Conclusões nas 3 pistas do campeão | 0/3 |

No checkpoint da geração 3200:

| Métrica | Geração 3200 |
|---|---:|
| Fitness do campeão | 1,850 |
| Fitness médio da população | 1,510 |
| Velocidade útil média do campeão | 293,6 u/s |
| Conclusões nas 3 pistas do campeão | 3/3 |

Entre esses dois checkpoints:

- a velocidade útil do campeão aumentou aproximadamente **104%**;
- o fitness médio da população aumentou aproximadamente **6,7 vezes**;
- o campeão saiu de nenhuma conclusão nas pistas sorteadas para frequentemente completar todas.

Esses números não devem ser interpretados isoladamente como uma curva perfeitamente comparável, porque as três pistas sorteadas mudam entre gerações. Ainda assim, a melhora comportamental é inequívoca.

---

## 13.2 Marcos observados

| Geração | Fitness campeão | Média população | Velocidade útil | Conclusão treino | Validação |
|---:|---:|---:|---:|---:|---|
| 10 | 0,685 | 0,224 | 144,0 u/s | 0% | Spa: colisão, 4,5% |
| 20 | 1,217 | 0,749 | 192,3 u/s | 33,3% | Suzuka: timeout, 70,1% |
| 50 | 1,387 | 0,969 | 220,9 u/s | 33,3% | Suzuka: colisão, 31,3% |
| 100 | 1,441 | 1,182 | 241,7 u/s | 33,3% | Suzuka: timeout, 93,4% |
| 140 | 1,461 | 1,217 | 251,9 u/s | 33,3% | **Suzuka concluída** |
| 300 | 1,671 | 1,410 | 276,3 u/s | 66,7% | Suzuka: colisão |
| 600 | **1,850** | 1,447 | 294,2 u/s | **100%** | Spa concluída |
| 1000 | 1,693 | 1,445 | 289,0 u/s | 66,7% | Spa concluída |
| 1500 | 1,697 | 1,457 | 302,6 u/s | 66,7% | Suzuka concluída |
| 2000 | 1,697 | 1,410 | 299,5 u/s | 66,7% | Monaco: 89,2% |
| 2500 | 1,850 | 1,571 | 305,7 u/s | 100% | Spa concluída |
| 3200 | **1,850** | 1,510 | 293,6 u/s | **100%** | Spa concluída |

O primeiro checkpoint com validação concluída ocorreu na geração 140.

O primeiro checkpoint salvo em que o campeão concluiu as três pistas de treino sorteadas e atingiu o teto de fitness ocorreu na geração 600.

---

# 14. Resultados por pista no estágio tardio

O trecho do terminal preservado cobre aproximadamente as gerações 2371–3218. Nesse intervalo foi possível agregar estatísticas da população inteira por pista.

| Pista | Aparições no log | Conclusão média da população | Velocidade útil média | Timeout médio por 500 |
|---|---:|---:|---:|---:|
| Monza | 496 | **89,78%** | **307,20 u/s** | 0,00 |
| Red Bull Ring | 489 | **76,67%** | 296,59 u/s | 0,04 |
| Silverstone | 533 | **70,07%** | 256,38 u/s | 4,07 |
| Interlagos | 508 | **26,44%** | 300,53 u/s | 255,65 |
| Catalunya | 518 | **~0,00%** | 290,00 u/s | 345,81 |

O número de episódios `Stalled` nesse estágio era praticamente zero.

Esses dados mostram que o agente não tinha uma capacidade uniforme de dirigir. A dificuldade era altamente dependente da geometria da pista.

---

# 15. Catalunya como principal gargalo de treino

Catalunya é o caso mais extremo.

Mesmo depois de milhares de gerações:

- a velocidade útil média da população era alta;
- o melhor score da pista ficava próximo de `1,39–1,40`;
- a taxa de conclusão era praticamente zero;
- a maioria dos indivíduos terminava por timeout ou colisão.

Como um indivíduo não concluído já recebia até `+0,4` pelo termo saturado de velocidade, um score próximo de `1,40` é compatível com um carro que alcança **quase todo o progresso possível**, mas não consegue efetivamente satisfazer a condição de conclusão.

Isso sugere um gargalo localizado muito perto do fim da volta.

Há pelo menos quatro explicações plausíveis:

1. existe uma curva final particularmente difícil para essa política;
2. a rede chega muito perto do fim, mas produz uma ação recorrente que a impede de cruzar corretamente;
3. existe uma limitação de percepção no trecho;
4. existe algum detalhe de geometria/progress tracking que deve ser verificado.

A hipótese de bug não deve ser assumida, mas Catalunya merece inspeção visual específica com checkpoints maduros.

---

# 16. Generalização para pistas de validação

Os resultados de validação ficaram altamente dependentes da pista.

Considerando apenas checkpoints a partir da geração 500:

| Pista de validação | Avaliações | Concluídas | Taxa de conclusão | Progresso médio |
|---|---:|---:|---:|---:|
| Spa | 37 | 37 | **100%** | 100% |
| Suzuka | 35 | 35 | **100%** | 100% |
| Monaco | 37 | 0 | **0%** | **89,02%** |

Isso é uma evidência importante.

A população não apenas memorizou uma única pista de treino. O campeão foi capaz de dirigir duas pistas held-out com extrema consistência.

Ao mesmo tempo, a falha em Monaco foi igualmente consistente.

Depois de amadurecer, o comportamento em Monaco ficou aproximadamente estável:

```text
progresso ≈ 88–90%
motivo = Timeout
```

Essa regularidade é mais interessante do que uma falha aleatória. Ela sugere um **bottleneck comportamental específico e reproduzível**.

---

# 17. O agente generaliza?

A resposta correta para este experimento é:

> **há evidência forte de generalização, mas ainda não há evidência suficiente para uma conclusão estatística geral.**

A evidência favorável é clara:

- Spa e Suzuka não participam da seleção;
- a partir de aproximadamente geração 500, os checkpoints avaliados nessas pistas as completam consistentemente;
- a mesma arquitetura e os mesmos pesos são utilizados sem adaptação online.

Por outro lado:

- houve apenas uma execução evolutiva principal;
- a validação por checkpoint usava apenas uma pista sorteada;
- Monaco permaneceu não resolvida;
- pistas diferentes têm dificuldades muito diferentes.

Portanto, o experimento demonstra que **generalização é possível com a representação atual**, mas não demonstra que a política generaliza para qualquer pista.

---

# 18. A hipótese de “cegueira”

Uma suspeita importante é que seis entradas sejam sensorialmente pobres demais para algumas situações.

Essa hipótese é plausível, mas **não foi provada**.

O fato de a mesma rede aprender:

- Monza;
- Red Bull Ring;
- Silverstone;
- parte considerável de Interlagos;
- Spa;
- Suzuka;

mostra que o vetor de seis entradas contém informação suficiente para comportamentos bastante sofisticados.

Portanto, os dados atuais não justificam afirmar:

> “seis entradas não são suficientes”.

Eles justificam afirmar:

> “há pistas e trechos nos quais a política atual de seis entradas e MLP sem memória apresenta um gargalo persistente”.

---

## 18.1 Aliasing perceptual

O problema técnico mais provável associado à “cegueira” não é simplesmente pouca quantidade de números. É **aliasing perceptual**.

Duas situações fisicamente diferentes podem produzir vetores sensoriais muito parecidos.

Exemplo conceitual:

```text
situação A:
parede a 30u à esquerda
frente livre
parede a 80u à direita
velocidade 200u/s

situação B:
mesmos cinco raycasts aproximadamente
mesma velocidade
mas curva seguinte exige decisão oposta
```

Para uma MLP feed-forward:

$$\huge
o_A \approx o_B
$$

implica:

$$\huge
\pi(o_A) \approx \pi(o_B)
$$

A rede não sabe em qual parte da pista está e não se lembra de como chegou ali.

Esse tipo de ambiguidade pode criar um teto de desempenho mesmo que a MLP possua capacidade matemática suficiente para aproximar funções complexas.

---

## 18.2 Informações ausentes

Com a observação atual, a rede não possui explicitamente:

- yaw rate;
- orientação relativa ao eixo da pista;
- velocidade lateral;
- histórico recente dos sensores;
- histórico das próprias ações;
- curvatura além do horizonte dos raycasts;
- identificação contextual do trecho.

Isso não significa que essas entradas devam ser adicionadas imediatamente. É possível melhorar bastante a política mantendo exatamente seis entradas e duas saídas.

---

# 19. Melhorias possíveis mantendo 6 entradas e 2 saídas

A dimensão da interface não precisa mudar para realizar experimentos muito melhores.

## 19.1 Aumentar capacidade interna

Comparar:

```text
6 -> 8 -> 2
6 -> 16 -> 2
6 -> 32 -> 2
6 -> 8 -> 8 -> 2
6 -> 16 -> 8 -> 2
```

O contrato permanece `6 in / 2 out`, mas a função que pode ser representada pela rede se torna mais rica.

O experimento atual não permite concluir se `6 -> 8 -> 2` é pequeno demais.

## 19.2 Alterar geometria dos cinco sensores

Sem adicionar entradas, é possível experimentar:

- ângulos diferentes;
- maior alcance;
- menor alcance;
- distribuição não simétrica;
- sensores mais concentrados na direção frontal;
- transformação não linear da distância.

O conjunto atual de ângulos pode ser adequado para curvas suaves e insuficiente para certas geometrias.

## 19.3 Melhorar a codificação da distância

Uma distância linear normalizada pode dar resolução demais longe da parede e de menos perto dela.

Podem ser testadas transformações monotônicas que preservam uma entrada por sensor, como:

$$\huge
x = \frac{1}{1+d/k}
$$

ou outras curvas que aumentem sensibilidade em distâncias criticamente pequenas.

## 19.4 Rever a semântica da velocidade

Caso a entrada de velocidade use apenas magnitude, vale testar uma codificação assinada ou outra representação que permita distinguir movimento para frente e para trás sem aumentar o número de entradas.

## 19.5 Introduzir memória sem aumentar a dimensão externa

Uma política recorrente poderia continuar recebendo seis entradas e produzindo duas saídas, mas manter estado interno.

Isso mudaria a arquitetura do controlador, porém não o contrato sensorial:

$$\huge
(o_t, h_{t-1}) \rightarrow (a_t, h_t)
$$

Esse experimento seria particularmente relevante se Monaco/Catalunya apresentarem situações de aliasing perceptual.

---

# 20. O principal erro da função de fitness tardia: saturação da velocidade

A execução longa revelou um erro técnico importante.

A velocidade útil era normalizada como aproximadamente:

$$\huge
v =
\mathrm{clamp}
\left(
\frac{v_{\text{útil}}}{120},
0,
1
\right)
$$

Porém, após algumas centenas de gerações, os campeões frequentemente atingiam:

```text
250–307 u/s
```

Assim:

```text
120 u/s -> v = 1
200 u/s -> v = 1
250 u/s -> v = 1
300 u/s -> v = 1
```

Todas essas políticas recebiam os mesmos `+0,4` de velocidade.

### 20.1 Consequência

O peso de velocidade foi aumentado para produzir pressão evolutiva por maior velocidade, mas os indivíduos maduros já estavam acima do ponto de saturação.

Portanto, **o fitness deixou de distinguir indivíduos rápidos de indivíduos muito rápidos**.

---

# 21. Teto matemático do fitness

A fórmula tardia possui teto:

$$\huge
F_{\max}
=
1.0
+
0.4
+
0.45
=
1.85
$$

Esse valor aparece repetidamente nos checkpoints.

Logo, quando um indivíduo:

- completa a pista;
- alcança velocidade útil acima da normalização;

ele recebe exatamente o mesmo score que qualquer outro indivíduo nessas condições.

Isso produz um problema importante:

> o algoritmo evolutivo pode continuar encontrando políticas fisicamente melhores, mas a função de fitness não consegue mais observá-las.

Esse fenômeno deve ser chamado de **saturação da função objetivo**, e não de convergência genética.

---

# 22. Fitness plateau não é necessariamente ótimo local

O gráfico mostra um platô claro depois das primeiras centenas de gerações.

É tentador chamar isso de “ótimo local”.

Isso seria prematuro.

Há pelo menos três tipos de platô que podem produzir a mesma aparência visual:

### 22.1 Platô da função objetivo

A função de fitness satura.

É comprovadamente um fator neste experimento porque `1.85` é o máximo matemático e foi atingido repetidamente.

### 22.2 Platô evolutivo / ótimo local

A população pode convergir para uma região do espaço de pesos em que mutações pequenas tendem a piorar a política, embora existam políticas melhores em regiões mais distantes.

Isso é um ótimo local no sentido evolutivo.

É plausível, especialmente nos comportamentos persistentes de Monaco e Catalunya, mas os dados atuais não conseguem separar esse efeito da saturação do fitness.

### 22.3 Platô representacional

A política pode ter atingido o melhor comportamento possível com:

- aqueles seis sinais;
- aquela geometria de sensores;
- uma MLP sem memória;
- oito neurônios ocultos.

Também é plausível, mas não demonstrado.

### 22.4 Conclusão

O experimento apresenta um **platô real de desempenho medido**, mas não há evidência suficiente para afirmar que ele representa um ótimo local puro do algoritmo genético.

Antes dessa conclusão, é necessário remover a saturação do fitness e executar novos experimentos.

---

# 23. Normalização de velocidade recomendada

Uma alternativa é substituir o clamp duro por uma função assintótica:

$$\huge
v_n =
\frac{v}{v+k}
$$

Por exemplo, com \(k=120\):

| Velocidade | Valor normalizado |
|---:|---:|
| 60 u/s | 0,333 |
| 120 u/s | 0,500 |
| 240 u/s | 0,667 |
| 300 u/s | 0,714 |
| 600 u/s | 0,833 |

Essa função possui vantagens importantes:

- permanece limitada;
- nunca satura exatamente em `1`;
- continua recompensando melhorias em alta velocidade;
- oferece retornos decrescentes;
- mantém o termo de velocidade controlado.

Outra alternativa, especialmente para indivíduos que concluem, é utilizar tempo de volta como critério secundário.

---

# 24. O gráfico de fitness de treino não é uma curva pura de aprendizado

Outro ponto importante é que as três pistas de treino mudam entre gerações.

Portanto:

$$\huge
F_{g}
$$

e:

$$\huge
F_{g+1}
$$

podem ter sido medidos em conjuntos de pistas diferentes.

Uma queda de fitness entre duas gerações não significa necessariamente regressão genética.

O padrão tardio deixa isso evidente:

- conjuntos sem Catalunya frequentemente permitem `best = 1.85`;
- conjuntos contendo Catalunya frequentemente ficam próximos de `1.697`.

Assim, a curva combina dois fatores:

$$\large
\text{fitness observado}
=
\text{qualidade da população}
+
\text{dificuldade do subconjunto sorteado}
$$

O sorteio continua sendo útil para treinamento, mas é ruim como métrica longitudinal principal.

---

# 25. Benchmark fixo recomendado

Para análise científica, a solução é adicionar um benchmark periódico independente da seleção.

A cada checkpoint, por exemplo:

1. pegar o campeão;
2. avaliá-lo em **todas** as pistas de treino;
3. avaliá-lo em **todas** as pistas held-out;
4. armazenar resultados sem devolver esses scores ao GA.

Isso produziria séries diretamente comparáveis:

```text
geração
  ↓
Interlagos score
Red Bull Ring score
Catalunya score
Silverstone score
Monza score
Spa score
Suzuka score
Monaco score
```

Os 139 checkpoints existentes já permitem fazer esse benchmark retrospectivamente sem repetir o treinamento.

Esse provavelmente será um dos conjuntos de dados mais úteis para o artigo.

---

# 26. Média de fitness pode esconder o “elo fraco”

O fitness de treino é a média das três pistas sorteadas.

Isso possui uma consequência:

> desempenho excelente em duas pistas pode compensar uma falha persistente na terceira.

Esse efeito pode contribuir para o caso Catalunya.

Considere:

```text
Pista A: excelente
Pista B: excelente
Pista C: ruim
```

Se a média ainda for suficiente para vencer a população, a pressão para resolver C pode ser menor do que parece.

Algumas alternativas experimentais futuras:

- média + termo do pior score;
- média harmônica;
- prioridade lexicográfica por número de pistas completadas;
- fitness baseado em média, mas com bônus por completar todas;
- amostragem mais frequente das pistas atualmente difíceis;
- curriculum adaptativo.

Essas opções devem ser testadas cuidadosamente porque modificam o problema de otimização.

---

# 27. Pressão seletiva variável entre pistas

Como apenas três das cinco pistas de treino aparecem por geração, uma pista difícil não exerce pressão em todas as gerações.

Isso aumenta diversidade de cenários, porém introduz ruído evolutivo.

Uma política que é ótima para três pistas sorteadas pode tornar-se campeã mesmo sendo ruim em uma quarta pista que ficou de fora.

Esse mecanismo pode explicar parte do padrão de:

```text
100% de conclusão do campeão
↓
66,7%
↓
100%
```

sem que a população tenha realmente “esquecido” como dirigir.

---

# 28. Aumento de velocidade funcionou

Uma das mudanças com efeito mais claro foi o aumento da pressão por velocidade.

A velocidade útil do campeão foi:

- geração 10: ~144 u/s;
- geração 100: ~242 u/s;
- geração 300: ~276 u/s;
- geração 600: ~294 u/s;
- região tardia: normalmente ~290–305 u/s.

A melhora ocorreu rapidamente e depois estabilizou.

Isso mostra duas coisas:

1. a rede consegue aprender uma política agressiva sem necessariamente retornar ao comportamento inicial de bater constantemente;
2. a função de fitness realmente influencia o estilo de condução emergente.

O estilo “conservador demais” não era uma propriedade inevitável da arquitetura. Era em grande parte consequência da função objetivo.

---

# 29. Evolução comportamental observada

O processo pode ser resumido qualitativamente em quatro estágios.

### Estágio 1 — aleatoriedade com stepping stones

Alguns indivíduos da geração zero já conseguiam, por acaso, manter-se na pista por trechos e realizar curvas.

Isso é importante porque a população inicial já continha “stepping stones”: políticas incompletas, mas melhores que ruído puro.

### Estágio 2 — exploração da colisão

Com fitness baseado essencialmente em distância, surgiu uma estratégia agressiva que aceitava contato com paredes.

### Estágio 3 — conservadorismo

Quando colisão passou a encerrar o episódio, sobreviver tornou-se mais importante. Surgiram políticas lentas que evitavam colisões mas frequentemente terminavam por timeout.

### Estágio 4 — direção rápida e relativamente generalista

Com stall mais agressivo e maior peso de velocidade:

- velocidade útil subiu;
- as colisões dos campeões caíram;
- várias pistas passaram a ser completadas;
- Spa e Suzuka passaram a ser generalizadas;
- restaram gargalos persistentes em Catalunya e Monaco.

Esse histórico mostra que o “estilo de direção” emergente mudou diretamente em resposta ao desenho do fitness.

---

# 30. Otimização não produz intenção

Uma conclusão conceitual importante deste experimento é que não existe uma noção interna de:

- “dirigir bonito”;
- “ser seguro”;
- “seguir a linha ideal”;
- “evitar bater”;
- “ser rápido”.

Existem apenas pressões seletivas.

Quando a função recompensava distância, surgiu agressividade.

Quando colisão virou término, surgiu conservadorismo.

Quando velocidade ganhou peso, surgiu aceleração.

Esse comportamento é exatamente o esperado de um sistema de otimização e é um exemplo didático da importância de **objective design** em sistemas de aprendizado.

---

# 31. Erros e limitações encontrados

## 31.1 Fitness inicial incompleto

Apenas distância em tempo fixo não representava adequadamente condução.

## 31.2 Colisão recuperável

A física permitia incorporar a parede à estratégia.

## 31.3 Stall permissivo

O limiar inicial permitia políticas extremamente lentas.

## 31.4 Saturação da velocidade

`120 u/s` tornou-se muito baixo depois que a população amadureceu.

## 31.5 Teto de fitness

O score máximo de `1.85` removeu resolução seletiva entre indivíduos excelentes.

## 31.6 Métrica longitudinal contaminada pelo sorteio de pistas

Fitness de gerações diferentes não é medido necessariamente no mesmo conjunto de pistas.

## 31.7 Validação aleatória única por checkpoint

Uma única pista held-out sorteada produz uma curva de validação com grande dependência da identidade da pista.

## 31.8 Apenas uma execução longa

Não existem múltiplas seeds independentes suficientes para medir variância e significância.

## 31.9 CSV com um registro malformado

O checkpoint da geração 2875 contém um valor textual de fitness com espaço indevido (`1.6976 199`), indicando um pequeno problema no processo de exportação/formatação.

Antes de análises automatizadas finais, o exportador deve garantir que todas as colunas numéricas sejam válidas.

## 31.10 Warnings do Bevy em execução extremamente longa

Durante execuções aceleradas muito longas apareceram warnings indicando sistemas que não haviam executado dentro da janela máxima de change detection.

O treinamento continuou, mas isso deve ser eliminado ou compreendido antes dos experimentos formais para evitar uma variável de infraestrutura potencialmente confusora.

---

# 32. Ameaças à validade

## 32.1 Seed única

Uma execução pode ter encontrado uma trajetória evolutiva particularmente boa ou ruim.

## 32.2 Dependência das pistas escolhidas

Resultados são fortemente influenciados pela geometria das pistas.

## 32.3 Fitness modificado durante o desenvolvimento

Resultados de fases antigas e tardias não devem ser misturados como se pertencessem ao mesmo experimento.

## 32.4 Score saturado

Parte do platô tardio é um artefato conhecido da função de fitness.

## 32.5 Validação parcial

Spa e Suzuka foram resolvidas, Monaco não. Isso demonstra generalização parcial, não universal.

## 32.6 Ausência de baseline

Ainda não há comparação formal com:

- controlador manual;
- controlador heurístico;
- arquitetura maior;
- mais sensores;
- outra função de fitness;
- treinamento em pista única;
- treino em todas as pistas simultaneamente.

---

# 33. Experimentos futuros prioritários

## 33.1 Benchmark retrospectivo dos 139 checkpoints

Executar cada checkpoint selecionado em todas as pistas:

```text
10
50
100
200
300
500
600
800
1000
1500
2000
2500
3000
3200
```

ou em todos os 139 checkpoints, se o custo for baixo.

Registrar por pista:

- progresso;
- conclusão;
- tempo;
- velocidade útil;
- motivo de término.

Esse experimento exige zero novo treinamento e permite reconstruir a evolução da capacidade.

## 33.2 Remover saturação dura da velocidade

Comparar clamp atual com normalização assintótica.

## 33.3 Inspecionar Catalunya e Monaco

Usar Champion Mode, sensores desenhados e telemetria para localizar exatamente o trecho de falha.

## 33.4 Aumentar arquitetura mantendo 6/2

Comparar, com múltiplas seeds:

```text
6 -> 8 -> 2
6 -> 16 -> 2
6 -> 32 -> 2
6 -> 8 -> 8 -> 2
```

## 33.5 Alterar apenas geometria dos sensores

Manter cinco distâncias + uma velocidade e mudar ângulos/alcance.

## 33.6 Testar treino com todas as pistas

Comparar:

```text
RandomSubset(3)
vs
All(5)
```

Isso mede o custo/benefício de reduzir ruído e aumentar pressão sobre pistas difíceis.

## 33.7 Múltiplas seeds

Para o artigo, executar várias seeds para cada configuração principal.

Métricas recomendadas:

- mediana do score final;
- intervalo interquartil;
- taxa de conclusão por pista;
- geração da primeira conclusão;
- área sob a curva de benchmark;
- velocidade útil;
- variância entre seeds.

---

# 34. Hipóteses específicas para testar

### H1 — A arquitetura 6→8→2 não é o principal gargalo

Se redes maiores com as mesmas seis entradas melhorarem Catalunya/Monaco, a limitação atual é em parte capacidade interna.

### H2 — A observação é parcialmente ambígua

Se redes maiores não melhorarem os mesmos trechos, mas uma política com memória melhorar, há evidência de aliasing perceptual.

### H3 — O plateau tardio é principalmente causado pelo fitness

Se uma função de velocidade não saturante produzir progresso adicional sem mudar arquitetura/sensores, o platô anterior era majoritariamente objetivo, não representacional.

### H4 — Catalunya sofre com pressão seletiva insuficiente

Se `All(5)` ou uma amostragem ponderada por dificuldade resolver Catalunya, a falha não era cegueira.

### H5 — Monaco representa generalização fora da distribuição mais difícil

Se Monaco continuar falhando com diferentes funções de fitness e arquiteturas, sua geometria pode exigir informação que os cinco sensores atuais não fornecem adequadamente.

---

# 35. O que os dados já permitem afirmar

Com cautela, este primeiro experimento permite afirmar:

1. Neuroevolução foi capaz de produzir políticas de direção não triviais a partir de pesos aleatórios.
2. O algoritmo genético recebeu sinal suficiente para melhorar substancialmente a população.
3. Uma MLP `6 -> 8 -> 2` com `tanh` consegue controlar o veículo em várias pistas.
4. A política conseguiu completar pistas held-out que nunca participaram diretamente do fitness.
5. O comportamento emergente é extremamente sensível à função de avaliação.
6. Distância isolada não foi suficiente para produzir direção desejável.
7. Penalizar colisão sem incentivar velocidade produziu conservadorismo.
8. A pressão adicional por velocidade aumentou fortemente a velocidade útil.
9. A normalização dura da velocidade criou saturação e um teto artificial no fitness.
10. Há forte heterogeneidade de dificuldade entre pistas.
11. Catalunya e Monaco apresentam gargalos persistentes.
12. O platô tardio não pode ser interpretado diretamente como ótimo local.
13. Seis entradas não se mostraram claramente insuficientes; ao contrário, produziram generalização relevante.
14. Ainda existe bastante espaço para melhorar desempenho mantendo exatamente seis entradas e duas saídas.

---

# 36. O que ainda não pode ser afirmado

Ainda não há base suficiente para afirmar:

- que `6 -> 8 -> 2` é a melhor arquitetura;
- que oito neurônios ocultos são suficientes;
- que seis entradas são o limite mínimo;
- que a população chegou ao ótimo global;
- que a população chegou definitivamente a um ótimo local;
- que Monaco é impossível com a observação atual;
- que Catalunya possui bug;
- que a generalização observada é robusta a qualquer seed;
- que 3000 gerações são necessárias;
- que o fitness atual é adequado para experimentos finais.

---

# 37. Interpretação do possível ótimo local

Se, após corrigir a saturação do fitness, a população voltar a estabilizar em um comportamento que:

- resolve as mesmas pistas;
- falha nos mesmos trechos;
- resiste a milhares de mutações;
- apresenta baixa diversidade;
- melhora pouco mesmo com score ainda sensível;

então a hipótese de ótimo local ficará muito mais forte.

Um teste útil seria comparar:

- diversidade genética média da população;
- distância entre genomes;
- taxa de melhoria do campeão;
- taxa de aceitação indireta de mutantes superiores;
- comportamento ao aumentar temporariamente sigma de mutação;
- comportamento ao reiniciar parte da população.

Um ótimo local evolutivo é uma propriedade do **landscape induzido pela combinação entre representação, fitness e operadores genéticos**, não apenas da pista.

---

# 38. Observação sobre convergência

Convergência pode significar coisas diferentes:

- convergência de fitness;
- convergência genética;
- convergência comportamental.

Neste experimento há forte evidência de convergência do **fitness observado** e do **comportamento médio** após algumas centenas de gerações.

Não há, com os dados atuais, uma medição explícita de diversidade genética suficiente para afirmar convergência completa dos genomes.

Adicionar métricas de diversidade genética em experimentos futuros pode ajudar a distinguir:

```text
população geneticamente presa
vs
fitness incapaz de distinguir melhorias
```

---

# 39. Curiosidades técnicas reveladas pelo experimento

## 39.1 Redes aleatórias não eram completamente inúteis

Com 500 indivíduos, a geração inicial já continha políticas que, por acaso, produziam sequências úteis de steering/aceleração.

Isso cria pontos de partida para seleção incremental.

## 39.2 Aprendizado não foi monotônico por geração

Mesmo com elitismo, a métrica exibida pode cair porque a prova muda quando as pistas sorteadas mudam.

## 39.3 Generalização apareceu antes de perfeição no treino

A primeira conclusão registrada em validação ocorreu antes de uma solução universal nas pistas de treino.

Isso mostra que “resolver todo o treino” não é requisito para transferência parcial.

## 39.4 Pistas rápidas não são necessariamente mais difíceis

Monza teve alta velocidade útil e alta taxa de conclusão.

A dificuldade parece depender mais de geometria e observabilidade do que simplesmente de velocidade máxima.

## 39.5 Alta velocidade não garante conclusão

Catalunya combina velocidade útil alta com conclusão praticamente nula.

Isso é um forte exemplo de por que uma única métrica não descreve competência.

---

# 40. Dados e figuras a preservar para o artigo

Recomenda-se manter junto deste relatório:

1. CSV completo dos 139 checkpoints.
2. Log tardio por pista, especialmente gerações 2371–3218.
3. Gráfico de fitness por geração.
4. Gráfico de velocidade útil.
5. Gráfico de progresso/conclusão de validação.
6. Gráfico de score de validação separado por motivo de término.
7. Gráfico de motivos de término do campeão nas pistas de treino.
8. Checkpoints `.ron` originais.
9. Seed e configuração completa do algoritmo genético.
10. Versão/commit do código correspondente ao experimento.

Para publicação, as figuras devem deixar explícito quando uma métrica é:

- de treino;
- de validação;
- do campeão;
- da população;
- agregada entre pistas;
- dependente de subconjunto sorteado.

---

# 41. Recomendações para o artigo futuro

Uma estrutura natural para o artigo seria:

1. **Introdução**
   - neuroevolução;
   - controle autônomo;
   - objetivo de estudar emergência e generalização.

2. **Metodologia**
   - ambiente 2D;
   - sensores;
   - MLP;
   - algoritmo genético;
   - fitness;
   - treino e validação;
   - pistas.

3. **Desenvolvimento do fitness**
   - distância;
   - exploração da parede;
   - episódios;
   - conservadorismo;
   - velocidade.

4. **Resultados**
   - curvas;
   - checkpoints;
   - desempenho por pista;
   - generalização.

5. **Análise de falhas**
   - Catalunya;
   - Monaco;
   - saturação;
   - possíveis ótimos locais;
   - observabilidade.

6. **Ablations**
   - arquitetura;
   - sensores;
   - fitness;
   - seleção de pistas.

7. **Discussão**
   - reward/objective design;
   - partial observability;
   - generalização;
   - limitações.

8. **Conclusão**

---

# 42. Conclusão geral deste primeiro experimento

O resultado mais importante não é que um carro tenha completado uma determinada pista.

O resultado mais importante é que um sistema extremamente pequeno:

```text
5 sensores de distância
+
1 medida de velocidade
        ↓
MLP 6 -> 8 -> 2
        ↓
steering + aceleração
```

foi capaz, através de neuroevolução, de sair de políticas praticamente aleatórias e desenvolver:

- progressão consistente;
- redução de colisões;
- maior velocidade;
- conclusão de várias pistas;
- transferência para pistas held-out;
- comportamentos específicos e reproduzíveis.

Ao mesmo tempo, a execução mostrou que a qualidade do comportamento não depende apenas da rede.

A função objetivo moldou profundamente o que emergiu.

As falhas observadas — colisões inicialmente exploradas, excesso de conservadorismo depois, saturação posterior do termo de velocidade — não são apenas “bugs” do experimento. Elas são resultados informativos sobre a interação entre:

$$
\text{observação}
+
\text{representação}
+
\text{fitness}
+
\text{evolução}
+
\text{ambiente}
$$

A suspeita de que a rede esteja “cega” demais é tecnicamente plausível, especialmente para Monaco e Catalunya, mas os resultados também mostram que seis entradas já são suficientes para uma quantidade surpreendente de comportamento e generalização.

Portanto, antes de aumentar a quantidade de informação fornecida à IA, há um caminho experimental forte em manter o mesmo contrato `6 -> ? -> 2` e investigar:

- capacidade interna;
- geometria/codificação dos sensores;
- memória;
- função de fitness;
- pressão sobre pistas difíceis;
- diversidade genética;
- benchmarks fixos.

O platô após milhares de gerações não deve ser interpretado como evidência de que “a IA chegou ao limite”. Parte desse platô é sabidamente causada pelo próprio instrumento usado para medi-la.

O próximo passo científico mais importante é criar uma avaliação que continue distinguindo boas políticas de políticas melhores e, em seguida, testar os checkpoints existentes em um benchmark fixo por pista.

---

## Apêndice A — Configuração de avaliação da execução longa

```text
Arquitetura:
6 -> 8 -> 2
Tanh -> Tanh
74 parâmetros

População:
500 indivíduos

Treino:
5 pistas disponíveis
3 sorteadas por geração
mesmo subconjunto para todos os indivíduos

Validação:
1 pista held-out sorteada para o campeão
não influencia evolução

Finalização do episódio:
Completed
Collision
Stalled
Timeout

maximum_episode_duration = 60 s
stall_timeout = 2 s
significant_progress_epsilon = 60 u

Fitness:
progress_weight = 1.00
speed_weight = 0.40
collision_penalty = 0.08
completion_bonus = 0.45
progress_speed_normalization = 120 u/s

Fitness máximo da configuração:
1.85
```

---

## Apêndice B — Pistas

### Treino

```text
Interlagos
Red Bull Ring
Catalunya
Silverstone
Monza
```

### Validação

```text
Spa
Suzuka
Monaco
```

---

## Apêndice C — Checklist antes dos experimentos formais

- [ ] remover saturação dura da velocidade;
- [ ] definir benchmark fixo;
- [ ] corrigir exportação CSV;
- [ ] investigar warnings de execução longa do Bevy;
- [ ] avaliar Catalunya visualmente;
- [ ] avaliar Monaco visualmente;
- [ ] rodar checkpoints antigos no benchmark;
- [ ] registrar commit exato;
- [ ] registrar seed;
- [ ] repetir com múltiplas seeds;
- [ ] testar pelo menos uma arquitetura alternativa;
- [ ] testar pelo menos uma configuração alternativa de sensores;
- [ ] registrar diversidade genética, se possível;
- [ ] separar claramente resultados exploratórios dos resultados finais.
