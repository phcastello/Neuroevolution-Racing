---
title: "Neuroevolution Racing — Fase 2: Sweep de Arquiteturas, Generalização e Race"
aliases:
  - "Relatório Fase 2 Neuroevolution Racing"
  - "Sweep de Arquiteturas"
date: 2026-08-31
status: "exploratório concluído"
project: "Neuroevolution Racing"
phase: 2
tags:
  - neuroevolution
  - genetic-algorithm
  - mlp
  - reinforcement-learning-adjacent
  - racing
  - generalization
  - robustness
  - partial-observability
  - experiment
  - tcc
related:
  - "[[Neuroevolution Racing]]"
  - "[[Experimento 01 - Primeiro Treinamento Autônomo]]"
  - "[[Artigo Neuroevolution Racing]]"
---

# Neuroevolution Racing — Fase 2
## Sweep de arquiteturas, generalização e comparação em Race

> [!abstract] Resumo
> Esta fase abandonou a pergunta simples de **“a IA consegue aprender a dirigir?”** e passou a investigar **como a arquitetura da MLP afeta desempenho, generalização e robustez comportamental**.
>
> O experimento executou **8 arquiteturas**, todas com o mesmo contrato externo de **6 entradas e 2 saídas**, por **1000 gerações**, população de **500 indivíduos**, mesmo GA, mesma função de avaliação, mesmas pistas disponíveis e mesma física.
>
> O resultado principal foi inesperado: **o fitness final de treino ficou quase indistinguível entre as arquiteturas**, enquanto o comportamento em validação e no modo Race divergiu fortemente.
>
> A arquitetura que apresentou a melhor combinação observada de velocidade, generalização e estabilidade foi:
>
> $$
> \boxed{6 \rightarrow 16 \rightarrow 16 \rightarrow 2}
> $$
>
> Ela não foi necessariamente a mais rápida em toda pista, mas foi a política que visualmente se manteve com maior frequência no topo, sem colisões observadas no conjunto de comparações realizadas, e obteve **97,5% de conclusão em validação nas gerações 800–999**.
>
> Ao mesmo tempo, os testes revelaram uma limitação cada vez mais evidente: **a política parece estar limitada mais pela informação disponível do que pela capacidade da MLP**. Assistindo aos agentes, decisões de curva, frenagem e posicionamento frequentemente parecem depender fortemente de um único raycast ou de uma combinação local muito pequena de sensores.

---

# 1. Objetivo desta fase

A primeira fase do projeto mostrou que uma MLP extremamente pequena, treinada por neuroevolução, conseguia aprender comportamento de direção não trivial.

A segunda fase teve quatro objetivos principais:

1. comparar arquiteturas de diferentes larguras e profundidades;
2. verificar se aumento de capacidade melhora treino e generalização;
3. observar checkpoints lado a lado em um modo **Race** comum;
4. identificar limitações que não eram visíveis olhando apenas fitness.

A principal variável experimental foi a arquitetura interna da rede.

O contrato externo permaneceu:

```text
6 inputs
    ↓
X hidden layers
    ↓
2 outputs
```

As seis entradas continuaram sendo:

```text
5 raycasts de distância
+
velocidade normalizada
```

As saídas continuaram sendo:

```text
steering
acceleration
```

Todas as hidden layers e a saída utilizaram `tanh`.

---

# 2. Configuração experimental

## 2.1 Algoritmo genético

A configuração do GA permaneceu:

```text
population_size = 500

elite_fraction = 0.05
tournament_size = 3

crossover_probability = 0.80

mutation_probability = 0.10
mutation_sigma = 0.15

initial_gene_min = -1.0
initial_gene_max = 1.0
```

Os parâmetros da MLP são diretamente codificados no genome.

O tamanho do genome foi calculado automaticamente a partir de:

```text
Architecture::parameter_count()
```

e não ficou preso aos 74 parâmetros da arquitetura original.

---

## 2.2 Avaliação

Configuração utilizada:

```text
maximum_episode_duration = 180 s

laser:
    grace_period = 3 s
    acceleration = 30 u/s²
    maximum_speed = 130 u/s

progress_weight = 1.00
speed_weight = 0.40
collision_penalty = 0.08
completion_bonus = 0.45

progress_speed_half_saturation = 120 u/s
```

A velocidade útil passou a ser normalizada por:

$$\huge
v_n = \frac{v}{v+k}
$$

com:

$$\huge
k = 120
$$

removendo a antiga saturação dura em 120 u/s.

---

## 2.3 Pistas

### Treino

```text
Monza
Red Bull Ring
Interlagos
Catalunya
Silverstone
```

A cada geração 3 das 5 pistas são sorteadas e aplicadas igualmente a todos os indivíduos.
### Validação held-out

```text
Spa
Suzuka
Monaco
```

A validação do campeão não entra no fitness nem na evolução.

---

# 3. Arquiteturas avaliadas

| Arquitetura | Parâmetros |
|---|---:|
| `6 → 8 → 2` | 74 |
| `6 → 16 → 2` | 146 |
| `6 → 8 → 8 → 2` | 146 |
| `6 → 32 → 2` | 290 |
| `6 → 16 → 16 → 2` | 418 |
| `6 → 16 → 32 → 2` | 722 |
| `6 → 32 → 16 → 2` | 786 |
| `6 → 32 → 32 → 2` | 1346 |

> [!important] Comparação particularmente valiosa
> `6 → 16 → 2` e `6 → 8 → 8 → 2` possuem **exatamente 146 parâmetros**.
>
> Isso cria uma comparação quase controlada entre:
>
> - uma hidden layer mais larga;
> - duas hidden layers menores;
>
> sem confundir o resultado simplesmente com número total de parâmetros.

---

# 4. Volume de dados produzido

Cada arquitetura executou:

```text
1000 gerações
×
3 pistas de treino por geração
×
500 indivíduos
```

Além da validação do campeão.

Os logs preservam, no total:

$$\huge
8 \times 1000 \times 3 = 24.000
$$

avaliações agregadas de pista no treinamento.

Também foram preservados:

- `metrics.csv` por arquitetura;
- `worker.log` por arquitetura;
- `run.ron` por arquitetura;
- checkpoints de melhores redes por geração;
- checkpoints de população para resume;
- resultados exportados de Race.

---

# 5. Resultado geral

## 5.1 O fitness de treino quase não separou as arquiteturas

Nas últimas 100 gerações, a média do **best training fitness** ficou aproximadamente:

| Arquitetura | Best fitness médio — gen 900–999 |
|---|---:|
| `6 → 16 → 16 → 2` | 1,7315 |
| `6 → 16 → 2` | 1,7299 |
| `6 → 16 → 32 → 2` | 1,7304 |
| `6 → 32 → 16 → 2` | 1,7312 |
| `6 → 32 → 2` | 1,7303 |
| `6 → 32 → 32 → 2` | 1,7304 |
| `6 → 8 → 2` | 1,7295 |
| `6 → 8 → 8 → 2` | 1,7306 |

A diferença é minúscula.

Se o experimento terminasse aqui, seria tentador concluir:

> “as arquiteturas são praticamente equivalentes.”

Isso estaria errado.

---

# 6. Generalização tardia

Considerando as gerações:

```text
800–999
```

a taxa de conclusão das validações held-out foi:

| Arquitetura | Validação concluída |
|---|---:|
| `6 → 16 → 16 → 2` | **97,5%** |
| `6 → 32 → 32 → 2` | **97,5%** |
| `6 → 32 → 2` | 81,0% |
| `6 → 8 → 2` | 76,0% |
| `6 → 16 → 2` | 71,5% |
| `6 → 32 → 16 → 2` | 64,5% |
| `6 → 16 → 32 → 2` | 60,5% |
| `6 → 8 → 8 → 2` | **35,5%** |

> [!important] Resultado central
> Políticas com praticamente o mesmo fitness de treino apresentaram diferenças de mais de **60 pontos percentuais** de conclusão em validação.
>
> Portanto:
>
> $$
> \text{fitness de treino semelhante}
> \not\Rightarrow
> \text{comportamento equivalente}
> $$
>
> e:
>
> $$
> \text{fitness de treino semelhante}
> \not\Rightarrow
> \text{generalização semelhante}
> $$

---

# 7. A campeã: `6 → 16 → 16 → 2`

A arquitetura:

$$ \huge
6 \rightarrow 16 \rightarrow 16 \rightarrow 2
$$

foi a que apresentou a melhor combinação qualitativa observada.

Ela possui:

```text
418 parâmetros
```

contra:

```text
1346 parâmetros
```

da `6 → 32 → 32 → 2`.

Ou seja, usa apenas cerca de:

$$
31\%
$$

dos parâmetros da maior rede, aproximadamente **69% menos parâmetros**.

Mesmo assim, nas gerações 800–999, ambas obtiveram:

$$
97,5\%
$$

de conclusão de validação.

A `6 → 16 → 16 → 2` também foi a arquitetura que, no modo Race, foi percebida como a mais consistentemente competitiva:

- permaneceu repetidamente entre os líderes;
- não apresentou colisões nos testes visuais realizados;
- não dependeu de ser simplesmente a mais rápida em toda pista.

> [!note] Interpretação
> Isso sugere um possível **sweet spot de capacidade**.
>
> A rede parece suficientemente expressiva para lidar com diferentes geometrias sem pagar todo o custo comportamental observado nas arquiteturas maiores.

---

# 8. Validação por pista

Taxa de conclusão em validação nas gerações `800–999`:

| Arquitetura | Monaco | Spa | Suzuka |
|---|---:|---:|---:|
| `6 → 16 → 16 → 2` | **98,4%** | **95,8%** | **98,5%** |
| `6 → 32 → 32 → 2` | **100,0%** | **92,8%** | **100,0%** |
| `6 → 32 → 2` | 93,4% | 63,6% | 86,3% |
| `6 → 16 → 2` | 90,8% | 31,8% | 91,3% |
| `6 → 16 → 32 → 2` | 95,5% | 6,9% | 85,5% |
| `6 → 32 → 16 → 2` | 84,5% | 37,3% | 71,0% |
| `6 → 8 → 2` | 36,9% | 90,0% | 98,7% |
| `6 → 8 → 8 → 2` | **0,0%** | **0,0%** | **100,0%** |

---

# 9. O caso extremamente estranho da `6 → 8 → 8 → 2`

A `6 → 8 → 8 → 2` é provavelmente um dos resultados mais interessantes de toda a fase.

Ela possui os mesmos:

$$
146\text{ parâmetros}
$$

da `6 → 16 → 2`.

Porém, nas últimas 200 gerações:

```text
6 → 16 → 2
validation completion = 71,5%

6 → 8 → 8 → 2
validation completion = 35,5%
```

A especialização por pista é ainda mais extrema:

```text
Suzuka = 100%
Spa    = 0%
Monaco = 0%
```

Isso elimina uma interpretação simples de:

> “a rede não aprendeu.”

Ela claramente aprendeu uma política capaz de resolver Suzuka de forma consistente.

O problema é que essa política aparentemente depende de relações sensoriais que não transferem para Spa e Monaco.

> [!hypothesis] Hipótese — política baseada em proxies sensoriais
> A rede pode ter evoluído regras altamente dependentes de padrões locais dos cinco raycasts.
>
> Exemplo conceitual:
>
> ```text
> sensor +30° cai
> +
> sensor frontal permanece alto
> ↓
> ação específica
> ```
>
> Esse padrão pode funcionar perfeitamente em Suzuka e ser um péssimo proxy para a geometria encontrada em outra pista.

---

# 10. Rede rasa vs rede profunda com o mesmo número de parâmetros

A comparação:

```text
6 → 16 → 2
```

vs:

```text
6 → 8 → 8 → 2
```

é um dos experimentos mais valiosos desta fase.

Ambas:

```text
146 parâmetros
```

Mas o resultado foi fortemente diferente.

### Validação tardia

```text
6 → 16 → 2      = 71,5%
6 → 8 → 8 → 2   = 35,5%
```

### Conclusão da população em treino — gen 900–999

```text
6 → 16 → 2      = 74,7%
6 → 8 → 8 → 2   = 63,3%
```

Portanto:

> [!important]
> **Quantidade de parâmetros não explica sozinha o desempenho.**
>
> A forma como esses parâmetros são distribuídos em profundidade/largura altera fortemente o landscape funcional e evolutivo.

---

# 11. Robustez da população vs qualidade do campeão

Nas gerações 900–999, a taxa média de conclusão da **população inteira** nas três pistas de treino sorteadas foi:

| Arquitetura | Conclusão média da população |
|---|---:|
| `6 → 8 → 2` | **75,5%** |
| `6 → 32 → 2` | 74,8% |
| `6 → 16 → 2` | 74,7% |
| `6 → 16 → 32 → 2` | 70,9% |
| `6 → 16 → 16 → 2` | 67,7% |
| `6 → 8 → 8 → 2` | 63,3% |
| `6 → 32 → 32 → 2` | 55,7% |
| `6 → 32 → 16 → 2` | **45,2%** |

Esse resultado é quase o inverso do que seria esperado por uma interpretação ingênua de “rede maior = população melhor”.

A menor rede, `6 → 8 → 2`, apresentou uma das populações tardias mais robustas nas pistas de treino.

---

# 12. Gap entre campeão e população

Nas últimas 100 gerações:

| Arquitetura | Fitness campeão | Fitness médio população | Gap |
|---|---:|---:|---:|
| `6 → 16 → 2` | 1,730 | 1,449 | 0,280 |
| `6 → 32 → 2` | 1,730 | 1,443 | 0,287 |
| `6 → 8 → 2` | 1,730 | 1,422 | 0,308 |
| `6 → 16 → 32 → 2` | 1,730 | 1,377 | 0,354 |
| `6 → 16 → 16 → 2` | 1,732 | 1,340 | 0,392 |
| `6 → 8 → 8 → 2` | 1,731 | 1,292 | 0,439 |
| `6 → 32 → 32 → 2` | 1,730 | 1,145 | 0,586 |
| `6 → 32 → 16 → 2` | 1,731 | **0,984** | **0,747** |

> [!hypothesis] Hipótese — picos estreitos em arquiteturas maiores
> As arquiteturas grandes parecem conseguir produzir **indivíduos excelentes**, mas dentro de populações muito menos robustas.
>
> Uma interpretação possível:
>
> - genomes maiores criam mais direções em que mutação/crossover podem degradar a política;
> - soluções muito boas podem ocupar regiões pequenas do espaço de parâmetros;
> - elites preservam o campeão;
> - descendentes próximos não necessariamente preservam sua competência.
>
> Isso sugeriria um landscape com algo semelhante a:
>
> ```text
> redes menores:
> bacias largas de soluções razoáveis
>
> redes maiores:
> picos altos, porém estreitos
> ```
>
> Esta hipótese ainda precisa de métricas explícitas de diversidade genética.

---

# 13. Desempenho por pista de treino

Nas gerações 900–999:

## `6 → 16 → 16 → 2`

| Pista | Conclusão média da população | Useful speed |
|---|---:|---:|
| Monza | **87,5%** | 286,1 u/s |
| Interlagos | 72,6% | 262,1 u/s |
| Red Bull Ring | 68,0% | 266,3 u/s |
| Catalunya | 58,1% | 257,5 u/s |
| Silverstone | 52,8% | 226,2 u/s |

## `6 → 8 → 2`

| Pista | Conclusão média da população | Useful speed |
|---|---:|---:|
| Monza | **95,6%** | 286,7 u/s |
| Interlagos | 74,0% | 265,4 u/s |
| Red Bull Ring | 71,5% | 269,0 u/s |
| Catalunya | 70,3% | 250,6 u/s |
| Silverstone | 66,0% | 232,9 u/s |

A menor rede formou uma população muito consistente em treino.

Porém, essa estabilidade não se converteu em generalização uniforme para as validações.

---

# 14. O laser deixou de ser o gargalo tardio

No início da evolução, o laser possui papel importante na eliminação de políticas lentas.

No estágio tardio, porém, quase todas as mortes das arquiteturas menores e intermediárias se tornaram:

```text
Completed
ou
Collision
```

Para várias arquiteturas:

```text
laser elimination ≈ 0
timeout = 0
```

A conclusão relevante é:

> [!important]
> O gargalo tardio já não era velocidade mínima ou incapacidade de avançar.
>
> Ele passou a ser predominantemente **controle e decisão**.

Isso fortalece a hipótese de limitação de observabilidade.

---

# 15. Evolução da generalização ao longo das gerações

Taxa de conclusão da validação em blocos de 100 gerações:

| Arquitetura | 0–99 | 100–199 | 200–299 | 300–399 | 400–499 | 500–599 | 600–699 | 700–799 | 800–899 | 900–999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `6→16→16→2` | 52% | 79% | 75% | 86% | 82% | 87% | 90% | 95% | 97% | **98%** |
| `6→32→32→2` | 39% | 75% | 91% | 72% | 88% | 99% | 91% | 87% | 95% | **100%** |
| `6→32→2` | 28% | 71% | 79% | 85% | 85% | 79% | 73% | 64% | 78% | 84% |
| `6→8→2` | 25% | 72% | 87% | 91% | **95%** | 85% | 57% | 84% | 77% | 75% |
| `6→16→2` | 52% | **82%** | 80% | 66% | 56% | 66% | 56% | 61% | 74% | 69% |
| `6→16→32→2` | 53% | 56% | **78%** | 66% | 66% | 67% | 56% | 50% | 65% | 56% |
| `6→32→16→2` | 36% | 63% | 68% | **78%** | 63% | 70% | 34% | 48% | 53% | 76% |
| `6→8→8→2` | 25% | **54%** | 41% | 52% | 48% | 36% | 25% | 25% | 38% | 33% |

---

# 16. Generalização não monotônica

Várias arquiteturas melhoraram no treino enquanto a validação oscilou ou piorou.

Exemplos:

### `6 → 8 → 2`

A validação chegou a:

```text
95% em gen 400–499
```

e terminou em:

```text
75% em gen 900–999
```

### `6 → 8 → 8 → 2`

Chegou a:

```text
54% em gen 100–199
```

e caiu para:

```text
33% em gen 900–999
```

> [!note] Interpretação cuidadosa
> Isso não deve ser chamado automaticamente de **catastrophic forgetting**.
>
> O resultado demonstrado é:
>
> **generalização held-out não monotônica durante a evolução.**
>
> Para falar em esquecimento catastrófico seria necessário um protocolo específico que demonstrasse perda de uma habilidade anteriormente consolidada sob condições comparáveis.

---

# 17. Resultados do modo Race

O Race colocou checkpoints de arquiteturas diferentes na mesma pista e no mesmo ambiente.

Todos receberam:

- mesma posição inicial;
- mesmo heading;
- mesma velocidade;
- mesmos sensores;
- mesma física;
- ausência de colisão carro-carro.

Isso tornou diferenças de comportamento muito mais fáceis de perceber.

---

# 18. Race — Monza

Checkpoint de geração 999 de cada arquitetura:

| Arquitetura | Resultado | Tempo / progresso |
|---|---|---:|
| `6 → 32 → 16 → 2` | Completed | **37,966 s** |
| `6 → 16 → 16 → 2` | Completed | **38,700 s** |
| `6 → 32 → 32 → 2` | Completed | 38,800 s |
| `6 → 16 → 32 → 2` | Completed | 39,533 s |
| `6 → 8 → 2` | Completed | 39,700 s |
| `6 → 32 → 2` | Completed | 39,716 s |
| `6 → 8 → 8 → 2` | Completed | 40,350 s |
| `6 → 16 → 2` | Collision | 88,84% |

A arquitetura considerada campeã geral não foi a mais rápida em Monza.

Isso é importante.

> [!important]
> A seleção da `6 → 16 → 16 → 2` como melhor arquitetura observada não ocorreu apenas por tempo de volta.
>
> O destaque veio da combinação de:
>
> - velocidade;
> - consistência;
> - ausência de colisões observadas;
> - generalização;
> - estabilidade multi-pista.

---

# 19. Race — Monaco

| Arquitetura | Resultado | Tempo / progresso |
|---|---|---:|
| `6 → 16 → 32 → 2` | Completed | **69,850 s** |
| `6 → 32 → 2` | Completed | **69,884 s** |
| `6 → 16 → 16 → 2` | Completed | **69,917 s** |
| `6 → 32 → 32 → 2` | Completed | 70,751 s |
| `6 → 8 → 2` | Collision | 35,22% |
| `6 → 8 → 8 → 2` | Collision | 7,08% |
| `6 → 16 → 2` | Collision | 6,75% |
| `6 → 32 → 16 → 2` | Collision | 0,54% |

Três das arquiteturas que completaram ficaram separadas por apenas:

```text
~0,07 s
```

entre a primeira e a terceira.

Isso sugere que, quando diferentes políticas conseguem resolver a geometria da pista, suas performances podem convergir para tempos muito próximos.

A diferença principal deixa de ser:

> “qual é a mais rápida?”

e passa a ser:

> **“qual consegue resolver a geometria de forma robusta?”**

---

# 20. Resultado inesperado: segunda volta

Durante os testes visuais de Race foi observada uma anomalia extremamente interessante.

> [!observation] Observação visual ainda não formalizada em série estatística
> Na segunda volta:
>
> - a maioria das arquiteturas grandes bate em aproximadamente **3 das 5 pistas de treino**;
> - essas mesmas arquiteturas grandes não apresentaram o mesmo padrão nas pistas de validação;
> - arquiteturas menores não bateram nas pistas de treino observadas;
> - porém as pequenas batem com frequência em validações já na primeira volta.
>
> A `6 → 16 → 16 → 2` se destacou por não apresentar esse problema nos testes observados.

Essa observação ainda precisa ser exportada de forma quantitativa, mas é uma das descobertas qualitativas mais importantes da fase.

---

# 21. Hipótese: distribution shift na segunda volta

A MLP é feed-forward e não possui memória.

Ela não recebe:

```text
lap_number
```

Portanto ela **não sabe que está na segunda volta**.

Se o comportamento muda na volta 2, a causa deve estar no estado físico/sensorial.

No treinamento, o início de cada episódio sempre acontece aproximadamente em:

```text
posição = canônica
heading = canônico
velocidade = ~45 u/s
```

Na segunda volta real, o carro cruza a mesma região com algo como:

```text
velocidade = 200–300 u/s
offset lateral variável
heading variável
estado dinâmico herdado da curva anterior
```

Assim:

$$
s_{\text{largada de treino}}
\neq
s_{\text{passagem da volta 2}}
$$

A rede nunca foi diretamente selecionada para resolver o setor inicial a partir desse segundo conjunto de estados.

---

# 22. Hipótese: terminal-state blind spot

O treinamento original é de uma volta.

Quando o carro cruza a linha:

```text
Completed
```

o episódio termina.

A seleção não considera o que aconteceria:

```text
0,1 s
0,5 s
1 s
```

depois da linha de chegada.

Portanto duas políticas podem receber uma conclusão equivalente mesmo se:

```text
A:
cruza alinhada e estável

B:
cruza muito rápida, mal posicionada e já comprometida para a próxima curva
```

No treinamento de uma volta, B não é punida pela consequência posterior.

> [!hypothesis]
> Algumas redes grandes podem ter aprendido políticas que otimizam muito bem o estado **até a fronteira terminal**, mas produzem estados terminais pouco recuperáveis para operação contínua.

---

# 23. Teste proposto: canonical reset entre voltas

Um teste simples pode separar:

```text
incapacidade de dirigir o trecho
```

de:

```text
incapacidade de lidar com o estado herdado da volta anterior
```

Procedimento:

```text
completa volta 1
↓
reset para:
    posição canônica
    heading canônico
    velocidade 45 u/s
    controles neutros
↓
executa volta 2
```

Como a MLP é determinística e sem memória, espera-se que a volta seguinte reproduza essencialmente a primeira.

Se:

```text
volta contínua 2 → collision
```

mas:

```text
volta 2 com canonical reset → completed
```

a hipótese de distribution shift fica fortemente suportada.

---

# 24. Teste proposto: hot-start

Outro experimento importante:

Mesma rede.

Mesma posição.

Mesmo heading.

Variar apenas velocidade inicial:

```text
45 u/s
100 u/s
150 u/s
200 u/s
250 u/s
300 u/s
```

Medir:

- conclusão;
- colisão;
- progress;
- tempo;
- trajetória.

Isso cria uma métrica de:

> **robustez a estados iniciais fora da distribuição de treinamento**

que pode ser comparada entre arquiteturas.

---

# 25. Percepção visual mais importante desta fase: falta informação

Assistindo aos agentes, uma característica ficou muito evidente:

> [!observation]
> Muitas decisões parecem depender fortemente de **um único sensor de distância**.
>
> Em diferentes situações é possível observar comportamentos semelhantes a:
>
> ```text
> um raycast atinge certo padrão
> ↓
> freia
> ```
>
> ou:
>
> ```text
> um raycast muda
> ↓
> inicia steering da curva
> ```
>
> ou:
>
> ```text
> distância lateral específica
> ↓
> reposiciona o carro
> ```

Isso sugere que as redes estão utilizando os raycasts como **proxies** para estados do mundo que não estão diretamente observáveis.

---

# 26. Perceptual aliasing

A observação atual é:

$$
o_t =
[r_{+60},r_{+30},r_0,r_{-30},r_{-60},v]
$$

Duas situações fisicamente diferentes podem produzir observações muito semelhantes:

$$
o(s_1) \approx o(s_2)
$$

mesmo que exijam:

$$
a_1 \neq a_2
$$

Como a MLP é reativa:

$$
a_t=\pi(o_t)
$$

ela não possui informação para distinguir esses estados.

Esse fenômeno é conhecido como:

**perceptual aliasing**.

---

# 27. Mais capacidade não cria informação

Uma rede:

```text
6 → 32 → 32 → 2
```

é capaz de construir uma função muito mais complexa do que:

```text
6 → 8 → 2
```

Mas ambas recebem os mesmos seis números.

Portanto:

$$
\text{capacidade maior}
\neq
\text{observabilidade maior}
$$

A rede grande pode aprender relações mais sofisticadas entre os mesmos sensores.

Por exemplo:

```text
ray +30
+
ray 0
+
velocidade
↓
heurística extremamente específica
```

Mas ela não consegue reconstruir informação que não está implicitamente presente na observação.

---

# 28. Hipótese: redes grandes aprendem políticas mais específicas

Uma explicação coerente com os resultados é:

### Redes pequenas

```text
menos capacidade
↓
funções mais simples
↓
heurísticas mais grosseiras
↓
boa robustez dentro do treino
↓
limitação em geometrias novas
```

### Redes grandes

```text
mais capacidade
↓
funções mais específicas
↓
exploração mais refinada dos padrões sensoriais
↓
maior generalização em alguns casos
↓
maior possibilidade de fragilidade fora da distribuição local
```

Isso combina com:

- alta validação de `6→16→16→2` e `6→32→32→2`;
- população menos robusta nas arquiteturas maiores;
- comportamento multi-volta inesperado;
- dependência visual de padrões sensoriais locais.

---

# 29. A `6 → 16 → 16 → 2` como possível sweet spot

A arquitetura campeã pode estar em uma região interessante:

```text
capacidade suficiente
+
complexidade moderada
+
generalização forte
+
robustez comportamental
```

Ela é muito maior do que o baseline:

```text
74 → 418 parâmetros
```

mas muito menor que:

```text
1346 parâmetros
```

da maior arquitetura.

Ainda não é possível afirmar que 418 parâmetros sejam uma região ótima geral.

Porém, nesta execução, ela foi a configuração que melhor equilibrou os critérios observados.

---

# 30. Presunções que se mostraram falsas ou incompletas

## 30.1 “Mais parâmetros devem produzir desempenho melhor”

**Falso.**

A relação não foi monotônica.

A maior arquitetura não dominou consistentemente.

---

## 30.2 “Mesmo fitness de treino significa comportamento parecido”

**Completamente falso.**

Arquiteturas com `best fitness ≈ 1,73` tiveram:

```text
35,5% até 97,5%
```

de conclusão em validação tardia.

---

## 30.3 “Duas arquiteturas com o mesmo número de parâmetros devem ser parecidas”

**Falso.**

```text
6→16→2
vs
6→8→8→2
```

possuem 146 parâmetros e apresentaram resultados drasticamente diferentes.

---

## 30.4 “Se a validação melhora uma vez, continuará melhorando”

**Falso.**

Diversas curvas de generalização foram não monotônicas.

---

## 30.5 “Uma política que completa uma volta sabe dirigir continuamente”

**Falso ou, no mínimo, não garantido.**

O comportamento na segunda volta mostrou que competência one-lap não implica robustez multi-lap.

---

## 30.6 “O gargalo principal deve ser a capacidade da rede”

**Cada vez menos provável.**

O conjunto de evidências qualitativas aponta fortemente para **observabilidade limitada**.

---

# 31. Ótimo local e convergência

O fitness de treino ficou muito próximo entre arquiteturas depois de bastante evolução.

Isso é um plateau real de desempenho medido.

Porém, ele não deve ser automaticamente chamado de ótimo local.

Nesta fase há três possibilidades principais:

## 31.1 Convergência no objetivo de treino

A função de fitness pode simplesmente estar atribuindo valores muito semelhantes a políticas comportamentalmente muito diferentes.

Isso já é demonstrado pela validação.

## 31.2 Ótimos locais evolutivos

Cada arquitetura pode ter encontrado regiões diferentes do landscape.

É possível que mutações pequenas ao redor de determinadas políticas sejam predominantemente destrutivas.

## 31.3 Limite representacional/sensorial

A rede pode estar próxima do máximo que consegue extrair dos seis inputs atuais.

As evidências visuais tornam essa hipótese particularmente relevante.

> [!warning]
> Não há evidência suficiente para afirmar que qualquer arquitetura tenha encontrado o ótimo global ou um ótimo local definitivo.
>
> O mais seguro é falar em:
>
> **plateau no objetivo de treino com forte divergência comportamental fora desse objetivo.**

---

# 32. Sensibilidade por input como próximo instrumento

Uma análise futura extremamente útil seria medir a sensibilidade das saídas aos inputs.

Para:

$$
o=[x_1,\dots,x_6]
$$

perturbar individualmente:

$$
x_i' = x_i+\epsilon
$$

e medir:

$$
\frac{\Delta steering}{\Delta x_i}
$$

e:

$$
\frac{\Delta acceleration}{\Delta x_i}
$$

Isso permitiria transformar a percepção visual:

> “essa rede parece dirigir usando principalmente esse raycast”

em uma medida quantitativa.

A análise pode ser separada por:

```text
reta
entrada de curva
meio de curva
saída de curva
instantes antes de colisão
```

e comparada entre arquiteturas.

---

# 33. Próxima linha experimental: observabilidade

A hipótese mais forte que emergiu desta fase é:

> **melhorar a informação disponível provavelmente terá impacto maior do que simplesmente continuar aumentando a MLP.**

Isso deve ser testado como ablation controlada.

---

# 34. Mais raycasts

Baseline atual:

```text
5 raycasts
```

Experimentos sugeridos:

```text
3
5
7
9
11
```

ou uma seleção menor e mais dirigida.

Exemplo de 9 raycasts:

```text
+90
+67.5
+45
+22.5
0
-22.5
-45
-67.5
-90
```

O objetivo não é apenas “ver mais longe”.

É permitir que a política receba uma aproximação melhor da **forma local da pista**.

---

# 35. Novas informações não privilegiadas

Além de raycasts, podem ser fornecidas informações que um veículo real poderia medir localmente.

Exemplos:

```text
yaw rate
aceleração longitudinal
aceleração lateral
velocidade longitudinal no referencial do carro
velocidade lateral no referencial do carro
ação de steering anterior
ação de acceleration anterior
```

Essas informações não precisam revelar:

- posição absoluta;
- índice da curva;
- centerline;
- trajetória ideal;
- waypoint;
- direção futura da pista.

Assim, a política continua sem informações privilegiadas do mapa.

---

# 36. Normalização de inputs

Outra dimensão futura:

```text
inputs normalizados
vs
inputs brutos
```

e diferentes formas de normalização.

Exemplos:

### Distância linear

$$
x=\frac{d}{d_{\max}}
$$

### Distância não linear

$$
x=\frac{d}{d+k}
$$

ou inversa, de acordo com a semântica desejada.

Como os pesos são evoluídos diretamente, a escala dos inputs altera o landscape de otimização e pode influenciar fortemente convergência.

---

# 37. Funções de ativação

Manter a saída em `tanh` é natural devido a:

```text
steering ∈ [-1,1]
acceleration ∈ [-1,1]
```

Mas hidden layers podem ser comparadas:

```text
Tanh
ReLU
Leaky ReLU
```

Possíveis experimentos:

```text
Tanh → Tanh
ReLU → Tanh
LeakyReLU → Tanh
```

Sempre mantendo o restante da configuração fixo.

---

# 38. Memória sem informação privilegiada

Outro experimento particularmente interessante é introduzir memória.

Manter:

```text
mesmos sensores
mesmas informações locais
```

mas substituir a política puramente feed-forward por uma pequena rede recorrente.

Se isso melhorar estados ambíguos, haveria evidência de que parte do problema atual é:

**partial observability temporal**.

Uma RNN poderia distinguir:

```text
“estou vendo esta geometria depois de uma sequência de aproximação à esquerda”
```

de:

```text
“estou vendo uma geometria parecida depois de uma sequência de aproximação à direita”
```

mesmo que o vetor sensorial instantâneo seja parecido.

---

# 39. Experimentos multi-lap

A descoberta da segunda volta justifica uma nova dimensão experimental:

```text
treino de 1 volta
vs
treino de 2 voltas
vs
treino contínuo
```

Hipótese:

> treinamento multi-lap deve pressionar a política a cruzar a linha de chegada em estados que permaneçam dirigíveis.

Se as colisões de segunda volta desaparecerem após treino de 2 voltas, isso seria forte evidência do **terminal-state blind spot**.

---

# 40. Benchmark futuro recomendado

Para artigo, cada arquitetura/configuração deveria ser avaliada num benchmark fixo independente do treino.

Por exemplo:

```text
checkpoint selecionado
↓
todas as 5 pistas de treino
+
todas as 3 pistas held-out
+
N voltas contínuas
+
hot-starts
```

Registrar:

- completion;
- lap time;
- useful speed;
- collision point;
- multi-lap survival;
- trajectory;
- sensibilidade dos inputs;
- state ao cruzar linha;
- robustez a velocidade inicial.

---

# 41. Limitações científicas desta fase

## 41.1 Uma seed por arquitetura

As arquiteturas desta bateria utilizaram seeds diferentes.

Portanto este sweep é excelente como:

**estudo exploratório**

mas não deve ser usado sozinho para afirmar superioridade estatística universal.

---

## 41.2 Falta de replicações

Ainda não existe distribuição de resultados por arquitetura.

Para publicação formal será necessário executar múltiplas seeds.

---

## 41.3 Validação sorteada por geração

Cada geração avalia o campeão em apenas uma pista held-out.

Isso produz ruído de pista na série temporal.

---

## 41.4 Race quantitativo ainda limitado

Os CSVs preservados nesta fase incluem Race exportado para:

```text
Monza
Monaco
```

As observações multi-lap nas demais pistas são importantes, mas ainda são qualitativas.

---

## 41.5 Segunda volta ainda não instrumentada formalmente

A diferença entre arquiteturas grandes e pequenas em segunda volta foi observada visualmente.

Ainda faltam métricas como:

```text
lap number
lap crossing speed
heading
offset lateral
collision lap
collision progress
```

---

# 42. Arquiteturas prioritárias para replicação

Para uma bateria estatística posterior, as configurações mais informativas são:

## `6 → 8 → 2`

Baseline mínimo e surpreendentemente robusto em treino.

## `6 → 16 → 2`

Rasa, 146 parâmetros.

## `6 → 8 → 8 → 2`

Mesmos 146 parâmetros, comportamento extremamente diferente.

## `6 → 16 → 16 → 2`

Campeã qualitativa desta fase.

## `6 → 32 → 32 → 2`

Maior arquitetura e excelente validação, porém maior custo e fragilidades observadas.

---

# 43. Protocolo futuro de seeds

Após selecionar as principais configurações:

```text
10–20 seeds por configuração
```

ou quantidade compatível com custo computacional.

Métricas:

- mediana de performance;
- média;
- desvio padrão;
- intervalo interquartil;
- taxa de conclusão;
- geração até primeira conclusão;
- área sob curva de validação;
- robustez multi-lap;
- tempo computacional;
- diversidade genética.

---

# 44. Um novo resultado metodológico

A fase mostrou que avaliar apenas:

```text
best training fitness
```

é insuficiente.

Uma política pode ter:

```text
fitness ≈ outra política
```

e ainda ser completamente diferente em:

- robustez;
- pista held-out;
- segunda volta;
- trajetória;
- distribuição de falhas.

Portanto a avaliação futura deve ser multidimensional:

$$
\text{competência}
=
f(
\text{treino},
\text{generalização},
\text{robustez},
\text{velocidade},
\text{multi-lap},
\text{custo}
)
$$

---

# 45. História experimental que emerge

Esta fase cria uma narrativa muito mais rica para o artigo.

```text
MLP pequena aprende a dirigir
↓
aumentamos capacidade
↓
fitness de treino converge para valores quase iguais
↓
validação revela diferenças enormes
↓
Race revela diferenças comportamentais que o fitness esconde
↓
multi-lap revela instabilidade pós-terminal
↓
observação visual revela dependência forte de poucos sensores
↓
nova hipótese:
observabilidade é um gargalo central
```

Isso transforma o trabalho de:

> “evoluir pesos de uma MLP para dirigir”

em uma pergunta científica mais interessante:

> **Como capacidade da política e observabilidade sensorial afetam competência, generalização e robustez de agentes de direção neuroevoluídos?**

---

# 46. Conclusões desta fase

Com os dados atuais, é razoável concluir:

1. todas as arquiteturas testadas são capazes de encontrar políticas de alto fitness em treino;
2. fitness de treino não é suficiente para comparar competência real;
3. generalização varia drasticamente entre arquiteturas;
4. mais parâmetros não produzem melhora monotônica;
5. profundidade e largura importam independentemente do número total de parâmetros;
6. `6→16→16→2` apresentou a melhor combinação qualitativa observada;
7. `6→32→32→2` atingiu excelente generalização, mas com muito mais parâmetros;
8. `6→8→2` mostrou robustez surpreendente no domínio de treino;
9. `6→8→8→2` mostrou especialização extrema em Suzuka;
10. populações de redes grandes podem ser muito piores que seus melhores campeões;
11. generalização pode melhorar e piorar ao longo da evolução;
12. competência one-lap não garante robustez multi-lap;
13. a segunda volta introduz estados fora da distribuição canônica de largada;
14. a tarefa one-lap possui um potencial terminal-state blind spot;
15. os cinco raycasts + velocidade parecem impor um gargalo real de observabilidade;
16. aumentar a capacidade da rede não resolve informação ausente;
17. novos sensores locais não privilegiados são uma direção prioritária;
18. Race é essencial porque expõe diferenças que métricas agregadas escondem.

---

# 47. O que ainda não pode ser afirmado

Ainda não há base suficiente para afirmar:

- que `6→16→16→2` é universalmente a melhor arquitetura;
- que 418 parâmetros são um ótimo;
- que redes maiores necessariamente overfitam;
- que redes pequenas são necessariamente mais robustas;
- que a instabilidade de segunda volta é causada apenas pela velocidade;
- que mais sensores necessariamente resolverão todas as falhas;
- que existe um ótimo local definitivo;
- que as diferenças observadas são estatisticamente significativas;
- que determinada função de ativação será superior;
- que memória será necessária depois de melhorar observabilidade.

---

# 48. Próximos experimentos — ordem sugerida

> [!todo]
> **1. Instrumentar Race multi-lap**
>
> Registrar por volta:
>
> - tempo;
> - velocidade na linha;
> - heading;
> - offset lateral;
> - motivo da falha;
> - progress da colisão.

> [!todo]
> **2. Canonical-reset test**
>
> Separar falha de continuação de falha de política.

> [!todo]
> **3. Hot-start sweep**
>
> Avaliar robustez à velocidade inicial.

> [!todo]
> **4. Replicar principais arquiteturas com múltiplas seeds**
>
> Antes de conclusões estatísticas.

> [!todo]
> **5. Sensor ablation**
>
> 5 raycasts vs 7/9/11 raycasts.

> [!todo]
> **6. Instrumentação física local**
>
> Testar yaw rate, acelerações e velocidades no frame do carro.

> [!todo]
> **7. Sensibilidade por input**
>
> Quantificar dependência de cada raycast.

> [!todo]
> **8. Normalização**
>
> Raw vs normalizado, linear vs assintótico.

> [!todo]
> **9. Ativações**
>
> Tanh vs ReLU/LeakyReLU em hidden layers.

> [!todo]
> **10. Memória**
>
> Apenas depois de caracterizar melhor a observabilidade feed-forward.

> [!todo]
> **11. One-lap vs multi-lap training**
>
> Testar diretamente o terminal-state blind spot.

---

# 49. Dados-fonte desta fase

Os resultados utilizados neste relatório vieram de:

```text
results/
├── 6_8_2/
├── 6_16_2/
├── 6_8_8_2/
├── 6_32_2/
├── 6_16_16_2/
├── 6_16_32_2/
├── 6_32_16_2/
├── 6_32_32_2/
└── races/
```

Por arquitetura:

```text
metrics.csv
run.ron
worker.log
```

Resultados de Race preservados:

```text
races/1788202556_monza.csv
races/1788211252_monaco.csv
```

---

# 50. Nota para o artigo

> [!quote]
> O resultado mais relevante desta fase não foi descobrir uma arquitetura “vencedora”.
>
> Foi descobrir que **políticas quase indistinguíveis pelo fitness podem possuir competências, estratégias e falhas radicalmente diferentes**.
>
> O modo Race tornou visível o que o fitness escondia.
>
> E, ao tornar essas diferenças visíveis, também deixou claro que a próxima fronteira do projeto provavelmente não é simplesmente adicionar mais neurônios, mas **dar à política informação local suficiente para compreender melhor o estado dinâmico em que ela se encontra**.

---

# 51. Resumo em uma frase

> [!summary]
> **A arquitetura `6→16→16→2` foi a melhor política observada nesta bateria, mas o principal resultado científico foi demonstrar que capacidade, generalização e robustez se dissociam fortemente quando o agente tenta dirigir usando apenas cinco raycasts e velocidade.**
