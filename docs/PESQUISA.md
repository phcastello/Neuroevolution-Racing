# Observações, pistas, ACO e experimentos

## Observações e checkpoints

Extended histórico: 21 raycasts uniformes de +90 a−90° e quatro entradas (velocidade longitudinal, yaw medido, steering e aceleração anteriores), total25. `SensorLayout::FrontDense` usa `[90,80,60,45,33,25,18,12,7,3,0,-3,-7,-12,-18,-25,-33,-45,-60,-80,-90]`, com mesmo alcance1000, normalização e física. Não há sensores traseiros. A distribuição frontal é hipótese experimental.

Antecipação opcional acrescenta três pares (lateral à esquerda, longitudinal à frente), total31 entradas. Projeta na linha central conectada e amostra distâncias de arco150/350/700 com wrap circular; transforma desde a posição real do veículo e normaliza por700, com clamp finito. Esses valores são configuráveis em `AnticipationSpec`. A escala250 é normalização de velocidade, não vmax: aceleração125 com falloff140 não impõe teto rígido. A250, as distâncias correspondem a0,6/1,4/2,8s; a400,0,375/0,875/1,75s. Com aceleração lateral225, raio de referência é≈278 a250 e≈711 a400. As distâncias são hipóteses para ablação, não valores ótimos.

Novas especificações usam versão2 e guardam ângulos, distâncias e normalização. Checkpoints v1 ausentes do campo de antecipação continuam25, com Feature históricos preservados. Visualização lê a especificação salva. Quatro camadas256 têm204.805 parâmetros com25 entradas e206.341 com31; o código calcula ambos.

## Representação e gerador

O formato permanece `TrackDefinition`: pontos de controle, largura e Catmull-Rom fechado amostrado. Geração escolhe níveis radiais em setores ordenados e uma largura discreta. `shape: Radial` é o default; `shape: RoundedRectangle` usa setores no perímetro retangular (quantidade múltipla de quatro), permitindo retas Catmull-Rom exatas quando controles vizinhos usam o mesmo nível. Ambos podem produzir curvas sucessivas e mudanças de sentido de curvatura. O espaço de busca é deliberadamente limitado e não representa todas as topologias de circuitos reais. Um preset de retas e curvas suaves é `(shape: RoundedRectangle, radius: 700.0, radial_levels: [1.0], widths: [80.0])`; ele passa pelos mesmos limites geométricos.

Validação explícita rejeita interseções de centro/bordas, comprimento fora dos limites, largura insuficiente, curvatura excessiva, degenerações e spawn sem folga veicular. A física usa as mesmas paredes Avian e projeção de progresso. A geração tem tentativas limitadas, seed e hash FNV-1a estável dos bits da geometria. O artefato `.generated.ron` contém definição, seed, decisões, parâmetros e métricas; `.ron` exportado contém a definição tradicional. Carregamento revalida a proveniência. Pistas geradas são sempre Training; a biblioteca recusa registrá-las como Validation.

## ACO independente

Max-Min Ant System em grafo de decisões em camadas: cada posição escolhe raio de um setor; a última escolhe largura. Feromônios pertencem a essas alternativas, não às coordenadas físicas da pista. Seleção usa τ^αη^β, exploração probabilística e escolha gulosa opcional. η favorece continuidade de níveis vizinhos. Há evaporação, reforço elitista limitado, reinicialização por estagnação, deduplicação por hash e distância mínima entre decisões. Snapshots registram probabilidades efetivas, matrizes antes/evaporadas/reforçadas, candidatas, rejeições, melhores soluções e diversidade.

Sem piloto, score=0,4×dificuldade geométrica+0,6×diversidade. Com PPO congelado, score=0,4×(1−conclusão)+0,2×(1−progresso)+0,1×colisões+0,1×geometria+0,2×diversidade. Componentes ficam separados. A referência determinística pure pursuit usa a mesma física; candidatas sem conclusão da referência são marcadas como dirigibilidade não verificada e não reforçam feromônios. Isso é conservador: falha da referência não prova impossibilidade física.

A avaliação usa um episódio determinístico PPO e um episódio referência por candidata, dividindo o orçamento de decisões. `action_repeat` e observações do checkpoint são preservados; timeout é limitado ao orçamento. Não há rewards futuros/ações ótimas nas observações. Existem limites de candidatas e de simulação; a política congelada não recebe otimizador. RNGs de construção e avaliação são separados e persistidos com offsets ChaCha representáveis em RON.

```bash
BIN=target/release/neuroevolution-racing
$BIN aco generate --count 2 --seed 47 --output results_ppo/g47
$BIN aco search --output results_ppo/a47 --dry-run
$BIN aco search --config CONFIG_ACO.ron --checkpoint POLICY.ron --output results_ppo/search-policy
$BIN aco replay results_ppo/search-policy/search.ron
$BIN aco export results_ppo/search-policy/search.ron --output results_ppo/exported
```

`AcoConfig` e `GeneratorConfig` usam defaults explícitos com overrides RON; `aco search --config` aceita, por exemplo, `(ants: 2, iterations: 1, max_candidates: 2, max_selected: 1)`. Reprodução visual consome `search.ron`; não executa outro ACO.

## Currículo e integridade on-policy

`Config.curriculum: None` preserva treinamento original. `Some((kind:Aco, ...))` habilita ativações após100 rollouts, intervalo50,8 formigas×3 iterações, máximo24 candidatas e4 selecionadas. Defaults são hipóteses. `kind:Random` toma primeiras candidatas válidas sem avaliar dificuldade; `Difficulty` usa amostragem uniforme e o mesmo orçamento/evaluador de ACO. Não há acesso às pistas de validação no módulo de busca.

A ativação acontece após coleta/GAE/update, com política emprestada imutavelmente. Depois os ambientes são reconstruídos; nenhuma transição antiga entra no rollout seguinte. `mix:[5,2,2,1]` representa multiplicidades de amostragem por pista original/fresca/selecionada/histórica; não percentuais fixos por grupo. Histórico limitado24 reduz esquecimento e RAM. Pistas frescas são geradas independentemente das candidatas de seleção. Cada ativação registra custo, busca, pistas efetivamente ativas em `tracks/` e ranking da busca em `search-ranked-tracks/`; checkpoints guardam todos os conjuntos ativos, parâmetros do gerador e última busca.

Cada ativação começa uma busca com novos feromônios; a última busca é guardada para auditoria/reprodução. Não há continuidade dos feromônios entre políticas diferentes. Resume restaura conjuntos, contagem de ativações e seeds derivadas; reinicia episódios, como PPO histórico. A trajetória completa não é exatamente reproduzível após restart. O limite de candidatos conta também construções inválidas, e os dois seletores de dificuldade usam os mesmos limites, embora avaliações efetivas possam diferir por rejeições.

## Matriz e seleção progressiva

`experiments/factorial.ron` define uma base histórica e parâmetros operacionais. Geração programática: sensores original/front ×sem/com antecipação ×original/random/difficulty/aco=16 células. Seeds emparelhadas, mesmas quatro camadas256, reward, PPO, transições por rollout e validação. CPU padrão,500 rollouts máximos; seeds/repetições/paralelismo configuráveis. Não confundir rollouts com episódios concluídos.

```bash
BIN=target/release/neuroevolution-racing
$BIN research matrix --output results_ppo/full-plan --dry-run
# Triagem: preparar ou, posteriormente, executar explicitamente com --execute
$BIN research matrix --output results_ppo/screening --screening
# Continuação manual; baseline original-rays-original é sempre incluído
$BIN research matrix --output results_ppo/finalists --include front-future-aco,original-rays-original --continue-from results_ppo/screening
```

O plano guarda configurações efetivas e recibos por célula/seed. `research execute --plan DIRETORIO/plan.ron` executa um plano já preparado. `--continue-from` exige checkpoints de triagem reais; Adam e RNGs são retomados, com cap cumulativo500. Não há ranking automático que elimine baselines. Triagem100 coincide com aquecimento100: a busca ocorre na fronteira final e prepara o currículo da continuação; isoladamente essa triagem ainda não mede o efeito de aprender nas pistas selecionadas. Se quiser distinguir currículo na triagem, mude o protocolo explicitamente e mantenha seeds/orçamento comparáveis.

A configuração central também define `checkpoint_every`, `save_first`, `save_last` e `save_rollouts`. `checkpoint_every:None` segue `evaluate_every`, ou usa100 quando avaliação é desabilitada; as janelas inicial/final têm default0 e buffers de rollout não são salvos por default. Por exemplo, `checkpoint_every:Some(25)` com `evaluate_every:100` guarda checkpoints a cada25 rollouts e o final. A preparação rejeita cadência zero, avaliações sem checkpoint correspondente e combinações de seeds/repetições que produzam destinos duplicados, antes de criar diretórios. Essas opções operacionais ficam no plano e usam a mesma CLI e validação de persistência do treinador.

A métrica principal é conclusão em pistas desconhecidas. O executor da matriz avalia os checkpoints finais em CPU nas pistas Validation, mesmo com `evaluate_every:0`; também é possível executar `rl evaluate --checkpoint ... --cpu --tracks validation` separadamente; validação não ajusta feromônios ou currículo. A avaliação periódica histórica usa suite Training para diagnósticos. Telemetria existente registra rewards, términos, progresso, policy/value loss, entropia, KL, clip fraction, norma dos gradientes e early stop. Avaliação congelada registra episódios e resumos por pista. Desempenho/hardware reutiliza o mesmo coletor/treinador. `research compare` consolida artefatos reais em CSV/JSON/Markdown; não atribui resultados a planos não executados.

Cada avaliação final da matriz permanece em um diretório próprio; o run de treinamento registra seu caminho em `evaluation-path.txt`. Diretórios irmãos usam caminho relativo ao run, preservado ao transferir a árvore completa para outro host; destinos externos usam caminho absoluto. Ao comparar os diretórios de treinamento, o consolidador segue esse vínculo e inclui as métricas de validação na mesma identidade da execução. Um recibo quebrado produz erro explícito. Preserve os diretórios de treinamento e avaliação juntos.

O protocolo futuro de hardware está no [guia Servidor47](SERVIDOR_47.md). Ele usa treinamento completo de100 rollouts, sem currículo/novas observações. Não foi executado nesta tarefa.
