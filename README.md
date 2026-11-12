# Neuroevolution Racing

Projeto acadêmico em Rust para estudar controladores neurais de veículos e sua generalização em pistas desconhecidas. Bevy e Avian2D fornecem a simulação compartilhada; egui fornece a aplicação visual.

- **GA:** evolui os pesos de MLPs; preserva treinamento visual, indivíduos, gerações, corridas, checkpoints e workers headless.
- **PPO:** Actor-Critic com PPO-Clip, GAE, backward analítico e Adam implementados manualmente sobre tensores Burn. Treinamento headless com CPU NdArray por padrão; WGPU/Vulkan é opcional.
- **ACO:** Max-Min Ant System constrói pistas de treinamento por decisões discretas de geometria. Publica decisões, geometria, probabilidades, feromônios, fitness e rejeições durante a execução visual e preserva os arquivos para replay.
- **Currículo adaptativo:** mistura pistas originais, procedurais, selecionadas e históricas. A busca avalia uma política congelada e troca o conjunto somente entre rollouts.

A hipótese científica é que sensores concentrados à frente, antecipação da linha central e seleção adaptativa de pistas possam melhorar a conclusão em pistas desconhecidas. Nenhum resultado dessa hipótese é presumido.

## Compilar e executar

Rust stable, Linux x86-64 e compilador C. Para o Servidor 47, siga o [guia CPU e protocolo de hardware](docs/SERVIDOR_47.md).

```bash
# CPU sem janela, egui, WGPU, Vulkan ou CUDA
cargo build --release --locked -p neuroevolution-racing-app --no-default-features
# Aplicação visual: menu GA / PPO / ACO / Experimentos
cargo run --locked -p neuroevolution-racing-app
# Ajuda e teste pequeno: compilar CPU primeiro
./tools/server47.sh build-cpu
./tools/server47.sh smoke
```

A aplicação visual compartilha janela, renderizador, câmera e dashboard; os modos ativam seus próprios sistemas. PPO carrega checkpoints congelados e até 16 veículos. Experimentos consulta configurações, métricas, gráficos e checkpoints existentes.

Ao selecionar **PPO**, a tela já lista os checkpoints de `results_ppo`, incluindo subpastas e treinos antigos. Clique no checkpoint para carregar. O catálogo agrupa por treino, mostra o último checkpoint e permite expandir os anteriores ou filtrar por nome. A descoberta e a leitura dos metadados ocorrem em segundo plano; os pesos completos são validados somente ao carregar a política escolhida. No **ACO**, marque **Avaliar com PPO congelado** e selecione na mesma lista, sem copiar caminhos.

No modo **ACO**, configure formigas, iterações, seed e parâmetros, escolha execução sem piloto ou com checkpoint PPO congelado e clique em **Iniciar**. Pausa, continuação, passo por decisão/iteração, velocidade, interrupção e reinício controlam o motor real em uma thread separada. A construção mostra pontos e segmentos abertos; candidatas completas exibem o circuito e a validação geométrica. As abas mostram probabilidades, feromônios, evaporação/reforço, candidatas, fitness, diversidade e histórico. **Salvar / exportar / reproduzir arquivo** preserva a exportação e o replay das buscas anteriores. Uma candidata também pode ser aberta no piloto PPO. Ao trocar de modo, a busca pausa; use **Continuar** ao retornar.

O padrão agora é **Sequencial v2**: retas, arcos, hairpins, chicanes e curvas em S alteram posição e orientação. Cada expansão mantém um fechamento viável por primitivas, com tangente, largura, curvatura e orçamento verificados. **Radial legado** continua disponível. Veja a [reformulação e as pistas produzidas](docs/ACO_SEQUENCIAL.md), incluindo a calibração por La Source, métricas e capturas reais. O preset headless pequeno é `configs/aco/sequential_smoke.ron`.

A construção atual inclui **exploração livre, exploração orientada e fechamento**, bends e composições assimétricas. A interface explica quando a formiga prefere passear mais. Veja a [comparação de organicidade com seeds fixas](docs/ACO_ORGANICO.md), com todas as candidatas, métricas e voltas de referência.

O padrão físico usa larguras **96/100/104** e raio base **75**, comparáveis a Interlagos. A aba **Candidatas** permite filtrar, comparar, salvar e testar válidas rejeitadas; geometrias inválidas têm diagnóstico separado. Veja [largura física, retenção e inspeção de rejeições](docs/ACO_LARGURA_INSPECAO.md).

O ACO usa a **visualização principal**, com **Ver melhor pista** e **Ver geração** na barra superior. **Ver pista**, na lista de candidatas, mostra a escolha nessa mesma cena. Não há janela flutuante de construção no modo ACO. [Uso e capturas](docs/ACO_VISUALIZACAO_PRINCIPAL.md).

**Testar com PPO** abre a candidata selecionada com um checkpoint congelado, sem herdar a pausa do teste anterior. O seletor mantém a pista vinculada enquanto você escolhe a política. [Fluxo e validação do teste ACO → PPO](docs/ACO_TESTE_PPO.md).

## Pistas e pesquisas

```bash
target/release/neuroevolution-racing aco generate --count 3 --seed 47 --output results_ppo/generated-47
# Busca independente sem piloto: somente geometria e diversidade
target/release/neuroevolution-racing aco search --output results_ppo/search-47
# Busca com piloto congelado CPU e referência geométrica
# target/release/neuroevolution-racing aco search --checkpoint CAMINHO_EXATO.ron --output NOVA_BUSCA

# Preparação inerte da matriz: não inicia os 16 treinamentos
target/release/neuroevolution-racing research matrix --output results_ppo/matrix --dry-run
target/release/neuroevolution-racing research matrix --output results_ppo/matrix
# Triagens independentes: A=observações (100), B=currículo (200)
target/release/neuroevolution-racing research matrix --screening --observations --output results_ppo/observations
target/release/neuroevolution-racing research matrix --screening --curricula --output results_ppo/curricula
# Execução futura explícita de um plano já preparado:
# target/release/neuroevolution-racing research execute --plan results_ppo/matrix/plan.ron
```

A [especificação científica](docs/PESQUISA.md) explica observações 25/31, geometria, ACO, fitness, currículo, matriz 2×2×4, seleção progressiva e limitações. `experiments/factorial.ron` é a configuração central; presets históricos em `configs/rl/` continuam intactos.

## Treinar, avaliar e retomar

```bash
# Use uma configuração efetiva preparada, CPU, com orçamento revisado
BIN=target/release/neuroevolution-racing
$BIN rl train --config CONFIG.ron --save-every 100 --save-first 0 --save-last 0 --no-save-rollouts
$BIN rl train --config CONFIG_CONTINUACAO.ron --resume CHECKPOINT_EXATO.ron --save-every 100 --save-first 0 --save-last 0 --no-save-rollouts
$BIN rl evaluate --checkpoint CHECKPOINT_EXATO.ron --cpu --tracks validation
# Requer build visual
$BIN rl view --checkpoint CHECKPOINT_EXATO.ron --cars 4
```

Resultados PPO ficam no `results_root` efetivo, em diretórios exclusivos `ppo-<tempo>-<pid>/`: manifest, log, métricas por rollout/pista/update, hardware, desempenho e checkpoints amostrados. `--run-path-file` registra o diretório exato. Checkpoints preservam observações, Adam, RNGs e currículo. A retomada reinicia os episódios físicos; não reproduz exatamente a trajetória interrompida.

Os comandos GA `batch experiments/smoke_test.ron` e `worker --architecture 6,8,2 --target-generation 1 --seed 47 --population-size 4 --results-root results --worker-threads 1` continuam disponíveis. Artefatos GA históricos e campeões permanecem compatíveis.

## Validar

```bash
cargo fmt --all --check
cargo check --workspace --locked --no-default-features
cargo test --workspace --locked --no-default-features
cargo check -p neuroevolution-racing-app --locked # visual
cargo check -p neuroevolution-racing-app --locked --no-default-features --features cpu-parallel
```

O [registro de validação](docs/rl/VALIDACAO.md#validação-final-do-release-cpu-e-publicação--2026-11-11) distingue as rodadas executadas. O release CPU e o smoke com backward/Adam foram aprovados; Clippy global continua falhando exclusivamente em diagnósticos legados comprovados, sem novos diagnósticos. Operação gráfica real e GPU permanecem pendentes.

`rl benchmark` continua medindo inferência/coleta. O **benchmark completo de treinamento**, com backward e Adam, usa `research benchmark`; preparar não executa. Nenhum build ou instalação dispara experimentos automaticamente.
