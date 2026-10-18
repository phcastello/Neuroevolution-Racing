# Executar PPO

Na raiz do workspace, Rust/Cargo estável compatível com o lockfile. Tudo em Rust; não precisa de Python para coletar, avaliar, fazer benchmarks ou treinar. As cinco funções de backpropagation estão implementadas explicitamente, sem autodiff. Presets RON em `configs/rl`: smoke `[8,8]`, base `[64,64]`, stress `[256,256,256,256,128]`. D vem do ObservationSpec; o preset Extended tem 25 entradas. São pontos iniciais de engenharia, sem alegação de hiperparâmetros ótimos.

## CPU e comandos

```powershell
cargo build -p neuroevolution-racing-app --locked
cargo run -p neuroevolution-racing-app -- rl --help
cargo run -p neuroevolution-racing-app -- rl collect --config configs/rl/smoke.ron
```

`collect` cria e imprime um diretório exclusivo em `results_ppo`. Produz rollout válido, checkpoint aleatório não treinado, loss forward calculada sobre as ações históricas, manifesto, métricas e log. Não atualiza pesos. Use o caminho impresso nos comandos seguintes:

```powershell
cargo run -p neuroevolution-racing-app -- rl inspect-rollout results_ppo/ppo-<execucao>/rollout-000000.ron
cargo run -p neuroevolution-racing-app -- rl evaluate --checkpoint results_ppo/ppo-<execucao>/checkpoint-000000.ron --episodes 4 --cpu
cargo run -p neuroevolution-racing-app -- rl view --checkpoint results_ppo/ppo-<execucao>/checkpoint-000000.ron --cars 4
cargo run -p neuroevolution-racing-app -- rl benchmark --config configs/rl/smoke.ron
cargo run -p neuroevolution-racing-app -- rl check-gradients --case list
cargo run -p neuroevolution-racing-app -- rl check-gradients --case all
cargo run -p neuroevolution-racing-app -- rl train --config configs/rl/smoke.ron
```

`inspect-rollout` valida shapes, finitude, ordenação env/time, ids de episódio, continuidade, flags/motivos, ação=Tanh(u) e GAE; mostra dimensões, contadores, recompensa e términos. `evaluate` executa o número pedido **por pista** da suite validation, com sementes próprias, e grava resultados por episódio. Políticas aleatórias são marcadas UNTRAINED. `view` é um visualizador PPO separado com Bevy, sem alterar os modos originais; CPU f32, política determinística, pistas de avaliação, até 16 carros. Treino/coleta não abrem janela.

No visualizador, `--frames 120` limita a prévia a 120 frames renderizados e fecha a janela automaticamente. A decisão RL usa período `action_repeat * physical_dt`; cada integração continua em ticks de 1/60 s.

`check-gradients --case all` compara o backward com diferenças centrais CPU f64 e agora deve imprimir `APROVADO`. `train` executa um preflight dos cinco hooks antes de criar cenas, inicializar GPU ou iniciar coleta. O treino usa exclusivamente derivadas analíticas e operações tensoriais. [Guia das cinco funções](BACKPROP_PARA_IMPLEMENTAR.md).

Treinamento e retomada na CPU:

```powershell
cargo run --release -p neuroevolution-racing-app -- rl train --config configs/rl/base.ron
cargo run --release -p neuroevolution-racing-app -- rl train --config configs/rl/base.ron --resume <checkpoint-ppo.ron>
```

Resume aceita extensão de total_rollouts e mudança de backend, exige os outros parâmetros compatíveis, preserva arquivo fonte e cria nova execução. Restaura Adam, parâmetros e RNGs, mas reinicia os ambientes. Não é continuidade exata das trajetórias.

## GPU Radeon/WGPU opcional

```powershell
cargo check -p neuroevolution-racing-app --features rl-gpu --locked
cargo build --release -p neuroevolution-racing-app --features rl-gpu --locked
```

Copie um preset para um novo RON e altere `device: Cpu` para `device: Gpu(adapter: 0)`. Rode collect/benchmark com esse arquivo e `--features rl-gpu`. O índice usa a lista de adaptadores **hardware** Vulkan/DX12 enumerados na máquina, registrada na saída e no manifesto. Índice inexistente ou ausência de hardware gera erro; não seleciona CPU/software silenciosamente. Precisão f32, sem CUDA, sem mixed precision. `evaluate --cpu` permite avaliar um checkpoint GPU na CPU; `view` sempre usa CPU.

Para iniciar uma execução mais longa na RX 6700 XT, a partir do preset base (não foi iniciada automaticamente):

```powershell
$base = [IO.File]::ReadAllText((Join-Path $PWD 'configs/rl/base.ron'))
[IO.File]::WriteAllText((Join-Path $PWD 'target/rl-base-gpu.ron'), $base.Replace('device: Cpu,', 'device: Gpu(adapter: 0),'))
cargo run --release -p neuroevolution-racing-app --features rl-gpu --locked -- rl train --config target/rl-base-gpu.ron
cargo run --release -p neuroevolution-racing-app --features rl-gpu --locked -- rl evaluate --checkpoint <diretorio-impresso>/checkpoint-000100.ron --episodes 4
```

O checkpoint final corresponde a `total_rollouts` do arquivo efetivo. A avaliação determinística usa pistas reservadas. `trained=true` indica que houve update, sem afirmar convergência. Para um smoke curto, copie `smoke.ron` e reduza **na cópia** `total_rollouts: 100` para `total_rollouts: 3`; mantenha os presets originais.

Para comparação, execute o mesmo preset com Cpu e Gpu, preservando seeds/rede/grade. O benchmark valida resultados do backend selecionado contra forward CPU com entradas/pesos iguais. Warm-up e sincronização estão incluídos; medições de inferência excluem upload/download e os reportam separadamente. Rollout mede física real, sensores e preparação GAE. Rode release para números de desempenho úteis. Não conclua vantagem da GPU a partir de ocupação ou de uma rede pequena.

Base e stress usam grade N=1,8,32,128,256; smoke só 1,8. Edite `benchmark_envs`, `benchmark_iterations` e `benchmark_warmup` para expandir a grade. Não há execução automática de stress. Memória estimada e limites são validados antes; driver/Bevy/kernel cache podem usar memória adicional. O benchmark existente mede inferência/coleta, sem executar updates. Para tempos reais de update, use `update_seconds` em `run.log` de um treino; `losses-*.ron` guarda losses, normas e passos do otimizador. A mensagem histórica do benchmark sobre backward indisponível não descreve mais o estado dos hooks.

## Verificações

```powershell
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
cargo clippy -p racing-rl --all-targets --no-deps --locked -- -D warnings
cargo clippy -p racing-rl --all-targets --features gpu --no-deps --locked -- -D warnings
cargo test -p racing-rl --features gpu --test gpu --locked -- --nocapture
cargo check -p neuroevolution-racing-app --features rl-gpu --locked
cargo tree -p racing-rl -e features
cargo tree -p racing-rl --features gpu -e features
```

Não devem aparecer features/dependências resolvidas de burn-autodiff, burn-train, CUDA, rede/otimizador pronto. Um lockfile pode listar pacotes opcionais não ativos; confira `cargo tree` e não apenas strings do lockfile. `neuroevolution` continua sem Burn.

Os testes cobrem infraestrutura e backward. Rode com `--nocapture` para ver as aprovações por caso; não há `#[ignore]` escondendo os checks. Consulte [ARQUITETURA](ARQUITETURA.md) para semântica de tempo, observações, reward/GAE e caches.

Os comandos GA existentes permanecem `batch <arquivo>` e `worker ...`; batch continua sendo launcher de experimentos, distinto dos minibatches de PPO. Executar sem subcomando mantém a interface original.

## Estado da validação desta entrega

Veja `VALIDACAO.md` para comandos e resultados efetivamente executados, hardware disponível, problemas prévios do checkout, testes CPU/GPU e treinos de três rollouts. Esses smokes comprovam integração e atualizações, sem alegação de convergência ou desempenho satisfatório na pista.
