# Registro de validação

Ambiente: Windows, Rust 1.97.1, Ryzen 5 5600X, Radeon RX 6700 XT. O adaptador realmente selecionado nos testes numéricos GPU foi Vulkan, driver AMD proprietário 26.8.1, WGSL e f32. Não se usou CUDA nem autodiff.

## Estado prévio e preservação

- O primeiro `git status --short` falhou porque o `.git` local não tem HEAD/config/índice normais; há referências e índices com nomes de conflito de sincronização. Esses metadados não foram reparados nem substituídos.
- A referência sobrevivente aponta para `0fb110eccc434e71b9def0c3a08b3dd11fab2351`, também HEAD remoto na inspeção. Foi criado checkout auxiliar somente sob `target/rl-upstream` para conferência.
- Existiam diferenças locais em rendering, controller, checkpoint, simulation/mod e systems, checkpoint_browser, dashboard, ui/mod, network_view e README. Foram preservadas. A comparação foi feita com uma cópia dos arquivos locais **anterior** às edições em `target/rl-baseline`.
- `assets/` estava ausente. A primeira suíte original falhou antes de executar por 11 sprites usados por `include_bytes!` inexistentes. Os assets de carros e pistas foram recuperados **do mesmo commit**, sem substituir circuitos por aproximações novas. Os arquivos locais existentes não foram restaurados a partir do remoto.
- Depois da recuperação de assets: **80 testes do neuroevolution e 117 do app passaram**, zero falhas/ignorados. Logs locais: `target/rl-baseline-tests.log`.
- No código antigo da simulação, só foram abertas visibilidades de funções/estruturas da física e registrada a adaptação RL. Núcleo neuroevolution, fitness, GA, controller legado, checkpoints antigos, UI e launcher/worker não foram reescritos. Não houve commit ou push automático.

## Comandos e evidência

Os logs de build/check/teste desta sessão estão sob `target/rl-*.log`; resultados PPO têm diretórios exclusivos em `results_ppo/`.

| Verificação | Resultado |
|---|---|
| `cargo check -p racing-rl` | Passou CPU |
| `cargo test -p racing-rl` | Passou suíte inicial de infraestrutura; ver rodada final para casos adicionados |
| `cargo test -p neuroevolution-racing-app rl_environment --locked -- --nocapture` | Passou os seis casos iniciais de adaptação; smoke de update explicitamente PENDENTE |
| `cargo check -p neuroevolution-racing-app --features rl-gpu` | Passou o caminho integrado GPU |
| `cargo test -p racing-rl --features gpu --test gpu --locked -- --nocapture` | Passou na RX 6700 XT: forward batches 1/8/32, distribuição, loss, norma global e Adam com gradientes conhecidos |
| `cargo clippy -p racing-rl --all-targets --no-deps --locked -- -D warnings` | Passou |
| `cargo clippy -p racing-rl --all-targets --features gpu --no-deps --locked -- -D warnings` | Passou também com GPU, no estado final |
| `cargo fmt --all --check` | Passou |
| `cargo check --workspace --locked` | Passou sem habilitar GPU |
| `cargo build -p neuroevolution-racing-app --features rl-gpu --locked` | Passou binário completo CPU/GPU |
| `cargo test --workspace --features neuroevolution-racing-app/rl-gpu --locked` | **219 passaram**: 80 neuroevolution, 124 app (117 antigos + 7 RL), 14 infraestrutura RL, 1 hardware GPU; zero falhas/ignorados |
| `cargo test --workspace --locked` | **218 passaram** no caminho sem feature GPU; zero falhas/ignorados |
| `cargo clippy --workspace --all-targets --features neuroevolution-racing-app/rl-gpu --locked` | Concluiu; avisos anteriores preservados, nenhum nos arquivos RL novos |
| `rl --help`, `rl check-gradients --case list` | Passaram, comandos reais disponíveis |
| `rl collect --config configs/rl/smoke.ron` | Passou CPU; 16 transições, [T=8,N=2,D=25], checkpoint e loss forward |
| `rl inspect-rollout <rollout CPU>` | Passou validações |
| `rl evaluate --checkpoint <checkpoint CPU> --episodes 1 --cpu` | Passou, três episódios reservados (Spa, Suzuka, Monaco), seed 55, UNTRAINED |
| `rl evaluate --checkpoint <checkpoint GPU> --episodes 1` | Passou também na Radeon, mesmos três resultados por episódio dentro da precisão observada; UNTRAINED |
| `rl benchmark --config configs/rl/smoke.ron` | Passou CPU, N=1 e 8, warm-up/sincronização, física real |
| `rl collect --config target/rl-smoke-gpu.ron` | Passou na Radeon/Vulkan f32: 16 transições, GAE, loss e checkpoint |
| `rl inspect-rollout <rollout GPU>` | Passou validações |
| `rl benchmark --config target/rl-smoke-gpu.ron` | Passou GPU em N=1 e 8; maior diferença de forward contra CPU `5.96e-8` |
| `rl view --checkpoint <checkpoint GPU> --cars 2 --frames 10` | Abriu janela Bevy/Vulkan, executou prévia CPU f32 e encerrou normalmente após o limite; sem inspeção visual automatizada de layout |
| `rl train --config configs/rl/smoke.ron` | Código 1 esperado: `MissingManualImplementation { function: activation_backward } (PENDENTE)` |
| `rl check-gradients --case all` | Código 1 esperado: pendência manual, sem anunciar derivada aprovada |

Os gradient checks/sintético/smoke de update **não foram aprovados como backprop**. A infraestrutura verifica e exibe as pendências; não há gradientes falsos nem demonstração de aprendizagem.

Artefatos de exemplo desta sessão (não são modelos treinados): `results_ppo/ppo-1792085148071970700-17080/` contém coleta/checkpoint GPU; `results_ppo/ppo-1792085179626952000-4560/` contém coleta/checkpoint CPU final. O arquivo GPU de teste foi gerado copiando smoke e trocando somente `device: Cpu` por `device: Gpu(adapter: 0)`. Os três presets versionados ficam em `configs/rl`.

## Compatibilidade e lint

A auditoria `cargo tree -p neuroevolution-racing-app --features rl-gpu -e features` não encontrou burn-autodiff, burn-train, burn-nn, burn-optim, cubecl-cuda ou cudarc ativos. O lockfile pode conter opcionais inativos; não confundir sua presença com features resolvidas. `cargo tree -p neuroevolution` continua sem Burn.

WGPU 26/Naga 26 e Bevy compartilham codespan-reporting com features distintas. Foi necessário habilitar `naga/termcolor` no caminho GPU para compatibilizar diagnósticos; não se modificou código de terceiros nem se suprimiram erros. Referência: [issue upstream de compilação Naga 26](https://github.com/gfx-rs/wgpu/issues/7968).

Clippy estrito do workspace encontrou avisos prévios no GA e no app antigo: conversões redundantes, `len_without_is_empty`, inicialização de campos após Default em testes, complexidade de queries ECS, argumentos de sistemas e outros. Foram mantidos fora do escopo. Um aviso novo `large_enum_variant` em Job foi corrigido com Box; não se adicionou allow geral.

## Limites das medições

Smoke é curto e foi executado em build dev, com compilação acontecendo nesta sessão. Os tempos servem para validar execução e instrumentação; não sustentam conclusão de ganho de velocidade. O preset stress não é executado automaticamente. Treinamento completo e desempenho aprendido são indisponíveis até os hooks. Retomada reinicia ambientes e não é continuidade exata de trajetória. A regra de rejeição de saltos de progresso é heurística documentada.
