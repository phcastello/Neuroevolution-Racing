# Registro de validação

**Estado atual (18/10/2026): os cinco hooks estão implementados e aprovados.** A seção [Backpropagation completo](#backpropagation-completo) registra os checks e treinos CPU/Radeon desta implementação. As seções anteriores a ela preservam o histórico da infraestrutura e da revisão de fixtures, quando os hooks ainda estavam pendentes.

Ambiente: Windows, Rust 1.97.1, Ryzen 5 5600X, Radeon RX 6700 XT. O adaptador realmente selecionado nos testes numéricos GPU foi Vulkan, driver AMD proprietário 26.8.1, WGSL e f32. Não se usou CUDA nem autodiff.

## Estado prévio e preservação

- O primeiro `git status --short` falhou porque o `.git` local não tem HEAD/config/índice normais; há referências e índices com nomes de conflito de sincronização. Esses metadados não foram reparados nem substituídos.
- A referência sobrevivente aponta para `0fb110eccc434e71b9def0c3a08b3dd11fab2351`, também HEAD remoto na inspeção. Foi criado checkout auxiliar somente sob `target/rl-upstream` para conferência.
- Existiam diferenças locais em rendering, controller, checkpoint, simulation/mod e systems, checkpoint_browser, dashboard, ui/mod, network_view e README. Foram preservadas. A comparação foi feita com uma cópia dos arquivos locais **anterior** às edições em `target/rl-baseline`.
- `assets/` estava ausente. A primeira suíte original falhou antes de executar por 11 sprites usados por `include_bytes!` inexistentes. Os assets de carros e pistas foram recuperados **do mesmo commit**, sem substituir circuitos por aproximações novas. Os arquivos locais existentes não foram restaurados a partir do remoto.
- Depois da recuperação de assets: **80 testes do neuroevolution e 117 do app passaram**, zero falhas/ignorados. Logs locais: `target/rl-baseline-tests.log`.
- No código antigo da simulação, só foram abertas visibilidades de funções/estruturas da física e registrada a adaptação RL. Núcleo neuroevolution, fitness, GA, controller legado, checkpoints antigos, UI e launcher/worker não foram reescritos. Não houve commit ou push automático.

## Implementação inicial: comandos e evidência

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

## Revisão da suíte de gradient checking

Esta correção alterou somente `crates/rl/src/verification.rs` e documentação. As cinco funções de `manual.rs` permaneceram intactas, assim como coleta, backends/GPU, física e persistência. Não houve commit nem push nesta revisão.

Os testes numéricos PPO agora usam a grade completa de vantagens positivas/negativas/zero com ratios 0,55/1,05/1,55, B=9/18 e coeficientes de entropia 0/0,017. Conferem os ratios e o ramo do objetivo antes do hook e conservam as regiões durante todos os stencils. `old_log_prob` é construído uma única vez por fixture, sem recalibração nas perturbações. Há casos separados `mu`, `value`, `log_std` e o caso integrado `ppo`; o teste denso usa shapes não quadradas `(B,I,O)=(2,3,4)` e `(5,4,2)`.

Verificações **realmente executadas nesta revisão**, em CPU:

| Comando/verificação | Resultado |
|---|---|
| `cargo test -p racing-rl --locked` antes da edição | 14 testes de infraestrutura passaram; hooks pendentes |
| `cargo test -p racing-rl --locked -- --nocapture` depois da correção | 3 testes novos da infraestrutura numérica + 14 existentes passaram; pendências manuais exibidas por caso |
| `cargo test --workspace --locked` | **221 passaram**: 80 neuroevolution, 124 app, 3 novos de verification, 14 de infraestrutura RL; zero falhas/ignorados |
| `cargo clippy -p racing-rl --all-targets --no-deps --locked -- -D warnings` | Passou |
| `cargo fmt --all --check` | Passou |
| `cargo build -p neuroevolution-racing-app --locked` | Passou |
| `rl check-gradients --case list` no binário reconstruído | Passou e listou `mu` e `value` |
| `rl check-gradients --case dense` | Código 1, `dense_backward` PENDENTE |
| `rl check-gradients --case mu`, `value`, `log_std`, `ppo` | Cada comando retornou código 1, `ppo_output_gradients` PENDENTE, sem APROVADO |
| `rl check-gradients --case all` | Código 1, `activation_backward` PENDENTE |
| SHA-256 de `manual.rs` antes/depois | Idêntico: `63a35ed4ccab4a64b0924243fbb93f20a5ef2c62b7994f22194f38d88f73f49e` |

Os três testes novos executaram as diferenças centrais nas saídas e em todos os parâmetros da rede para as quatro fixtures, verificando finitude, regiões e dados congelados. Também testaram a rejeição de fixtures inválidas. **Não compararam nem aprovaram derivadas manuais**, pois os hooks continuam vazios. Não se adicionaram soluções analíticas, autodiff ou gradientes numéricos ao treino. GPU e benchmarks não foram reexecutados nesta correção; as medições anteriores acima são históricas.

## Backpropagation completo

Implementação e validação em **18/10/2026**, na mesma máquina descrita no início. Toda a matemática de derivação foi implementada exclusivamente em `crates/rl/src/manual.rs`: ativações, densa, tronco, loss/distribuição e combinação das cabeças. Sem dependências novas, autodiff, cópias para CPU no cálculo das derivadas ou mudanças na loss. A média entra somente nos gradientes de saída; as camadas somam as contribuições. ReLU em zero usa derivada zero; empates do mínimo PPO escolhem o ramo sem clip. Os checks centrais evitam essas fronteiras.

A worktree válida é `target/rl-publish`, branch `feat/ppo-manual-backprop`, HEAD `302f6b7`. A pasta principal continua com os metadados Git incompletos; não foram reparados. Os três arquivos já modificados na worktree foram preservados. `verification.rs` não foi editado nesta implementação: SHA-256 `ec5e9f0917d9ce28c93493ca48f0d027d6963e0097d3d6eda466cb43118b2a54`, igual à versão mais recente da revisão anterior. Nenhum commit ou push foi feito.

### Comandos realmente executados

| Comando/verificação | Resultado |
|---|---|
| `cargo check -p racing-rl --locked` | Passou após implementar os cinco hooks |
| `cargo run -p neuroevolution-racing-app -- rl check-gradients --case activation` | APROVADO, CPU f64 |
| Mesmo comando com `dense`, `trunk`, `heads`, `log_std`, `ppo`, `synthetic`, `mu`, `value`, individualmente e nessa ordem | Todos APROVADOS, CPU f64 |
| `cargo run -p neuroevolution-racing-app -- rl check-gradients --case all` | APROVADO, CPU f64 |
| `cargo test -p racing-rl --locked -- --nocapture` | Passou: 3 testes de fixtures e 14 testes de infraestrutura/gradientes, antes de adicionar o teste unitário de preflight/ReLU zero |
| `cargo fmt --all --check` | Passou, inclusive após a última edição do teste GPU |
| `cargo check --workspace --locked` | Passou; um aviso de arquivo incremental antigo ausente em `target`, sem erro de código |
| `cargo test --workspace --locked` | **222 passaram**: 80 neuroevolution, 124 app, 4 unitários RL e 14 testes de integração RL; zero falhas/ignorados |
| `cargo clippy -p racing-rl --all-targets --no-deps --locked -- -D warnings` | Passou |
| `cargo build -p neuroevolution-racing-app --features rl-gpu --locked` | Passou |
| `cargo test -p racing-rl --features gpu --test gpu --locked -- --nocapture` | **2 passaram na RX 6700 XT**, incluindo backward completo e passo de Adam CPU/GPU |
| `cargo clippy -p racing-rl --all-targets --features gpu --no-deps --locked -- -D warnings` | Passou, inclusive após a última edição GPU |

Os checks CPU usaram a suíte recente, sem alterar suas tolerâncias ou objetivos: ratios 0,55/1,05/1,55, vantagens positivas/negativas/zero, B=9/18, entropia 0/0,017, saídas mu/value/log_std isoladas e todos os parâmetros integrados. `old_log_prob` permaneceu congelado em todas as perturbações. Densa verificada com `(B,I,O)=(2,3,4)` e `(5,4,2)`. O sintético exige diminuição real de um objetivo com Adam.

O novo teste GPU compara dmu, dV, dlog_std, todos os gradientes dos parâmetros e os pesos após um passo de Adam com clipping global. Usa hidden `[8,5]`, B=9/18, a grade completa de clipping e entropia 0/0,017. Confere os ratios nos dois dispositivos antes dos gradientes. Tolerância f32: `abs 1e-4 + rel 1e-3*abs(referência)`. O teste anterior de forward/distribuição/loss/otimizador também passou. Adaptador real: **RX 6700 XT, Vulkan, driver 26.8.1, WGPU/WGSL f32**, sem fallback CPU.

### Treinos reais curtos

Foram criadas cópias temporárias de `configs/rl/smoke.ron` em `target/rl-backprop-smoke-cpu.ron` e `target/rl-backprop-smoke-gpu.ron`, com `total_rollouts: 3`. A cópia GPU muda também `device` para `Gpu(adapter: 0)`. Os presets versionados não foram alterados. N=2, T=8, M=8, E=2, hidden `[8,8]`, entropia zero e demais valores originais do smoke. Executou-se o binário debug realmente compilado:

```powershell
.\target\debug\neuroevolution-racing.exe rl collect --config target/rl-backprop-smoke-cpu.ron
.\target\debug\neuroevolution-racing.exe rl train --config target/rl-backprop-smoke-cpu.ron
.\target\debug\neuroevolution-racing.exe rl collect --config target/rl-backprop-smoke-gpu.ron
.\target\debug\neuroevolution-racing.exe rl train --config target/rl-backprop-smoke-gpu.ron
```

As coletas fornecem checkpoints iniciais sem updates para comparação. Cada treino concluiu **48 transições físicas reais e 12 passos de Adam**, sem `MissingManualImplementation`, NaN/Inf ou interrupção por KL. Três checkpoints, três rollouts e três arquivos de losses foram gravados por treino. `policy_version` e `optimizer.step` foram 4/8/12 nas três fronteiras. Todos os 12 minibatches produziram normas globais finitas e positivas: CPU `[0,35168776; 1,8389622]`, GPU `[0,35168773; 1,8389622]`.

| Rollout | Parâmetros alterados desde a fronteira anterior (CPU e GPU) | Maior alteração absoluta |
|---|---|---|
| 1, comparado à inicialização | 259 de 309 | 0,0011877 |
| 2, comparado ao checkpoint 1 | 261 de 309 | 0,00102432 |
| 3, comparado ao checkpoint 2 | 290 de 309 | 0,00107139 |

Os dois `log_std` também mudaram: de `[-0,5; -0,5]` para `[-0,5028015; -0,499578]`. Parâmetros de neurônios inativos podem permanecer iguais, como esperado para ReLU. A diferença máxima observada entre os checkpoints finais CPU/GPU foi aproximadamente **4e-9**; igualdade bit a bit não é exigida.

Artefatos desta rodada:

- Inicialização CPU: `results_ppo/ppo-1792337923278192800-27504/checkpoint-000000.ron`.
- Treino CPU: `results_ppo/ppo-1792337923611295400-25380/`.
- Inicialização GPU: `results_ppo/ppo-1792338100246241100-26220/checkpoint-000000.ron`.
- Treino GPU: `results_ppo/ppo-1792338102534639700-8232/`.
- Logs de comandos: `target/rl-backprop-*.log`; conferência de parâmetros/contadores/normas: `target/rl-backprop-artifact-verification.log`.

Os checkpoints finais foram restaurados e avaliados com:

```powershell
.\target\debug\neuroevolution-racing.exe rl evaluate --checkpoint results_ppo/ppo-1792337923611295400-25380/checkpoint-000003.ron --episodes 1 --cpu
.\target\debug\neuroevolution-racing.exe rl evaluate --checkpoint results_ppo/ppo-1792338102534639700-8232/checkpoint-000003.ron --episodes 1
```

Ambos concluíram um episódio por pista reservada, seed 55: Spa/Suzuka/Monaco, 120 ticks por episódio, todos terminando por timeout, sem colisão ou conclusão. Recompensas observadas: aproximadamente 0,88447/0,89493/0,88989; progresso normalizado 0,00574/0,00667/0,00491. Valores CPU/GPU coincidiram na precisão exibida. `trained=true` indica updates efetivos, **não que o carro aprendeu a dirigir**.

Também foi executada uma retomada CPU a partir do checkpoint GPU 3, usando cópia temporária com `total_rollouts: 4`. Executou somente o rollout restante:

```powershell
.\target\debug\neuroevolution-racing.exe rl train --config target/rl-backprop-smoke-resume-cpu.ron --resume results_ppo/ppo-1792338102534639700-8232/checkpoint-000003.ron
```

Resultado em `results_ppo/ppo-1792338139692250700-30196/checkpoint-000004.ron`: rollout 4, Adam.step=16, policy_version=16, parâmetros novamente alterados e finitos. Adam/parâmetros/RNGs foram restaurados; ambientes reiniciados, conforme a política existente de resume.

### Limites e arquivos desta correção

Código de produção alterado somente em `manual.rs`. `crates/rl/tests/gpu.rs` ganhou o teste de paridade de backward/Adam. `BACKPROP_PARA_IMPLEMENTAR.md`, `COMO_EXECUTAR.md` e este registro foram atualizados para o estado implementado e os comandos reais. As mudanças anteriores de `verification.rs` foram preservadas. Não houve alteração de GA, cenas/física, coleta, backends, otimizadores, persistência, manifests ou lockfile.

As execuções foram debug/dev, incluindo compilação de kernels e cache frio. Não constituem benchmark de desempenho, prova de convergência ou escolha validada de hiperparâmetros. O benchmark existente continua restrito a inferência/coleta; seu texto histórico de update indisponível não é um diagnóstico dos hooks atuais. `run.log` do treino registra tempos de update reais. O comando release para treino mais longo na Radeon está em [COMO_EXECUTAR.md](COMO_EXECUTAR.md); não foi iniciado automaticamente. Nenhuma função de backpropagation permanece pendente.
