# Área de aprendizado: cinco funções manuais

Todas as pendências ficam em **`crates/rl/src/manual.rs`**. Não existem derivadas escondidas em shaders, helpers, testes ou bibliotecas de treino. Cada stub retorna `Error::MissingManualImplementation { function }`. A infraestrutura não produz gradientes zero para simular sucesso.

`B: Backend` é o backend Burn sem autodiff; o mesmo código matemático deve servir CPU e WGPU. Use `Tensor<B,2>` e as primitivas reexportadas por `backend`. Não será preciso implementar upload, seleção de linhas, shaders, buffers de rollout, resets ou comunicação com o app.

## Ordem sugerida e contratos exatos

### 1. Ativações

```rust
pub fn activation_backward<B: Backend>(
    kind: Activation,
    z: Tensor<B, 2>,
    activation: Tensor<B, 2>,
    upstream: Tensor<B, 2>,
) -> Result<Tensor<B, 2>>
```

Arquivo: `crates/rl/src/manual.rs`, função `activation_backward`. Enum local: `Relu`, `Linear`, `Tanh`. Inputs e output `[B,H]`. `z` é a pré-ativação, `activation` a saída já calculada, upstream é a derivada do escalar objetivo em relação à saída. Retorne a contribuição em relação a z. Escolha e documente uma convenção em ReLU zero; os testes centrais evitam esse ponto. Não há média/redução adicional aqui.

Teste associado: `rl check-gradients --case activation` compara os três tipos com diferenças centrais. Não fornece fórmulas analíticas.

### 2. Camada densa linear

```rust
pub fn dense_backward<B: Backend>(
    layer: &Dense<B>,
    input: Tensor<B, 2>,
    upstream: Tensor<B, 2>,
) -> Result<DenseGradients<B>>
```

Arquivo: `crates/rl/src/manual.rs`, função `dense_backward`. Input X `[B,I]`; layer.weights `[O,I]`, layer.bias `[1,O]`; upstream `[B,O]` é relativo à saída **linear**, depois da ativação já ter sido tratada pelo chamador.

Retorno `DenseGradients { input: [B,I], weights: [O,I], bias: [1,O] }`. A regra forward é `X*Wᵀ+b`. Bias faz broadcast entre linhas; preserve shapes e soma das contribuições. Não divida por B. Pesos são emprestados e não podem ser atualizados aqui.

Teste: `--case dense`, incluindo pesos, biases e entradas, com upstream fixo.

### 3. MLP/tronco

```rust
pub fn trunk_backward<B: Backend>(
    layers: &[Dense<B>],
    caches: &[LayerCache<B>],
    upstream: Tensor<B, 2>,
) -> Result<TrunkGradients<B>>
```

Arquivo: `crates/rl/src/manual.rs`, função `trunk_backward`. Recebe as hidden layers em ordem forward e um cache por camada. Cada cache guarda `input [B,I]`, `preactivation [B,O]`, `activation [B,O]`. Upstream `[B,H_final]` já contém as contribuições combinadas das cabeças.

Retorne `TrunkGradients { input: [B,D], parameters: Vec<Tensor<B,2>> }`. `parameters` deve seguir a ordem **forward** `W0,b0,W1,b1,...`, embora o processamento de dependências seja feito na ordem necessária ao backward. Use os hooks de ativação e dense. Não normalize, não modifique os caches/weights e não acumule resíduos de outro minibatch.

Teste: `--case trunk`, objetivo escalar com upstream fixo. O checker deve comparar parâmetros e entrada. A infraestrutura de inicialização/forward não precisa ser refeita.

### 4. Duas cabeças e tronco compartilhado

```rust
pub fn shared_heads_backward<B: Backend>(
    network: &Network<B>,
    cache: ForwardCache<B>,
    outputs: OutputGradients<B>,
) -> Result<Vec<Tensor<B, 2>>>
```

Arquivo: `crates/rl/src/manual.rs`, função `shared_heads_backward`. `outputs.mean [B,2]`, `outputs.value [B,1]`, `outputs.log_std [1,2]` são derivadas relativas às três saídas/parâmetros globais. `cache.head_input [B,H]` alimentou ambas cabeças. Actor e critic são lineares; critic não passa por Tanh.

Calcule os ramos das cabeças com `dense_backward`, combine suas contribuições para a entrada compartilhada e use `trunk_backward`. Retorne **todos** os parâmetros na ordem de `Network::parameters()`:

1. W,b de cada hidden em ordem forward;
2. W,b actor;
3. W,b critic;
4. log_std global `[1,2]`.

Não aplique média extra, clipping ou optimizer. O cache é consumido por valor; contém referências contadas aos tensores do forward, e deve ser descartado ao fim do backward. A versão deve coincidir com `network.version()` (o orquestrador já chama `validate_cache`). Nunca use um cache depois de `Network::replace`/restore/update. Nenhum parâmetro pode mudar enquanto o backward completo está em andamento.

Teste: `--case heads` usa upstreams fixos para os dois ramos e para log_std, comparando todos os parâmetros. Também verifica indiretamente a soma dos ramos no tronco.

### 5. Distribuição e loss PPO

```rust
pub fn ppo_output_gradients<B: Backend>(
    cache: &LossCache<B>,
) -> Result<OutputGradients<B>>
```

Arquivo: `crates/rl/src/manual.rs`, função `ppo_output_gradients`. Outputs: derivadas em relação a mu `[B,2]`, value `[B,1]`, log_std global `[1,2]`.

`LossCache` contém:

| Campo | Shape/significado |
|---|---|
| distribution.mean | `[B,2]`, mu atual |
| distribution.log_std | `[1,2]`, dois parâmetros compartilhados |
| distribution.u | `[B,2]`, pré-Tanh histórico congelado |
| distribution.action | `[B,2]`, ação histórica Tanh |
| distribution.joint_log_density | `[B,1]`, log-densidade conjunta reavaliada |
| distribution.latent_entropy | `[1,1]`, entropia auxiliar da normal latente |
| inputs.old_log_prob | `[B,1]`, congelado |
| inputs.advantage | `[B,1]`, normalizada e congelada |
| inputs.value_target | `[B,1]`, alvo bruto congelado |
| value | `[B,1]`, critic atual |
| ratio / clipped_ratio | `[B,1]`, cache do objetivo |
| total | `[1]`, loss escalar de minimização |
| epsilon / coeficientes / batch_size | configuração usada por este forward |

O forward está em `distribution.rs` e `ppo.rs`. Produza as derivadas do objetivo de minimização completo, incluindo value_coef e o sinal de latent_entropy_coef. Não há value clipping. Exploração existe pelo sorteio mesmo com entropia_coef zero.

**Inclua exatamente uma média por B aqui.** As demais funções não dividem por B. log_std é compartilhado pelas linhas: o retorno deve já combinar todas as suas contribuições. O regularizador de entropia latente é escalar e não deve ganhar um fator B indevido por broadcast.

Observações, u, ações, dados antigos, vantagens e targets são constantes de treinamento. O backward não atravessa o simulador, o reset ou o sorteio do rollout. Não sorteie ação nova durante o update; não recupere u com atanh. A projeção de log_std acontece **depois** do optimizer; não faz parte do grafo a derivar aqui.

Testes: `--case log_std` isola os dois parâmetros globais; `--case ppo` verifica a rede completa com distribuição/loss; `--case synthetic` exige redução de um objetivo congelado usando os hooks e Adam. Casos de forward com vantagens positivas, negativas e zero e ratios fora da faixa ficam em `crates/rl/tests/infrastructure.rs`.

## Executar e interpretar os testes

```powershell
cargo run -p neuroevolution-racing-app -- rl check-gradients --case list
cargo run -p neuroevolution-racing-app -- rl check-gradients --case activation
cargo run -p neuroevolution-racing-app -- rl check-gradients --case dense
cargo run -p neuroevolution-racing-app -- rl check-gradients --case trunk
cargo run -p neuroevolution-racing-app -- rl check-gradients --case heads
cargo run -p neuroevolution-racing-app -- rl check-gradients --case log_std
cargo run -p neuroevolution-racing-app -- rl check-gradients --case ppo
cargo run -p neuroevolution-racing-app -- rl check-gradients --case synthetic
cargo run -p neuroevolution-racing-app -- rl check-gradients --case all
```

Enquanto falta função, termina com código de erro, nome identificável e `PENDENTE`. Somente após uma comparação realmente passar imprime `APROVADO`. A suíte de infraestrutura pode passar verificando que a pendência foi reportada corretamente; isso **não** significa que a derivada passou. Use `cargo test -p racing-rl -- --nocapture` para ver esses estados.

Checker: `crates/rl/src/verification.rs`. Usa CPU NdArray **f64**, diferenças finitas **centrais**, passo 1e-5 e tolerância `abs 2e-6 + rel 2e-4*abs(referência)`. Fixtures pequenas, sem sorteio no objetivo, pesos/inputs/upstreams/u fixos; hidden preactivations afastadas de zero. Perturbações não devem cruzar os pontos não diferenciáveis de ReLU/min/clip; ao adicionar casos, confira essa condição. Não há soluções analíticas de teste para copiar.

Diferenças finitas são validação, nunca fonte de gradientes do optimizer. O preflight de treino chama todos os hooks reais num caso pequeno, sem diferenças finitas; o update usa exclusivamente os retornos manuais. Execute os gradient checks explicitamente antes de iniciar treino longo.

`simulation::rl_environment::tests::real_rollout_determinism_resets_at_different_times_and_manual_train_gate` coleta com física real e só exige optimizer.step>0 depois que o preflight passar; enquanto isso mostra `PENDENTE PPO train smoke`. O sintético e esse smoke não demonstram aprendizado na pista. Tal demonstração exigirá treinamento real e avaliação separada.
