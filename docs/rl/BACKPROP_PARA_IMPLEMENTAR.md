# Backpropagation explícito: cinco funções implementadas

As cinco funções estão implementadas em **`crates/rl/src/manual.rs`**, com operações tensoriais vetorizadas e comentários em português. Toda a matemática do backward permanece nesse arquivo, sem autodiferenciação, derivadas em shaders ou diferenças finitas no treino. O nome histórico deste documento foi preservado como guia de estudo.

`B: Backend` é o backend Burn sem autodiff; o mesmo código matemático serve CPU e WGPU. `Tensor<B,2>` e as primitivas reexportadas por `backend` mantêm as operações no dispositivo. Upload, seleção de linhas, buffers, resets, otimizadores e comunicação com o app continuam na infraestrutura existente.

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

Arquivo: `crates/rl/src/manual.rs`, função `activation_backward`. Enum local: `Relu`, `Linear`, `Tanh`. Inputs e output `[B,H]`. `z` é a pré-ativação, `activation` a saída já calculada, upstream é a derivada do escalar objetivo em relação à saída. Retorna a contribuição em relação a z pela regra da cadeia. ReLU em zero usa derivada zero, verificada em teste unitário; os testes centrais evitam esse ponto. Não há média/redução adicional aqui.

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

Teste: `--case dense`, incluindo pesos, biases e entradas, com upstream fixo. Usa `(B,I,O)=(2,3,4)` e `(5,4,2)`: pesos, entradas e upstream não são quadrados. As shapes dos gradientes são verificadas antes de comparar os valores, para detectar transposições e reduções no eixo errado.

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

O forward está em `distribution.rs` e `ppo.rs`. O backward deriva exatamente esse objetivo de minimização, incluindo value_coef e o sinal de latent_entropy_coef. Não há value clipping. Exploração existe pelo sorteio mesmo com entropia_coef zero. O mínimo zera a contribuição da policy para vantagem positiva acima do limite superior e vantagem negativa abaixo do limite inferior. Nos empates, inclusive nas fronteiras, escolhe-se o ramo sem clip, como no forward; os checks evitam essas fronteiras. Com u histórico fixo, a correção Jacobiana da Tanh é constante em relação a mu e log_std.

**Inclua exatamente uma média por B aqui.** As demais funções não dividem por B. log_std é compartilhado pelas linhas: o retorno deve já combinar todas as suas contribuições. O regularizador de entropia latente é escalar e não deve ganhar um fator B indevido por broadcast.

Observações, u, ações, dados antigos, vantagens e targets são constantes de treinamento. O backward não atravessa o simulador, o reset ou o sorteio do rollout. Não sorteie ação nova durante o update; não recupere u com atanh. A projeção de log_std acontece **depois** do optimizer; não faz parte do grafo a derivar aqui.

Testes: `--case mu`, `--case value` e `--case log_std` comparam separadamente cada grupo de saída, mantendo os outros dois fixos. `--case ppo` verifica todos os parâmetros pela propagação integrada com distribuição/loss. `--case synthetic` exige redução de um objetivo congelado usando os hooks e Adam; é um smoke separado, sem função de cobertura das regiões de clipping.

Os quatro casos numéricos PPO usam `FrozenPpoFixture`, em `crates/rl/src/verification.rs`, com a seguinte grade de **nove combinações** em cada minibatch:

| Vantagem fixa | Ratio 0,55 (abaixo) | Ratio 1,05 (dentro) | Ratio 1,55 (acima) |
|---|---|---|---|
| Positiva | objetivo sem limitação pelo clip | objetivo sem limitação pelo clip | objetivo limitado pelo clip |
| Negativa | objetivo limitado pelo clip | objetivo sem limitação pelo clip | objetivo sem limitação pelo clip |
| Zero | termo de policy constante | termo de policy constante | termo de policy constante |

Cada caso percorre **B=9 e B=18**, ambos com `latent_entropy_coef=0` e `0,017`: quatro fixtures por caso. B=18 repete a grade com magnitudes de vantagem diferentes e entradas/u variados. `value_coef=0,5` e epsilon=0,2 ficam fixos. O checker não contém soluções analíticas; compara o backward real de `manual.rs` com diferenças finitas.

`old_log_prob` é calibrado **uma única vez**, a partir da log-densidade da rede não perturbada e dos ratios desejados, e armazenado na fixture. Todas as avaliações seguintes clonam esse mesmo tensor. Não o recalcule a partir da política perturbada: isso mudaria o objetivo da diferença finita. Observações, u histórico, vantagens e targets também permanecem fixos; não há RNG durante as perturbações.

Antes de chamar o hook, a fixture confere os ratios iniciais contra os valores desejados (tolerância absoluta `1e-12`), o sinal/magnitude das vantagens e qual ramo limita o objetivo. **Cada avaliação perturbada** volta a verificar os dados congelados, a região de ratio e a seleção do ramo. A margem mínima exigida em relação às fronteiras de clipping é `0,1`; as pré-ativações do tronco devem permanecer positivas e maiores que `0,05`. Assim, um stencil que cruze uma fronteira gera erro de fixture, em vez de uma comparação de gradientes enganosa. Vantagem zero é uma constante histórica: o empate dos termos de policy permanece zero para todas as perturbações. Dentro da faixa, os dois termos coincidem como funções, sem criar uma troca de ramo não diferenciável nesse ponto.

## Executar e interpretar os testes

```powershell
cargo run -p neuroevolution-racing-app -- rl check-gradients --case list
cargo run -p neuroevolution-racing-app -- rl check-gradients --case activation
cargo run -p neuroevolution-racing-app -- rl check-gradients --case dense
cargo run -p neuroevolution-racing-app -- rl check-gradients --case trunk
cargo run -p neuroevolution-racing-app -- rl check-gradients --case heads
cargo run -p neuroevolution-racing-app -- rl check-gradients --case mu
cargo run -p neuroevolution-racing-app -- rl check-gradients --case value
cargo run -p neuroevolution-racing-app -- rl check-gradients --case log_std
cargo run -p neuroevolution-racing-app -- rl check-gradients --case ppo
cargo run -p neuroevolution-racing-app -- rl check-gradients --case synthetic
cargo run -p neuroevolution-racing-app -- rl check-gradients --case all
```

Os casos individuais e `all` agora imprimem `APROVADO` após comparações reais. O teste `manual::tests::completed_hooks_and_relu_zero_convention` exige que o preflight dos cinco hooks passe: regressão para stub falha a suíte, em vez de ser aceita como pendência. Use `cargo test -p racing-rl -- --nocapture` para ver os resultados por caso.

| Caso | O que será comparado com os hooks reais |
|---|---|
| `activation` | ReLU, Linear e Tanh com upstream congelado |
| `dense` | dW, db e dX em duas shapes não quadradas |
| `trunk` | parâmetros e entrada do tronco |
| `heads` | parâmetros de ambas cabeças, tronco e log_std com upstreams fixos |
| `mu` | cada coordenada de dmu `[B,2]`, sem perturbar value/log_std |
| `value` | cada coordenada de dV `[B,1]`, sem perturbar mu/log_std |
| `log_std` | os dois parâmetros globais `[1,2]`, sem perturbar mu/value |
| `ppo` | todos os parâmetros da rede, com a mesma grade de clipping e histórico congelado |
| `synthetic` | diminuição de um objetivo pequeno usando hooks e Adam |
| `all` | todos os casos anteriores; encerra na primeira pendência/erro |

Os testes unitários de `verification::tests` verificam a grade, os stencils numéricos dos três grupos e da rede, a preservação dos dados e a restauração dos pesos. Também verificam que a guarda rejeita região errada, histórico alterado e aproximação excessiva da fronteira. Esses testes validam **a infraestrutura numérica**. O teste de integração de nome histórico `manual_suite_reports_pending_instead_of_fake_success` chama os hooks reais e agora aprova todos os casos por comparação numérica.

Checker: `crates/rl/src/verification.rs`. Usa CPU NdArray **f64**, diferenças finitas **centrais**, passo 1e-5 e tolerância `abs 2e-6 + rel 2e-4*abs(referência)`. Fixtures pequenas, sem sorteio no objetivo, pesos/inputs/upstreams/u fixos; hidden preactivations afastadas de zero. Perturbações não devem cruzar os pontos não diferenciáveis de ReLU/min/clip; ao adicionar casos, confira essa condição. Não há soluções analíticas de teste para copiar.

Diferenças finitas são validação, nunca fonte de gradientes do optimizer. O preflight de treino chama todos os hooks reais num caso pequeno, sem diferenças finitas; o update usa exclusivamente os retornos manuais. Execute os gradient checks explicitamente antes de iniciar treino longo.

`simulation::rl_environment::tests::real_rollout_determinism_resets_at_different_times_and_manual_train_gate` coleta com física real e exige optimizer.step>0 depois do preflight, que agora passa. O teste hardware `actual_hardware_backward_clipping_entropy_and_adam_parity` compara saídas do backward, todos os gradientes e parâmetros após Adam em CPU/WGPU f32, com B=9/18, a grade de clipping e entropia 0/0,017. Tolerância: `abs 1e-4 + rel 1e-3*abs(referência)`.

O sintético e os smokes não demonstram aprendizado na pista. Veja [VALIDACAO.md](VALIDACAO.md) para os comandos efetivamente executados, treinos reais de três rollouts e avaliação separada.
