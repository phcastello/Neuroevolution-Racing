# PPO-Clip no Racing

O crate `racing-rl` é independente de Bevy, Avian, egui e do núcleo GA. O app depende dos dois crates de IA; `neuroevolution` continua sem dependência de ML. Os formatos históricos de genoma, Champion e TrainingCheckpoint não foram modificados. As únicas mudanças na física antiga são visibilidades para reutilizar funções existentes.

## Caminho executável

`simulation/rl_environment.rs` mantém N estados cinemáticos privados. Cada pista tem uma cena Avian estática compartilhada que contém **somente paredes**, construídas pelo mesmo `spawn_track_colliders` do GA. Carros não são colliders nem entidades consultáveis dessas cenas. Uma pista diferente tem outro World e outra árvore espacial. O adaptador reutiliza `integrate_vehicle`, `resolve_vehicle_motion`, `Track`, projeção local e raycasts Avian. Não instancia TrainingState, geração, fitness ou laser.

O coletor observa N carros, envia `[N,D]` ao dispositivo e usa uma mesma rede. Aplica `[steering, acceleration]` através do adaptador existente para `CarControls::new(acceleration, steering)`. Steering positivo gira no sentido anti-horário; aceleração negativa permite frenagem/ré conforme a física antiga.

Cada decisão mantém a ação por `action_repeat` ticks de 1/60 s, interrompe no término e soma as recompensas dos ticks executados. Gamma é **por decisão RL**, sem exponenciação pelo número de ticks. A taxa física, decisões, transições e frames renderizados são conceitos distintos; treino/coleta têm zero frames. A cena estática é inicializada antes da medição do rollout. A simulação usa CPU sequencial nesta entrega; não exige uma thread/processo por carro.

Depois da integração: resolve colisão, mede a rotação realmente realizada, projeta progresso, calcula recompensa/término e **reamostra os sensores**. Retorna a observação final antes de qualquer reset. Cada ambiente tem índice/env_id estável, episode_id, RNG, relógio, ação anterior e progresso próprios. O coletor reseta somente quem acabou; um episódio em andamento no final do buffer continua no próximo rollout.

## Observação permitida

`ObservationSpec` v1 persiste tipo, ângulos e escalas; `features()` gera os nomes, ordem, origem, unidades e normalização guardados no checkpoint.

| Modo | Ordem |
|---|---|
| Legacy6 | raios +60,+30,0,-30,-60 graus; `abs(v)/(abs(v)+250)` |
| Extended inicial | 21 raios de +90 a -90 graus, passo -9; velocidade longitudinal assinada; velocidade angular medida; steering anterior; acceleration anterior |

Distâncias são divididas por 1000 unidades; velocidade assinada por 250 unidades/s; yaw medido por 2.15 rad/s. Extended limita medidas a [-1,1], raios a [0,1]. Ângulos/quantidade e escalas são configuráveis. As duas ações anteriores são as realmente aplicadas, e yaw usa a diferença de heading **resolvido**, incluindo rejeição de rotação por parede. Reset zera yaw e ações. O estado inicial tem velocidade 45, como o início canônico GA.

Actor e critic recebem a mesma entrada. Nenhum recebe posição global, track id, geometria futura, progresso, fitness, alvos ou outros carros. Projeção interna e término estão somente no adaptador/recompensa. Sensores numerosos não garantem observabilidade completa. Escalas ficam congeladas; não há normalização adaptativa durante coleta/épocas.

## Recompensa v1

Progresso é a distância acumulada assinada desde o spawn, com correção da emenda da pista e projeção local. Recompensa somente incremento do **máximo positivo já alcançado**, limitado a uma volta. Ré e recuperação do mesmo trecho não dão novos pontos. Projeções com delta maior que `4 * deslocamento_real + 2 unidades` são rejeitadas, sem reancorar/creditar salto; isso é uma heurística conservadora e pode precisar de calibração para outras pistas. Reset estabelece nova origem e não retorna recompensa.

Defaults: `0.01 * incremento_em_unidades`, `+10` uma vez pela conclusão de uma volta desde o spawn, `-1` pela colisão. São hipóteses de engenharia configuráveis, separadas da fitness GA. O Step guarda os três componentes. Qualquer interseção física na rotação ou translação encerra o episódio PPO; a regra antiga GA continua intacta. Colisão e conclusão são `terminated`; limite operacional de ticks é `truncated`. Laser está desativado no PPO.

## Tensores e rede

Burn **0.20.1**, subcrates fixados com defaults desligados. CPU usa NdArray sem feature `std`, pois ela traz `burn-autodiff` nessa versão; `burn-tensor/std` é habilitada separadamente. GPU opcional usa WGPU/WGSL f32, com adaptador Vulkan/DX12 escolhido e registrado explicitamente. Não usa CUDA, autodiff, rede, loss ou otimizadores prontos do Burn. Features resolvidas devem ser auditadas com `cargo tree`, inclusive no app.

Layout lógico contíguo row-major; pesos `W[O,I]`, biases `[1,O]`, `Y=X*transpose(W)+bias`. Ordem dos parâmetros PPO: W,b de cada hidden, W,b actor, W,b critic e log_std `[1,2]`. O layout histórico GA não muda.

Hidden layers: ReLU, He normal (desvio sqrt(2/fan_in)); biases zero. Cabeça actor linear com desvio 0.01/sqrt(fan_in), critic linear com desvio 1/sqrt(fan_in), sem Tanh. Inicialização usa ChaCha12 local determinístico, gerada uma vez na CPU e enviada ao backend. Pesos continuam residentes entre forwards; o snapshot só é lido para persistência.

Dois log_std globais treináveis, independentes do estado. Depois de cada update completo, projeta em [-5,2] por default. Estão no otimizador, clipping global e checkpoint. Não há mixed precision. CPU f64 é referência para testes pequenos, não o preset de treino.

Inferência não guarda caches. Forward de treino guarda X, pré-ativação, ativação de cada hidden, entrada compartilhada das cabeças, mu, V e log_std, além da versão dos pesos. Orquestrador rejeita caches obsoletos, e não atualiza pesos até concluir os dois ramos. Gradientes retornam novos buffers por minibatch, sem acumulação implícita.

## Política e loss

`u=mu+exp(log_std)*epsilon`, epsilon independente por amostra/dimensão; ação `tanh(u)`. Avaliação usa `tanh(mu)`. Guarda u, ação e log-densidade conjunta antiga, com correção Tanh calculada por softplus estável. Não usa atanh de ação saturada nem clipping de uma normal. No update reavalia o **mesmo u histórico**. Não há derivação pelo sorteio passado ou pelo simulador.

Loss de minimização: policy PPO-Clip negativa média; value `0.5*mean((V-target)^2)`; total policy + value_coef*value - latent_entropy_coef*latent_entropy. Sem value clipping. `latent_entropy` é entropia analítica da **Gaussiana latente**, não das ações após Tanh; coeficiente inicial zero. O cache guarda inputs e ramos necessários ao aluno, sem derivadas prontas.

Cada média é aplicada uma única vez sobre B na loss; os hooks de saída incluem exatamente esse fator. Dense/tronco/cabeças somam contribuições sem dividir novamente por B. Razão, fração de ratios fora da faixa e KL aproximada `mean(ratio-1-log_ratio)` são métricas. Uma vantagem zero também pode ter ratio fora da faixa. Se a KL exceder target_kl, interrompe os updates restantes daquele rollout antes do próximo step. Clipping do objetivo não limita rigidamente a mudança da política.

NaN/Inf e inconsistências de shapes geram erros. O PPO valida ações antes de usar o adaptador histórico que sanitiza controles; não neutraliza silenciosamente erros numéricos.

## Rollout, GAE e updates

Buffer `[T,N,...]`; `rollout_size=N*T`. Cada transição guarda ids, observação, u/ação, reward parts, flags/motivo, old_log_prob, old_value, observação final, next_value, ticks e progresso de diagnóstico. Pesos ficam congelados durante toda coleta com policy_version.

GAE usa duas máscaras: término real elimina bootstrap; término e truncamento eliminam recursão. Timeout bootstrap usa a observação FINAL do episódio anterior ao reset. No fim do buffer ainda em andamento há bootstrap, sem reset. Targets são `vantagem_bruta + old_value`. Só a vantagem da policy é normalizada, com proteção de variância zero. Casos minúsculos verificam todos esses limites e múltiplos ambientes.

Após GAE, as linhas time-major são achatadas e enviadas ao dispositivo uma vez. Cada época embaralha índices com RNG próprio e seleciona os mesmos índices em todos os campos. Último minibatch menor é aceito, todas as amostras cobertas. Old dados, vantagens e targets não mudam. O buffer é descartado após as épocas; não há replay histórico.

Update: forward atual/cache → loss/cache → hook de saídas → hook de parâmetros/cabeças/tronco → validação de todos os gradientes → uma norma global → SGD/Adam → projeção de log_std. Adam implementa primeiro/segundo momentos, correção de viés e contador; testes usam gradientes fornecidos, sem alegar aprendizado.

## Persistência e medição

Cada execução cria diretório exclusivo `results_ppo/ppo-<nanos>-<pid>`, manifesto, run.log/métricas, rollout e checkpoint próprios. Arquivos são criados sem sobrescrita: escreve temp no mesmo diretório, flush/sync, rename. Checkpoints somente na fronteira de rollout completo/update completo.

Formato separado `racing-ppo` v1: arquitetura/config, shapes/parâmetros/log_std, Adam, contadores, estados ChaCha completos, ObservationSpec e metadados, recompensa, física e backend. A posição u128 do ChaCha é persistida como dois u64. Retomada restaura pesos/Adam/RNGs e **reinicia ambientes**; não promete continuidade exata de trajetórias. Cria outra execução, preservando o checkpoint fonte. Avaliação usa somente validation suite, RNGs independentes, pesos congelados e marca inicialização aleatória como UNTRAINED.

Benchmarks usam seed/inputs iguais entre backends, verificam paridade CPU, fazem warm-up e sincronização. Medem latência/rows por segundo e rollout real; separam observação/reset, upload, inferência/distribuição, download, simulação e GAE. As consultas finais de sensores fazem parte da simulação no temporizador, e observação inicial/reset tem campo separado. As operações de uma camada podem usar vários kernels. Tempo de treino completo fica indisponível até o backward; não é extrapolado.

Limites: coleta inicial é sequencial na CPU; Adam e seleção de minibatches ficam no backend, mas métricas/validação leem pequenos escalares. RON legível pode ocupar espaço significativo com redes/buffers grandes. Estimativa de memória é conservadora para tensores/buffers, não inclui toda memória de Bevy, driver e compilação de kernels. Stress e grades maiores são opt-in.

## Fontes conferidas

- [PPO-Clip e parada por KL](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
- [GAE](https://arxiv.org/abs/1506.02438).
- [Término versus truncamento](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- [Transformação Gaussiana/Tanh](https://spinningup.openai.com/en/latest/algorithms/sac.html), usada somente para a distribuição; o algoritmo é PPO.
- [Backends Burn](https://burn.dev/books/burn/basic-workflow/backend.html) e documentação específica [burn-tensor 0.20.1](https://docs.rs/burn-tensor/0.20.1/), [NdArray 0.20.1](https://docs.rs/burn-ndarray/0.20.1/), [WGPU 0.20.1](https://docs.rs/burn-wgpu/0.20.1/), conferidas também nos fontes efetivamente resolvidos pelo Cargo.
