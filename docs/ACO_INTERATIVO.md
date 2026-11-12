# ACO interativo: uso e evidências

**Visualização atual:** o ACO ocupa a cena principal, com **Ver melhor pista**, **Ver geração** e **Ver pista** por candidata. As janelas flutuantes do ACO foram removidas. Veja [fluxo atual e capturas reais](ACO_VISUALIZACAO_PRINCIPAL.md).

Agora a aba **Candidatas** reúne aceitas e rejeitadas de toda a busca, com filtros, comparação, diagnóstico físico e exportação individual. Veja [largura física e inspeção](ACO_LARGURA_INSPECAO.md).

O modo sequencial atual inclui três fases, motivos para fechar/passear e métricas de organicidade. Veja [ajustes e comparação visual](ACO_ORGANICO.md).

Abra o executável normalmente, selecione **ACO**, configure formigas, iterações e seed e clique em **Iniciar**. Não é necessário gerar um arquivo no terminal. Os parâmetros adicionais ficam em **Parâmetros principais / geometria**; **Avaliar com PPO congelado** exibe a lista de checkpoints existentes para seleção por clique. A política escolhida é carregada em CPU pela tarefa de busca.

**Pausar** suspende o cálculo em um ponto seguro e mostra a confirmação **Pausado · cálculo interrompido**. **Continuar**, **Uma decisão**, **Uma iteração**, **Interromper**, **Reiniciar com mesma seed** e **Decisões/s** controlam a execução. O passo por iteração conclui a iteração em andamento. Ao sair do modo ACO, a busca pausa; ao retornar, clique em Continuar.

A construção parcial v2 mostra o percurso de primitivas por posição/orientação: azul para o caminho, amarelo para o segmento escolhido, cinza para alternativas viáveis, vermelho para rejeitadas e verde para a continuação de fechamento já verificada. Somente a candidata completa é mostrada como pista com bordas e validação. O modo radial legado conserva seus pontos de controle e Catmull-Rom. Probabilidades e feromônios pertencem à decisão real publicada pelo motor. Veja [gerador sequencial e evidências atuais](ACO_SEQUENCIAL.md).

**Salvar / exportar / reproduzir arquivo** permite salvar as iterações concluídas, exportar pistas e abrir buscas anteriores. Buscas sequenciais e proveniência usam versão 2; o modo legado permanece v1, e arquivos antigos continuam legíveis. Arquivos existentes não são sobrescritos. A interrupção preserva na interface as iterações já concluídas; uma formiga/iteração interrompida não é salva como resultado completo.

## Arquitetura

- `SearchArchive::run_iteration` delega a `run_iteration_observed`. CLI, currículo PPO e interface compartilham seleção probabilística, avaliação, fitness, validação e atualização de feromônios.
- `SearchObserver` publica início de formiga, cada decisão, geometria completa, resultado da candidata e iteração. O gerador compartilha o cálculo dos pontos de controle com a visualização.
- `LiveSearch` executa o motor em uma thread, com caixa de mensagens limitada a oito eventos, `Mutex`/`Condvar` e cancelamento cooperativo. Não há `join` na thread do Bevy. Um reinício cancela a tarefa anterior e substitui seu canal.
- O checkpoint e os ambientes PPO são criados e mantidos dentro da tarefa. Os mesmos episódios de referência e PPO consultam o controle antes de cada decisão física/inferência. Uma inferência individual já em andamento termina antes da confirmação de pausa; o renderizador continua independente.
- O replay usa os arquivos existentes e não executa o motor. Ele também respeita o número de decisões visíveis antes de apresentar um circuito completo.

## Validação executada em 11/11/2026

Sem benchmarks pesados. Ambiente nativo Windows 11, Bevy 0.19.1, Vulkan e Radeon RX 6700 XT.

| Critério | Evidência |
| --- | --- |
| Iniciar pela interface e executar ACO real | Teste com eventos de ponteiro nos botões egui e teste nativo Bevy/WGPU acionando Iniciar. Nenhum arquivo de busca é carregado. |
| Decisões progressivas e geometria parcial | Passos individuais publicam decisões antes de qualquer candidata; pontos parciais conferidos com os pontos do gerador. Captura nativa mostra três pontos/segmentos abertos. |
| Circuito completo e validação | Captura nativa após a primeira iteração mostra os 12 controles, bordas do circuito e geometria validada. A segunda candidata é rejeitada por curvatura. |
| Feromônios reais, evaporação e reforço | Testes verificam as três matrizes e reforço positivo. Captura após duas iterações mostra valores 0,6400 e 0,9986, partindo de 1,0000. |
| Pausa computacional, retomada e passos | Contadores/eventos ficam imóveis durante a pausa, tanto no motor quanto nos ticks de avaliação. O teste nativo verifica imobilidade por 15 frames, avança uma iteração e retoma. |
| Mesma seed | Reinício pela interface e nova tarefa com configuração idêntica reproduzem decisões e feromônios. A tarefa visual também coincide com o headless em decisões, pistas selecionadas e feromônios. |
| Interrupção e troca de modo | Clique no botão Interromper encerra uma busca pausada. A suspensão ao sair do modo congela o contador de decisões. |
| PPO congelado | A tarefa carrega um checkpoint real: 1.509 decisões de referência e 157 decisões PPO. Arquivo e parâmetros permanecem iguais. Teste adicional pausa/retoma e cancela entre ticks reais. |
| Replay, persistência e exportação | Testes de round-trip, RNG, replay sem mutação, criação de diretório pela interface e exportação de pistas com proveniência. |
| GA/PPO e headless | Suíte do workspace: 391 aprovados, quatro ignorados. O teste gráfico ignorado foi executado separadamente e passou. Sem recursos visuais: 210 aprovados, três ignorados. O CLI também executou busca, replay e exportação de duas pistas, com quatro candidatas em duas iterações. |

Comandos de reprodução dos testes:

```powershell
cargo test --workspace -- --nocapture
cargo test -p neuroevolution-racing-app --no-default-features -- --nocapture
cargo test -p neuroevolution-racing-app native_bevy_start_progress_pause_resume_and_capture -- --ignored --nocapture
```

O teste nativo abre uma janela, injeta eventos de ponteiro no fluxo real de entrada egui e captura frames pelo próprio renderizador. É opt-in para não exigir desktop/GPU na suíte normal. A captura externa do desktop falhou durante a tentativa inicial; a validação nativa e a inspeção das imagens produzidas pelo Bevy foram concluídas.

Evidências desta execução (artefatos locais em `target`, não versionados):

- [Construção parcial pausada](../target/aco-rendered-smoke/1794425637814064500/01-partial-paused.png)
- [Circuito completo validado](../target/aco-rendered-smoke/1794425637814064500/02-complete-validated.png)
- [Feromônios após duas iterações](../target/aco-rendered-smoke/1794425637814064500/03-pheromones.png)
- [Busca nativa salva](../target/aco-rendered-smoke/1794425637814064500/search.ron)
- [Log do teste nativo](../target/aco-rendered-smoke.log)
- [Log do workspace](../target/aco-live-tests.log)
- [Log sem recursos visuais](../target/aco-headless-tests.log)
- [Busca produzida pelo CLI](../target/aco-cli-validation/1794425771035/search/search.ron)
