# Visualizador PPO integrado

O comando `rl view` abre a experiência visual do Neuroevolution-Racing para uma política PPO congelada: pista com a renderização original, sprites dos veículos, câmera compartilhada, painel egui, sensores, hitboxes e inspeção da rede Actor-Critic. Assistir às corridas não inicia treinamento.

## Abrir um checkpoint

Na raiz do repositório, execute no PowerShell:

Nesta cópia do projeto, o cache incremental existente em `target` contém objetos ausentes. Para usar os comandos abaixo sem depender desse cache, execute antes `$env:CARGO_INCREMENTAL = '0'` no mesmo PowerShell. Isso afeta somente a compilação, não a inferência ou os checkpoints.

```powershell
cargo run --locked -p neuroevolution-racing-app -- rl view --checkpoint results_ppo/ppo-1792340830035034100-29472/checkpoint-000100.ron --cars 1
```

Para começar em uma pista específica:

```powershell
cargo run --locked -p neuroevolution-racing-app -- rl view --checkpoint results_ppo/ppo-1792340830035034100-29472/checkpoint-000100.ron --cars 1 --track spa
```

Substitua o caminho para abrir outro checkpoint PPO. `--cars` aceita de 1 a 16 carros. Sem `--track`, o visualizador usa a primeira pista de avaliação configurada no checkpoint, com fallback para a biblioteca de validação. O seletor **Track** permite trocar a pista depois de abrir a janela, incluindo pistas de treinamento e validação.

A inferência do visualizador sempre usa CPU f32, inclusive para checkpoints treinados na GPU. Não é necessário compilar com `--features rl-gpu` para assistir a essas políticas.

O argumento opcional `--frames 120` fecha a janela após 120 frames renderizados. Ele serve para uma execução limitada; para inspecionar a interface e o comportamento, omita esse argumento.

## Playback e inspeção

O painel à direita mostra o checkpoint carregado, rollouts concluídos, versão da política, quantidade de parâmetros e arquitetura. As informações do carro selecionado incluem progresso da volta, velocidade, direção, aceleração, recompensa acumulada, tempo, decisões, ticks físicos e motivo de término do episódio.

Use os controles do painel:

- **Pause / Resume** ou `Espaço`: pausar e retomar a simulação.
- **Restart episodes** ou `R`: reconstruir os episódios da pista atual usando as sementes de avaliação salvas.
- **Playback speed**: ajustar a velocidade de 0,1× a 8×, preservando o passo físico e a ordem das decisões.
- **Automatically restart finished episodes**: controlar o reinício após colisão, conclusão de volta ou timeout.
- **Inspected car**: escolher de qual carro inspecionar a telemetria e as ativações.
- **Visual overlays**: exibir sensores reais do ambiente RL, hitbox e informações de depuração da pista.

Ao terminar um episódio, o carro mantém sua posição final e o painel informa **Collision**, **Lap completed** ou **Timeout**. Com reinício automático habilitado, essa apresentação permanece por aproximadamente um segundo de tempo simulado antes do reset. Desabilite a opção para examinar o estado terminal até reiniciar manualmente. A pausa também preserva o estado para inspeção.

Todos os carros executam a mesma política PPO carregada. Eles são episódios do ambiente com sementes próprias, definidas a partir de `evaluation_seed + índice do carro`; não representam indivíduos geneticamente distintos. Reiniciar os episódios repete as sementes iniciais. Para comparar políticas do mesmo treinamento, mantenha pista, número de carros e sementes dos checkpoints iguais.

## Câmera original

O PPO usa o mesmo sistema de câmera do aplicativo original, com os mesmos controles:

| Controle | Ação |
| --- | --- |
| Scroll sobre a pista | Zoom em torno do cursor |
| Arrastar com botão central ou direito | Pan |
| `Q` / `E` | Rotação da câmera |
| **Fit Track** | Ajustar a pista à área de visualização |
| **Reset View** | Restaurar a visão |
| **Free** | Câmera livre |
| **Follow Selected** | Acompanhar o carro escolhido em **Inspected car** |
| `F11` | Alternar fullscreen nativo |

Os controles do mouse atuam na área da pista. Sobre o dashboard, o scroll navega pelo painel. A seção **Display**, ao final do painel, oferece também os modos de janela e fullscreen do aplicativo original.

## Histórico de políticas

Role o dashboard para baixo até **PPO policy history**. O navegador procura recursivamente em `results_ppo`, no diretório configurado em `results_root` e no diretório do checkpoint aberto. Os arquivos são agrupados por diretório de execução.

Clique em um `checkpoint-*.ron` para carregar sua política. **Previous** e **Next** percorrem as políticas válidas da execução atual, ordenadas pelo número de rollouts concluídos registrado no checkpoint. O nome do arquivo não substitui esse metadado. Passe o mouse sobre uma entrada para ver arquitetura, número de parâmetros e versão da política.

**Refresh** atualiza a lista. Para incluir outro diretório, digite seu caminho no campo e clique em **Browse**. A descoberta e a leitura do histórico ocorrem ao abrir ou atualizar as fontes, sem reler os pesos a cada frame de playback.

A troca de checkpoint reinicia os episódios e mantém a pista selecionada. O novo checkpoint é validado antes de substituir o estado atual; se o carregamento falhar, a mensagem aparece no painel e a política anterior permanece disponível. Checkpoints inválidos no navegador ficam desabilitados.

Esse histórico representa políticas por rollout/update. Não há campeão por geração nem lista de melhores indivíduos no PPO. Os formatos e o navegador de checkpoints GA permanecem separados.

## Ativações do Actor-Critic

Marque **Inspect live Actor-Critic network** para abrir a janela neural. Ela pode ser movida, redimensionada e fechada; o painel permite abri-la novamente. A janela começa fechada para deixar a pista visível ao iniciar a apresentação.

A topologia exibida corresponde à implementação PPO:

```text
Entradas normalizadas → Tronco compartilhado ReLU ┬→ Actor: μ[2] → tanh(μ) → ações
                                                └→ Critic: V(s)
log_std[2]: parâmetros globais da distribuição do Actor
```

O Critic recebe o tronco compartilhado. Ele não recebe as saídas do Actor. Os dois parâmetros globais de `log_std` não são neurônios alimentados pelas observações.

As ativações vêm do forward real da política do carro escolhido. Ao abrir ou reiniciar um episódio, mostram a inferência do estado inicial; durante a corrida, mostram a última decisão da política, anterior ao passo físico que ela produziu. Progresso e velocidade podem, portanto, descrever o estado físico após esse passo, enquanto μ e V(s) descrevem a observação que gerou a ação. Pausar mantém esse registro, e escolher outro carro permite consultar o registro dele.

Passe o mouse sobre os neurônios para inspecionar seus valores. A seção **Todas as entradas normalizadas da última decisão** lista todas as entradas na ordem do checkpoint. O preset Extended mantém as 25 entradas reais do ambiente: raios locais, velocidade longitudinal, taxa angular medida e as duas ações anteriores, com a normalização persistida. Não são acrescentados posição global, identidade da pista, progresso ou outras informações privilegiadas à entrada da política.

Acima do gráfico, **Condensada | Expandida** alterna a apresentação imediatamente, preservando o playback e o registro do forward. O modo **Condensada**, usado por padrão, seleciona uma quantidade proporcional ao tamanho de cada camada. Mostra os primeiros e os últimos neurônios, com seus índices reais, e uma elipse vertical `⋮` entre eles quando existem intermediários omitidos. A contagem de omitidos é opcional. Camadas pequenas aparecem completas; camadas maiores mantêm mais nós e uma coluna visualmente maior. Cada ativação e conexão exibida pertence aos neurônios selecionados; a elipse não recebe conexões.

O modo **Expandida** inclui todos os neurônios de todas as camadas. O canvas acompanha o tamanho da arquitetura, sem comprimir as colunas: use o zoom e a rolagem horizontal/vertical para navegar. A exibição de conexões é independente. Habilitadas, todas as conexões reais ficam disponíveis para inspeção; o desenho usa cache e descarta da renderização os elementos fora da área visível, sem truncar a rede. Os controles de resumo e de orçamento de conexões aplicam-se somente ao modo condensado. Em **Detalhes da visualização**, também é possível controlar valores e outros detalhes visuais.

O grupo **Actor** contém quatro círculos: as duas médias `μ steering` e `μ acceleration`, conectadas ao tronco, e os dois parâmetros `log_std steering` e `log_std acceleration`. Estes têm contorno distinto e uma indicação de que são parâmetros globais treináveis, sem conexões com a última camada oculta. O **Critic** possui seu próprio círculo `V(s)`, conectado diretamente ao tronco compartilhado. Os círculos de saída mostram seus valores; os tooltips explicam nome, valor e significado. Isso representa a arquitetura existente, sem transformar `log_std` em saída dependente da observação.

O tronco completo, as duas cabeças e todos os parâmetros continuam sendo calculados, mesmo quando a apresentação resume neurônios, oculta conexões ou muda o zoom.

O controle **Zoom** ajusta apenas o canvas. **Enquadrar** oferece uma visão geral e **100%** restaura a escala natural, com espaço para ler os neurônios. Ao afastar muito o zoom, rótulos e valores que ficariam sobrepostos são ocultados; todos os círculos e conexões permanecem disponíveis. Aproxime novamente ou use os tooltips para inspecionar os valores. **Inspecionar conexão real** permite consultar diretamente qualquer peso por camada, índice de origem e índice de destino, incluindo conexões fora da área visível.

O playback é determinístico: a ação executada é `tanh(μ)`, na ordem `[direção, aceleração]`, como na avaliação PPO. `log_std` é apresentado para inspecionar a distribuição aprendida e não introduz amostragem durante a visualização. V(s) é a estimativa do Critic, distinta da recompensa acumulada do episódio.

## Reutilização e limites da integração

O visualizador compartilha `DisplayPlugin`, renderização da pista, assets e apresentação dos sprites com a interface original. O shell do dashboard e seus controles de câmera/display são funções compartilhadas. O desenho neural reutiliza cores de ativações e pesos, com uma topologia própria para o tronco e as cabeças PPO. O navegador PPO respeita seu próprio formato de persistência.

O ambiente RL continua responsável pela física, sensores, observações, ações, recompensa e términos. Entidades ECS refletem esse estado para apresentação; elas não usam o controlador GA. A aplicação não cria uma sessão invisível de treinamento GA para satisfazer o painel.

Essa integração não altera backpropagation, PPO-Clip, GAE, reward, física, algoritmo genético, formatos dos checkpoints ou política de avaliação. Treinamento e coleta continuam headless. `rl evaluate` continua uma avaliação headless independente da interface e mantém a seleção das pistas reservadas:

```powershell
cargo run --locked -p neuroevolution-racing-app -- rl evaluate --checkpoint results_ppo/ppo-1792340830035034100-29472/checkpoint-000100.ron --episodes 4 --cpu
```

Trocar a pista para inspeção interativa não modifica a suíte nem os critérios usados por esse comando. Resultados de testes e verificações efetivamente realizados são registrados separadamente em [VALIDACAO.md](VALIDACAO.md).
