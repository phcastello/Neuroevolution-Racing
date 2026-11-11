# Servidor 47 — CPU primeiro

Ambiente principal: Linux x86-64, aproximadamente 12 vCPUs Xeon E5-2470 v2, 23 GiB de RAM e RTX A2000 compartilhada. vCPUs não representam necessariamente núcleos físicos. A CPU é independente de drivers NVIDIA, display e Vulkan.

## Instalação e compilação

Em Ubuntu/Debian (ajustar ao sistema instalado):

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config curl ca-certificates python3 git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh
sh /tmp/rustup.sh -y --profile minimal
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
cd Neuroevolution-Racing
./tools/server47.sh prerequisites
CARGO_BUILD_JOBS=2 ./tools/server47.sh build-cpu
./tools/server47.sh smoke results_ppo/server47-smoke
```

O build usa `--release --locked --no-default-features`; não depende de bibliotecas X11/Wayland, renderizador ou CUDA. O script copia o executável para `target/release/neuroevolution-racing-cpu`, preservando-o quando GPU for compilada depois. Não definir `RUSTFLAGS=-C target-cpu=native` ao transportar binários. O build padrão usa instruções portáveis x86-64; não exige AVX2/AVX-512. Confira `lscpu` na máquina real.

NdArray é serial por padrão: `default-features=false` desabilita SIMD opcional e paralelismo upstream. Para testar paralelismo explicitamente, compile com `CPU_PARALLEL=1 ./tools/server47.sh build-cpu`. Limite usando `RAYON_NUM_THREADS=2 MATMUL_NUM_THREADS=2`; matrixmultiply 0.3.11 suporta efetivamente 1, 2 ou 4 threads, com heurística para matrizes pequenas. Rayon tem seu próprio limite. Esses valores devem acompanhar os resultados. Não há novo scheduler de tensores. `taskset -c <CPUs permitidas>` limita também a afinidade e o paralelismo percebido pelo Bevy; consulte `Cpus_allowed_list` em `/proc/self/status`.

## Benchmark futuro, sem iniciar durante preparação

```bash
# Apenas prepara config e plan, não consulta GPU nem treina
./tools/server47.sh prepare-cpu results_ppo/hardware-cpu-plan
./tools/server47.sh prepare-gpu results_ppo/hardware-gpu-plan
# Conferência sem escrever
./target/release/neuroevolution-racing-cpu research benchmark --backend cpu --output NOVO_DIRETORIO --dry-run

# SOMENTE quando autorizado: diretório NOVO em cada execução
./tools/server47.sh run-cpu results_ppo/hardware-cpu-execution
# Posteriormente, após compilar/verificar GPU:
./tools/server47.sh run-gpu results_ppo/hardware-gpu-execution
```

Os comandos `run-*` executam treinamento real de 100 rollouts do zero, com quatro camadas de 256, 204.805 parâmetros, 32 ambientes ×256 passos, minibatch512, quatro épocas, seeds11/22/33/44/55, reward e PPO do preset E. Sem ACO, antecipação, sensores novos ou avaliação periódica. Só dispositivo e destino mudam entre CPU/GPU. O código calcula o número de parâmetros. Cada rollout tem 8.192 transições; 100 rollouts não significam 100 episódios.

`config.ron` e `plan.ron` ficam no diretório preparado. A execução cria `run-path.txt`, `launcher.log` e um run exclusivo. Persistência: apenas checkpoint final no orçamento100, sem buffers de rollout; métricas completas. Telemetria inclui coleta, observação, upload/inferência/download sincronizados, física, preparação GAE, update completo (backward/Adam), persistência e total. O tempo de update agrega forward da loss, backward, Adam e diagnósticos; não promete decomposição do backward isolado. `/proc` fornece RSS e pico RAM, quando disponível.

## Monitoramento, interrupção e retomada

```bash
free -h
df -h .
ps -o pid,etime,pcpu,pmem,rss,args -p PID
# Diretório exato fornecido no recibo
cat results_ppo/hardware-cpu-execution/run-path.txt
tail -f DIRETORIO_EXATO/performance.csv
```

Ctrl-C no processo `rl train` solicita parada na próxima fronteira de rollout. Ele termina coleta/update em andamento e salva checkpoint final; aguarde. Se houver um launcher, envie SIGINT ao PID do processo de treinamento (`kill -INT PID`), mantendo o launcher vivo para recolher o status. SIGKILL/queda de energia não podem garantir checkpoint da etapa atual; checkpoints já sincronizados permanecem utilizáveis. Não interrompa durante escrita com `kill -9` se puder evitar.

Para continuar, copie a configuração efetiva, aumente somente `total_rollouts`, selecione destino novo e use `rl train --config ... --resume CHECKPOINT_EXATO.ron`. Arquitetura, observações, rewards, seeds e currículo são verificados. Adam/RNGs/currículo são restaurados, ambientes reiniciados. Um benchmark retomado deve ser identificado separadamente; a comparação de referência deve começar do zero.

## GPU NVIDIA opcional

Somente quando for usar GPU:

```bash
# Em Ubuntu, conforme versão/driver aprovado pelo administrador:
sudo apt-get install -y libvulkan1 vulkan-tools
# Instalar driver NVIDIA compatível com o kernel pelo procedimento do servidor.
vulkaninfo --summary
nvidia-smi
./tools/server47.sh build-gpu
# Não executar run-gpu antes da autorização do experimento.
```

WGPU usa Vulkan em Linux e não exige CUDA. Ao selecionar `Gpu(adapter: 0)`, o programa enumera apenas adaptadores de hardware, registra inventário/driver/backend/índice e falha claramente se não houver adaptador Vulkan. Não existe fallback para CPU. CPU não faz essas consultas. Compilar GPU não inicializa dispositivos. Valide permissões de `/dev/dri`, ICD Vulkan NVIDIA e disponibilidade real da A2000 no servidor.

## Recolher e comparar

```bash
# Pode recolher execuções feitas em datas distintas
rsync -av servidor47:/CAMINHO/results_ppo/hardware-cpu-execution/ ./cpu-execution/
rsync -av servidor47:/CAMINHO/results_ppo/hardware-gpu-execution/ ./gpu-execution/
./tools/server47.sh compare results_ppo/comparison DIRETORIO_RUN_CPU DIRETORIO_RUN_GPU
```

Conserve os diretórios completos e recibos, não apenas screenshots. O relatório inclui JSON, CSV e Markdown; registros ausentes não são preenchidos. Compare mesmo commit, configuração científica, seeds, número de transições e opções de paralelismo. Registre serviços concorrentes e disponibilidade das vCPUs/GPU. Média, mediana, P90/P95, throughput global e RAM são medidos pelo mesmo treinador. O benchmark de inferência `rl benchmark` não substitui este protocolo.

Ainda precisam ser verificados no Servidor47: distribuição Linux real, instruções/afinidade expostas pela VM, espaço para compilação, RAM durante rollout32, desempenho serial/paralelo, driver/ICD NVIDIA e execução visual se desejada. Nenhum resultado CPU×GPU foi produzido nesta implementação.
