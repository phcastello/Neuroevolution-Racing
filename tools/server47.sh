#!/usr/bin/env bash
# CPU commands deliberately never invoke vulkaninfo/nvidia-smi or a graphics executable.
set -euo pipefail
cd "$(dirname "$0")/.."
usage() {
cat <<'HELP'
Usage: tools/server47.sh COMMAND [arguments]
  prerequisites                 CPU prerequisites, memory, CPU flags (no GPU probes)
  build-cpu                     Portable release CPU executable, --locked, no graphics
  build-gpu                     Optional portable headless WGPU/Vulkan executable
  smoke [new-directory]         1 rollout CPU with small network, backward and Adam
  prepare-cpu NEW-DIRECTORY      Prepare 100 rollouts, no execution
  prepare-gpu NEW-DIRECTORY      Prepare 100 rollouts, no execution/device probe
  run-cpu NEW-DIRECTORY          EXPLICIT full 100 rollout CPU benchmark
  run-gpu NEW-DIRECTORY          EXPLICIT full 100 rollout GPU benchmark
  compare NEW-OUTPUT RUN...      Compare existing run directories; no new training
  help                          This help
CPU executable: target/release/neuroevolution-racing-cpu
GPU executable: target/release/neuroevolution-racing-gpu
Optional parallel CPU build: CPU_PARALLEL=1; RAYON_NUM_THREADS=2 MATMUL_NUM_THREADS=2
Use taskset -c <allowed CPUs> to limit effective CPU availability when necessary.
HELP
}
bin_cpu=${RACING_CPU_BIN:-target/release/neuroevolution-racing-cpu}
bin_gpu=${RACING_GPU_BIN:-target/release/neuroevolution-racing-gpu}
case "${1:-help}" in
help|--help|-h) usage ;;
prerequisites)
    command -v cargo; cargo --version; rustc --version
    command -v cc; command -v pkg-config
    uname -a; getconf _NPROCESSORS_ONLN; free -h; df -h .
    lscpu | sed -n '1,32p'
    ;;
build-cpu)
    features=(); if [[ "${CPU_PARALLEL:-0}" == 1 ]]; then features=(--features cpu-parallel); fi
    cargo build --release --locked -p neuroevolution-racing-app --no-default-features "${features[@]}"
    cp target/release/neuroevolution-racing "$bin_cpu"
    ;;
build-gpu)
    cargo build --release --locked -p neuroevolution-racing-app --no-default-features --features rl-gpu
    cp target/release/neuroevolution-racing "$bin_gpu"
    ;;
smoke)
    output=${2:-"results_ppo/smoke-$(date -u +%Y%m%dT%H%M%S)-$$"}
    mkdir -p "$(dirname "$output")"
    mkdir "$output"
    # Historical small smoke preset; only output path overridden in generated effective config.
    python3 - "$output" <<'PY'
import json,pathlib,re,sys
out=pathlib.Path(sys.argv[1]); text=pathlib.Path('configs/rl/smoke.ron').read_text()
text=re.sub(r'device:\s*Gpu\(adapter:\s*\d+\)', 'device: Cpu',text)
text=re.sub(r'total_rollouts:\s*\d+', 'total_rollouts: 1',text)
text=re.sub(r'results_root:\s*"[^"]*"','results_root: '+json.dumps(str(out/'runs')),text)
(out/'config.ron').write_text(text)
PY
    "$bin_cpu" rl train --config "$output/config.ron" --save-every 100 --save-first 0 --save-last 0 --no-save-rollouts --run-path-file "$output/run-path.txt"
    ;;
prepare-cpu|prepare-gpu|run-cpu|run-gpu)
    [[ $# == 2 ]] || { usage >&2; exit 2; }
    backend=cpu; executable=$bin_cpu; action=()
    [[ $1 == *gpu ]] && backend=gpu
    # Preparing GPU on a CPU-only binary is intentional: no device is touched.
    [[ $1 == run-gpu ]] && executable=$bin_gpu
    [[ $1 == run-* ]] && action=(--execute)
    "$executable" research benchmark --backend "$backend" --output "$2" "${action[@]}"
    ;;
compare)
    [[ $# -ge 3 ]] || { usage >&2; exit 2; }
    output=$2; shift 2
    "$bin_cpu" research compare --output "$output" "$@"
    ;;
*) usage >&2; exit 2 ;;
esac
