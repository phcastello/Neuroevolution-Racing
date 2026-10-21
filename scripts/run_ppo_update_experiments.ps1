[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

Push-Location -LiteralPath $repoRoot
try {
    # Compile once; do not silently use a stale binary after a failed build.
    & cargo build --release --locked -p neuroevolution-racing-app --features rl-gpu
    if ($LASTEXITCODE -ne 0) {
        throw "Release GPU build failed (exit $LASTEXITCODE). No experiment was started."
    }

    $executable = Join-Path $repoRoot 'target/release/neuroevolution-racing.exe'
    $experiments = @(
        @{ Name = 'A - control'; Config = 'configs/rl/ppo_update_a_control.ron' },
        @{ Name = 'B - lower LR'; Config = 'configs/rl/ppo_update_b_lr.ron' },
        @{ Name = 'C - lower LR + latent entropy'; Config = 'configs/rl/ppo_update_c_lr_entropy.ron' }
    )

    foreach ($experiment in $experiments) {
        Write-Host "Starting $($experiment.Name) from scratch: $($experiment.Config)"
        # Each preset has its own results_root; run_directory creates a unique child.
        # No --resume: weights, Adam, RNGs and rollout counters start from the preset.
        & $executable rl train --config $experiment.Config `
            --save-every 100 --save-first 0 --save-last 0 --no-save-rollouts
        if ($LASTEXITCODE -ne 0) {
            throw "$($experiment.Name) failed (exit $LASTEXITCODE). Remaining experiments were not started."
        }
    }
}
finally {
    Pop-Location
}
