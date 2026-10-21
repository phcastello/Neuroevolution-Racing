[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'ppo_speed_horizon_summary.ps1')
Push-Location -LiteralPath $repoRoot
try {
    & cargo build --release --locked -p neuroevolution-racing-app --features rl-gpu
    if ($LASTEXITCODE -ne 0) { throw "Release GPU build failed (exit $LASTEXITCODE). No experiment was started." }
    $executable = Join-Path $repoRoot 'target/release/neuroevolution-racing.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Release executable not found: $executable" }
    $batch = 'batch-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ([Guid]::NewGuid().ToString('N'))
    $reportDirectory = Join-Path $repoRoot "results_ppo/speed_horizon_ablation/reports/$batch"
    $null = New-Item -ItemType Directory -Path $reportDirectory
    $experiments = @(
        @{ Experiment='D_speed'; Config='configs/rl/ppo_speed_d.ron' },
        @{ Experiment='E_speed_horizon'; Config='configs/rl/ppo_speed_horizon_e.ron' }
    )
    $runs = @()
    foreach ($experiment in $experiments) {
        $receipt = Join-Path $reportDirectory "$($experiment.Experiment)-run-path.txt"
        Write-Host "Starting $($experiment.Experiment) from scratch: $($experiment.Config)"
        # One uninterrupted training process per experiment. Evaluation is opt-in
        # at rollout boundaries; no resume, checkpoint discovery or console filtering.
        & $executable rl train --config $experiment.Config `
            --save-every 100 --save-first 0 --save-last 0 --no-save-rollouts `
            --evaluate-every 100 --experiment $experiment.Experiment --run-path-file $receipt
        $exitCode = $LASTEXITCODE
        if (Test-Path -LiteralPath $receipt) {
            $path = [IO.File]::ReadAllText($receipt).Trim()
            $runs += [pscustomobject]@{ Experiment=$experiment.Experiment; Config=$experiment.Config; Path=$path; ExitCode=$exitCode }
            # This batch owns the registry. It survives a later failed experiment.
            $runs | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportDirectory 'runs.json') -Encoding UTF8
        }
        if ($exitCode -ne 0) { throw "$($experiment.Experiment) failed (exit $exitCode). Remaining experiments were not started. Receipts: $reportDirectory" }
        if (-not (Test-Path -LiteralPath $receipt)) { throw "$($experiment.Experiment) returned no exact run receipt. Reports: $reportDirectory" }
    }
    Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory $reportDirectory
}
finally { Pop-Location }
