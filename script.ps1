$base = "results_ppo/update_ablation"
$dest = "ppo_comparativo"

New-Item -ItemType Directory -Force $dest | Out-Null

foreach ($case in @("A_control", "B_lr", "C_lr_entropy")) {
    $run = Get-ChildItem (Join-Path $base $case) -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $run) { throw "Execução não encontrada: $case" }

    $out = Join-Path $dest $case
    New-Item -ItemType Directory -Force $out | Out-Null

    Copy-Item (Join-Path $run.FullName "*metrics.csv") $out

    & ".\target\release\neuroevolution-racing.exe" rl evaluate `
        --checkpoint (Join-Path $run.FullName "checkpoint-001000.ron") `
        --episodes 1 --cpu |
        Tee-Object (Join-Path $out "evaluation.txt")
}

Compress-Archive -Path "$dest\*" `
    -DestinationPath "ppo_comparativo.zip" -Force