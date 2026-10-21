# Read-only input processing. Exact receipt and checkpoint index; no discovery.
. ([IO.Path]::Combine($PSScriptRoot, 'ppo_speed_horizon_summary.ps1'))
function Write-PpoEContinuationReport {
    param([Parameter(Mandatory)][string]$RunDirectory,
          [Parameter(Mandatory)][string]$ReportDirectory)
    $run = Get-PpoReportAbsolutePath $RunDirectory 'RunDirectory'
    $report = Get-PpoReportAbsolutePath $ReportDirectory 'ReportDirectory'
    if (-not [IO.Directory]::Exists($report)) { throw "Missing report directory: $report" }
    $required = @('checkpoint_index.csv','periodic_evaluations.csv','metrics.csv','update_metrics.csv','track_metrics.csv','run.log','continuation_state.csv')
    foreach ($name in $required) { if (-not [IO.File]::Exists([IO.Path]::Combine($run,$name))) { throw "Missing continuation artifact: $run / $name" } }
    $state = @(Import-Csv -LiteralPath ([IO.Path]::Combine($run,'continuation_state.csv')))
    if ($state.Count -ne 1) { throw 'Expected one final continuation state.' }
    $state = $state[0]
    $index = @(Import-Csv -LiteralPath ([IO.Path]::Combine($run,'checkpoint_index.csv')))
    $rows = @(Import-Csv -LiteralPath ([IO.Path]::Combine($run,'periodic_evaluations.csv')))
    $tracks = @('monza','red_bull_ring','interlagos','catalunya','silverstone')
    $checkpoints = @()
    foreach ($row in $rows) {
        if (@($index | Where-Object { $_.checkpoint -ceq $row.checkpoint }).Count -ne 1) { throw "Unindexed/ambiguous evaluation checkpoint: $($row.checkpoint)" }
    }
    foreach ($entry in $index) {
        $path = Get-PpoReportAbsolutePath (Get-PpoReportStringProperty $entry 'checkpoint' 'Checkpoint index') 'Checkpoint'
        if (-not [IO.File]::Exists($path)) { throw "Missing indexed checkpoint: $path" }
        $individual = @($rows | Where-Object { $_.checkpoint -ceq $path })
        $evaluated = $individual.Count -gt 0
        $completed = $null; $mean = $null
        if ($evaluated) {
            if ($individual.Count -ne 5) { throw "Incomplete evaluation: $path" }
            $completed = 0; $sum = 0.
            foreach ($track in $tracks) {
                $found = @($individual | Where-Object { $_.track -ceq $track -and $_.rollout -eq $entry.rollout })
                if ($found.Count -ne 1) { throw "Missing/duplicate evaluation $track / $path" }
                $row = $found[0]
                $p = [double][single]::Parse($row.progress,[Globalization.CultureInfo]::InvariantCulture)
                if ([double]::IsNaN($p) -or [double]::IsInfinity($p) -or $p -lt 0 -or $p -gt 1) { throw "Invalid progress: $path" }
                if (([int][bool]::Parse($row.completed)+[int][bool]::Parse($row.collision)+[int][bool]::Parse($row.timeout)) -ne 1) { throw "Invalid outcome: $path" }
                $completed += [int][bool]::Parse($row.completed); $sum += $p
            }
            $mean = $sum/5.
        }
        $checkpoints += [pscustomobject]@{ role=$entry.role; path=$path; rollout=[long]$entry.rollout; adam_step=[long]$entry.adam_step;
            sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash; evaluated=$evaluated;
            completed_training_tracks=$completed; mean_progress=$mean; tracks=$individual }
    }
    $source = @($checkpoints | Where-Object { $_.role -ceq 'source' })
    if ($source.Count -ne 1 -or $source[0].path -cne $state.source_checkpoint) { throw 'Source provenance mismatch.' }
    $best = $checkpoints | Where-Object evaluated | Sort-Object @{e='completed_training_tracks';Descending=$true},@{e='mean_progress';Descending=$true},rollout | Select-Object -First 1
    if ($null -ne $best -and $best.path -cne $state.best_checkpoint) { throw 'Runtime/report best-checkpoint mismatch.' }
    $latest = @($checkpoints | Where-Object { $_.path -ceq $state.latest_checkpoint })
    if ($latest.Count -ne 1) { throw 'Latest checkpoint is not uniquely indexed.' }
    $latest = $latest[0]
    if ($latest.rollout -ne [long]$state.latest_rollout -or $latest.adam_step -ne [long]$state.adam_step -or
        $source[0].rollout -ne [long]$state.source_rollout -or $latest.evaluated -ne [bool]::Parse($state.latest_evaluated)) { throw 'Final counters/evaluation state disagree with indexed artifacts.' }
    $manifest = [ordered]@{ version=1; run_directory=$run; environment_semantics=$state.environment_semantics;
        stop_reason=$state.stop_reason; source_checkpoint=$source[0]; best_checkpoint=$best; latest_checkpoint=$latest;
        criteria=@('completed_training_tracks_descending','equal_weight_mean_progress_descending','earliest_rollout'); checkpoints=$checkpoints;
        historical_branch_401_500_included=$false; checkpoint_rollback_during_training=$false }
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('PPO E continuation: deterministic CPU training-suite evaluation')
    $lines.Add("Run: $run"); $lines.Add("Environment semantics: $($state.environment_semantics)")
    $lines.Add("Source checkpoint: $($source[0].path)")
    $lines.Add("Best checkpoint: $($state.best_checkpoint)"); $lines.Add("Latest checkpoint: $($latest.path)")
    $lines.Add("Stop reason: $($state.stop_reason)")
    foreach ($cp in $checkpoints) {
        $lines.Add("$($cp.role) rollout=$($cp.rollout) adam_step=$($cp.adam_step) evaluated=$($cp.evaluated) completed=$($cp.completed_training_tracks)/5 mean_progress=$($cp.mean_progress) sha256=$($cp.sha256) path=$($cp.path)")
        foreach ($row in $cp.tracks) { $lines.Add("  $($row.track): completed=$($row.completed) collision=$($row.collision) timeout=$($row.timeout) progress=$($row.progress)") }
    }
    $lines.Add('Selection preserves the measured result. It never restores a policy into the live trainer.')
    foreach ($name in @('comparison.txt','comparison_manifest.json')) { if ([IO.File]::Exists([IO.Path]::Combine($report,$name))) { throw "Refusing to overwrite: $report / $name" } }
    [IO.File]::WriteAllLines([IO.Path]::Combine($report,'comparison.txt'),$lines,[Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText([IO.Path]::Combine($report,'comparison_manifest.json'),(ConvertTo-Json -InputObject $manifest -Depth 12),[Text.UTF8Encoding]::new($false))
}
