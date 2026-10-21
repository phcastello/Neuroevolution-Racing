# Shared by the runner and synthetic fixture tests. Never discovers run directories.
function Get-PpoReportStringProperty {
    param([object] $InputObject, [string] $Name, [string] $Context)
    if ($null -eq $InputObject) { throw "${Context}: expected an object with property '$Name', found null." }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if (-not $InputObject.Contains($Name)) { throw "${Context}: missing property '$Name'." }
        $value = $InputObject[$Name]
    } else {
        $property = $InputObject.PSObject.Properties[$Name]
        if ($null -eq $property) {
            throw "${Context}: missing property '$Name' (available: $($InputObject.PSObject.Properties.Name -join ', '))."
        }
        $value = $property.Value
    }
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        throw "${Context}: property '$Name' must be a non-empty string."
    }
    return $value
}

function Get-PpoReportAbsolutePath {
    param([object] $Value, [string] $Context)
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace($Value)) {
        throw "${Context}: path must be a non-empty string."
    }
    # Receipts and evaluation CSVs use Rust canonicalize(), including \\?\ on
    # Windows. Keep that spelling for provenance; Join-Path's PS 5.1 provider
    # rejects it with a null 'drive'. Use System.IO for all report paths instead.
    try {
        if (-not [IO.Path]::IsPathRooted($Value) -or
            ($env:OS -eq 'Windows_NT' -and $Value -notmatch '^(?:[A-Za-z]:[\\/]|\\\\)')) {
            throw 'An absolute filesystem path is required.'
        }
        return [IO.Path]::GetFullPath($Value)
    } catch { throw "${Context}: invalid path '$Value': $($_.Exception.Message)" }
}

function Write-PpoSpeedHorizonSummary {
    param([Parameter(Mandatory)] [AllowNull()] [AllowEmptyCollection()] [object[]] $Runs,
          [Parameter(Mandatory)] [AllowNull()] [AllowEmptyString()] [string] $ReportDirectory)
    if ($null -eq $Runs -or $Runs.Count -eq 0) { throw 'Runs: expected at least one run object with Experiment and Path properties.' }
    $reportPath = Get-PpoReportAbsolutePath -Value $ReportDirectory -Context 'ReportDirectory'
    if (-not [IO.Directory]::Exists($reportPath)) { throw "ReportDirectory: directory does not exist: $reportPath" }
    $summaryPath = [IO.Path]::Combine($reportPath, 'comparison.txt')
    $manifestPath = [IO.Path]::Combine($reportPath, 'comparison_manifest.json')
    foreach ($path in @($summaryPath, $manifestPath)) {
        if ([IO.File]::Exists($path) -or [IO.Directory]::Exists($path)) { throw "Refusing to overwrite report: $path" }
    }
    $tracks = @('monza', 'red_bull_ring', 'interlagos', 'catalunya', 'silverstone')
    $checkpoints = @(100, 200, 300, 400, 500)
    $scores = @()
    $best = @()
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('PPO speed / temporal horizon ablation: deterministic training-track evaluation')
    $lines.Add('Ranking: completed tracks, then equal-weight mean normalized progress. Reward is diagnostic only.')
    $lines.Add('One fixed-start attempt per track and checkpoint; these are not independent statistical samples.')
    $runIndex = 0
    foreach ($run in $Runs) {
        $context = "Runs[$runIndex]"
        $experiment = Get-PpoReportStringProperty -InputObject $run -Name 'Experiment' -Context $context
        $sourcePath = Get-PpoReportStringProperty -InputObject $run -Name 'Path' -Context "$context ($experiment)"
        $runPath = Get-PpoReportAbsolutePath -Value $sourcePath -Context "$context ($experiment).Path"
        if (-not [IO.Directory]::Exists($runPath)) { throw "${experiment}: run directory does not exist: $runPath" }
        $csvPath = [IO.Path]::Combine($runPath, 'periodic_evaluations.csv')
        if (-not [IO.File]::Exists($csvPath)) { throw "${experiment}: missing evaluation file: $csvPath" }
        $rows = @(Import-Csv -LiteralPath $csvPath -ErrorAction Stop)
        if ($rows.Count -ne 25) { throw "$($run.Experiment): expected exactly 25 evaluation rows, found $($rows.Count)." }
        $requiredColumns = @('experiment', 'rollout', 'checkpoint', 'track', 'completed', 'collision', 'timeout',
            'progress', 'reward', 'decisions', 'physical_ticks', 'simulated_seconds', 'evaluation_wall_seconds')
        foreach ($column in $requiredColumns) {
            if ($null -eq $rows[0].PSObject.Properties[$column]) { throw "${experiment}: missing CSV column '$column' in $csvPath" }
        }
        $lines.Add('')
        $lines.Add("$($run.Experiment): $($run.Path)")
        $local = @()
        foreach ($rollout in $checkpoints) {
            $group = @($rows | Where-Object { [int]$_.rollout -eq $rollout })
            if ($group.Count -ne 5) { throw "$($run.Experiment) rollout ${rollout}: expected five tracks." }
            $checkpoint = [IO.Path]::Combine($runPath, ('checkpoint-{0:D6}.ron' -f $rollout))
            if (-not [IO.File]::Exists($checkpoint)) { throw "${experiment}: missing checkpoint: $checkpoint" }
            $completed = 0
            $progress = 0.0
            foreach ($track in $tracks) {
                $match = @($group | Where-Object { $_.track -eq $track })
                if ($match.Count -ne 1) { throw "$($run.Experiment) rollout ${rollout}: missing or duplicate track $track." }
                $row = $match[0]
                $recordedPath = Get-PpoReportStringProperty -InputObject $row -Name 'checkpoint' -Context "$experiment rollout $rollout / $track"
                $recordedCheckpoint = Get-PpoReportAbsolutePath -Value $recordedPath -Context "$experiment rollout $rollout / $track checkpoint"
                if ($row.experiment -cne $run.Experiment -or $recordedCheckpoint -cne $checkpoint) {
                    throw "Evaluation provenance mismatch for $($run.Experiment) rollout $rollout / $track."
                }
                $p = [double]::Parse($row.progress, [Globalization.CultureInfo]::InvariantCulture)
                if ([double]::IsNaN($p) -or [double]::IsInfinity($p) -or $p -lt 0 -or $p -gt 1) { throw 'Invalid normalized progress.' }
                $done = [bool]::Parse($row.completed)
                $collision = [bool]::Parse($row.collision)
                $timeout = [bool]::Parse($row.timeout)
                if (([int]$done + [int]$collision + [int]$timeout) -ne 1) { throw 'Invalid evaluation outcome.' }
                $completed += [int]$done
                $progress += $p
            }
            $score = [pscustomobject]@{ Experiment = $run.Experiment; Rollout = $rollout; Checkpoint = $checkpoint
                CompletedTracks = $completed; MeanProgress = $progress / 5.0 }
            $local += $score
            $scores += $score
            $highlight = if ($rollout -ge 300) { ' [LATE CHECKPOINT]' } else { '' }
            $lines.Add(('  rollout {0}: completed={1}/5 mean_progress={2:F6}{3}' -f $rollout, $completed, $score.MeanProgress, $highlight))
            foreach ($track in $tracks) {
                $row = $group | Where-Object { $_.track -eq $track }
                $status = if ([bool]::Parse($row.completed)) { 'completed' } elseif ([bool]::Parse($row.collision)) { 'collision' } else { 'timeout' }
                $lines.Add("    ${track}: $status progress=$($row.progress) reward=$($row.reward) decisions=$($row.decisions) ticks=$($row.physical_ticks) simulated_seconds=$($row.simulated_seconds) evaluation_wall_seconds=$($row.evaluation_wall_seconds)")
            }
        }
        # Exact ties use the earliest checkpoint only to make the manifest reproducible.
        $winner = $local | Sort-Object @{Expression='CompletedTracks';Descending=$true}, @{Expression='MeanProgress';Descending=$true}, Rollout | Select-Object -First 1
        $best += $winner
        $lines.Add("  Best recorded checkpoint (ties: earliest): rollout=$($winner.Rollout) completed=$($winner.CompletedTracks)/5 mean_progress=$($winner.MeanProgress) path=$($winner.Checkpoint)")
        $runIndex++
    }
    $lines.Add('')
    $lines.Add('D / E late evolution (D then E; inspect individual tracks above):')
    foreach ($rollout in @(300, 400, 500)) {
        foreach ($score in @($scores | Where-Object { $_.Rollout -eq $rollout })) {
            $lines.Add(('  {0} @ {1}: {2}/5, mean_progress={3:F6}' -f $score.Experiment, $rollout, $score.CompletedTracks, $score.MeanProgress))
        }
    }
    $lines.Add('Small differences or isolated advantages do not establish superiority. No configuration is selected for longer training.')
    $manifest = [ordered]@{ Version=1; Runs=$Runs; Criteria=@('completed_tracks_descending', 'equal_weight_mean_progress_descending'); TieBreak='earliest_rollout'; BestCheckpoints=$best; Checkpoints=$scores; LongerTrainingSelected=$false }
    [IO.File]::WriteAllLines($summaryPath, $lines, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    Write-Host ($lines -join [Environment]::NewLine)
    Write-Host "Comparison: $summaryPath"
    Write-Host "Best-checkpoint manifest: $manifestPath"
}
