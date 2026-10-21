# Synthetic fixtures only: no build, training command or policy evaluation.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ppo_speed_horizon_summary.ps1')
function Assert-True($condition, $message) { if (-not $condition) { throw $message } }
function Assert-Fails([scriptblock]$action, [string]$messagePattern = '*') {
    $failed = $false
    try { & $action } catch {
        $failed = $true
        Assert-True ($_.Exception.Message -like $messagePattern) "Unexpected rejection: $($_.Exception.Message)"
    }
    Assert-True $failed 'Expected fixture rejection.'
}
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('ppo-speed-horizon-test-' + [Guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $fixtureRoot
try {
    $tracks = @('monza','red_bull_ring','interlagos','catalunya','silverstone')
    $runs = @()
    foreach ($experiment in @('D_speed','E_speed_horizon')) {
        $run = Join-Path $fixtureRoot "$experiment/ppo-synthetic"
        $null = New-Item -ItemType Directory -Path $run
        $rows = @()
        foreach ($rollout in @(100,200,300,400,500)) {
            $checkpoint = Join-Path $run ('checkpoint-{0:D6}.ron' -f $rollout)
            [IO.File]::WriteAllText($checkpoint, 'synthetic placeholder - never evaluated')
            foreach ($track in $tracks) {
                $done = ($rollout -in @(200,300,400)) -and $track -eq 'monza'
                $rows += [pscustomobject]@{
                    experiment=$experiment; rollout=$rollout; checkpoint=$checkpoint; track=$track
                    completed=$done; collision=(-not $done); timeout=$false
                    progress=$(if ($done) { 1.0 } elseif ($rollout -eq 500) { 0.95 } elseif ($rollout -eq 200) { 0.1 } else { 0.2 })
                    reward=$(if ($rollout -eq 500) { 9999 } else { 1 }); decisions=2; physical_ticks=2
                    simulated_seconds=0.033333; evaluation_wall_seconds=0.001
                }
            }
        }
        $rows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $run 'periodic_evaluations.csv')
        $runs += [pscustomobject]@{ Experiment=$experiment; Path=$run }
    }
    $null = New-Item -ItemType Directory -Path (Join-Path $fixtureRoot 'D_speed/ppo-unrelated-newest')
    $report = Join-Path $fixtureRoot 'report'
    $null = New-Item -ItemType Directory -Path $report
    Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory $report
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $report 'comparison_manifest.json') | ConvertFrom-Json
    Assert-True ($manifest.Checkpoints.Count -eq 10) 'Missing checkpoint scores.'
    Assert-True ($manifest.BestCheckpoints.Count -eq 2) 'Missing best checkpoints.'
    Assert-True (@($manifest.BestCheckpoints | Where-Object { $_.Rollout -ne 300 }).Count -eq 0) 'Incorrect ranking or tie break.'
    Assert-True (-not $manifest.LongerTrainingSelected) 'Must never select longer training.'
    Assert-True ((Get-ChildItem -LiteralPath $fixtureRoot -Filter 'checkpoint-*.ron' -Recurse).Count -eq 10) 'Original checkpoints changed.'
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory $report }

    # The production runner and ConvertFrom-Json both supply Experiment/Path
    # objects. Rust's Windows receipts AND CSV checkpoint paths have a \\?\ prefix.
    # Exercise that exact contract, including an extended output directory, on
    # Windows PowerShell 5.1 without invoking any application executable.
    if ($env:OS -eq 'Windows_NT') {
        $extendedRuns = @()
        $originalCsvs = @{}
        foreach ($run in $runs) {
            $csvPath = [IO.Path]::Combine($run.Path, 'periodic_evaluations.csv')
            $originalCsvs[$csvPath] = [IO.File]::ReadAllText($csvPath)
            $extendedPath = '\\?\' + $run.Path
            $rows = @(Import-Csv -LiteralPath $csvPath)
            foreach ($row in $rows) { $row.checkpoint = '\\?\' + $row.checkpoint }
            $rows | Export-Csv -NoTypeInformation -LiteralPath $csvPath
            $extendedRuns += [pscustomobject]@{ Experiment=$run.Experiment; Config='fixture.ron'; Path=$extendedPath; ExitCode=0 }
        }
        try {
            $roundTrip = ConvertFrom-Json -InputObject (ConvertTo-Json -InputObject $extendedRuns -Depth 5)
            $extendedReport = Join-Path $fixtureRoot 'extended-report'
            $null = New-Item -ItemType Directory -Path $extendedReport
            Write-PpoSpeedHorizonSummary -Runs $roundTrip -ReportDirectory ('\\?\' + $extendedReport)
            $extendedManifest = Get-Content -Raw -LiteralPath (Join-Path $extendedReport 'comparison_manifest.json') | ConvertFrom-Json
            Assert-True ($extendedManifest.Checkpoints.Count -eq 10) 'Extended paths lost checkpoint scores.'
            Assert-True ($extendedManifest.Runs[0].Path -ceq $extendedRuns[0].Path) 'Receipt spelling was not preserved.'
            Assert-True ($extendedManifest.Checkpoints[0].Checkpoint.StartsWith('\\?\')) 'Checkpoint provenance lost extended prefix.'
        } finally {
            foreach ($csvPath in $originalCsvs.Keys) { [IO.File]::WriteAllText($csvPath, $originalCsvs[$csvPath]) }
        }
    }

    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @() -ReportDirectory $fixtureRoot } '*expected at least one run object*'
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($null) -ReportDirectory $fixtureRoot } '*found null*'
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @([pscustomobject]@{ Path=$runs[0].Path }) -ReportDirectory $fixtureRoot } "*missing property 'Experiment'*"
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @([pscustomobject]@{ Experiment='D_speed'; FullName=$runs[0].Path }) -ReportDirectory $fixtureRoot } "*missing property 'Path'*"
    foreach ($badPath in @($null, '', '  ', @('a','b'))) {
        $invalidRun = [pscustomobject]@{ Experiment='D_speed'; Path=$badPath }
        Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } "*property 'Path' must be a non-empty string*"
    }
    $invalidRun = [pscustomobject]@{ Experiment='D_speed'; Path='relative/run' }
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } '*absolute filesystem path*'
    $invalidRun.Path = Join-Path $fixtureRoot 'missing-run'
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } '*run directory does not exist*'
    $invalidRun.Path = Join-Path $fixtureRoot 'incomplete-run'
    $null = New-Item -ItemType Directory -Path $invalidRun.Path
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } '*missing evaluation file*'
    $incompleteCsv = Join-Path $invalidRun.Path 'periodic_evaluations.csv'
    Copy-Item -LiteralPath (Join-Path $runs[0].Path 'periodic_evaluations.csv') -Destination $incompleteCsv
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } '*missing checkpoint*'
    $badColumns = @(Import-Csv -LiteralPath $incompleteCsv | Select-Object * -ExcludeProperty checkpoint)
    $badColumns | Export-Csv -NoTypeInformation -LiteralPath $incompleteCsv
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs @($invalidRun) -ReportDirectory $fixtureRoot } "*missing CSV column 'checkpoint'*"
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory '  ' } '*ReportDirectory: path must be a non-empty string*'
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory (Join-Path $fixtureRoot 'missing-report') } '*ReportDirectory: directory does not exist*'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $fixtureRoot 'comparison.txt'))) 'Rejected input wrote a report.'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $fixtureRoot 'comparison_manifest.json'))) 'Rejected input wrote a manifest.'
    $csv = Join-Path $runs[1].Path 'periodic_evaluations.csv'
    $original = [IO.File]::ReadAllText($csv)
    $bad = @(Import-Csv -LiteralPath $csv)
    $bad[0].checkpoint = (Join-Path $runs[0].Path 'checkpoint-000100.ron')
    $bad | Export-Csv -NoTypeInformation -LiteralPath $csv
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory $fixtureRoot }
    [IO.File]::WriteAllText($csv, $original)
    $bad = @(Import-Csv -LiteralPath $csv)
    $bad[0].track = $bad[1].track
    $bad | Export-Csv -NoTypeInformation -LiteralPath $csv
    Assert-Fails { Write-PpoSpeedHorizonSummary -Runs $runs -ReportDirectory $fixtureRoot }
    $runner = Join-Path $PSScriptRoot 'run_ppo_speed_horizon_experiments.ps1'
    $tokens = $null; $parseErrors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile($runner, [ref]$tokens, [ref]$parseErrors)
    Assert-True ($parseErrors.Count -eq 0) 'Runner syntax errors.'
    $source = [IO.File]::ReadAllText($runner)
    Assert-True (([regex]::Matches($source, 'cargo build --release --locked')).Count -eq 1) 'Must build once.'
    Assert-True ($source.Contains('--features rl-gpu')) 'Missing GPU feature.'
    Assert-True (-not $source.Contains('--resume')) 'Must start from scratch.'
    Assert-True ($source.IndexOf("Experiment='D_speed'") -lt $source.IndexOf("Experiment='E_speed_horizon'")) 'Wrong sequence.'
    Assert-True ($source.Contains('--run-path-file $receipt')) 'Missing exact receipt.'
    Assert-True ($source.Contains('--save-every 100 --save-first 0 --save-last 0 --no-save-rollouts')) 'Wrong persistence.'
    Assert-True ($source.Contains('--evaluate-every 100')) 'Missing periodic evaluation.'
    Assert-True ($source.Contains('if ($exitCode -ne 0)')) 'Missing exit guard.'
    Write-Host "PASS (PowerShell $($PSVersionTable.PSVersion), $($PSVersionTable.PSEdition)): synthetic comparison, extended Windows paths, explicit input errors, provenance, preservation, ranking and runner checks."
}
finally {
    $resolved = (Resolve-Path -LiteralPath $fixtureRoot).Path
    $temp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($temp, [StringComparison]::OrdinalIgnoreCase) -or
        -not ([IO.Path]::GetFileName($resolved)).StartsWith('ppo-speed-horizon-test-')) { throw 'Unsafe fixture cleanup path.' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
