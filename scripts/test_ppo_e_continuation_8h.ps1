# Synthetic processes and data only. Never calls the continuation runner.
$ErrorActionPreference='Stop'
$repoRoot=[IO.Path]::GetFullPath([IO.Path]::Combine($PSScriptRoot,'..'))
$selection=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText([IO.Path]::Combine($repoRoot,'configs\rl\ppo_e_continue_8h_source.json')))
$assembly=[IO.Path]::Combine($repoRoot,$selection.supervisor_assembly)
if ((Get-FileHash -LiteralPath $assembly -Algorithm SHA256).Hash -cne $selection.supervisor_assembly_sha256) { throw 'Prepared supervisor hash mismatch.' }
Add-Type -Path $assembly
. ([IO.Path]::Combine($PSScriptRoot,'ppo_e_continuation_report.ps1'))
function Assert($condition,[string]$message) { if (-not $condition) { throw $message } }
$root=[IO.Path]::Combine([IO.Path]::GetTempPath(),('ppo-e-session-test-'+[Guid]::NewGuid().ToString('N')))
$null=[IO.Directory]::CreateDirectory($root)
$jobA=$null; $jobB=$null
try {
    foreach ($name in @('run_ppo_e_continuation_8h.ps1','ppo_e_continuation_report.ps1')) {
        $tokens=$null; $errors=$null
        $null=[Management.Automation.Language.Parser]::ParseFile([IO.Path]::Combine($PSScriptRoot,$name),[ref]$tokens,[ref]$errors)
        Assert ($errors.Count -eq 0) "Syntax errors in $name"
    }
    # Invoke just the runner's real subprocess helper via its AST, never its body.
    $tokens=$null; $errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile([IO.Path]::Combine($PSScriptRoot,'run_ppo_e_continuation_8h.ps1'),[ref]$tokens,[ref]$errors)
    $helper=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-SessionProcess'},$true)
    Invoke-Expression $helper.Extent.Text
    $SessionDirectory=$root
    $powershell=[IO.Path]::Combine($env:SystemRoot,'System32\WindowsPowerShell\v1.0\powershell.exe')
    $code=Invoke-SessionProcess $powershell @('-NoProfile','-Command','[Console]::Error.WriteLine("synthetic stderr"); exit 7') 'synthetic-exit'
    Assert ($code -eq 7) 'Subprocess exit code was not propagated.'
    Assert ([IO.File]::ReadAllText([IO.Path]::Combine($root,'synthetic-exit.stderr.log')).Contains('synthetic stderr')) 'Subprocess stderr was not captured.'
    # Exercise the actual worker's setup and expired-budget branch with a tiny
    # inert source file. No build, checkpoint load/evaluation or training can start.
    $expiredDirectory=[IO.Path]::Combine($root,'expired-session')
    $null=[IO.Directory]::CreateDirectory($expiredDirectory)
    $inertSource=[IO.Path]::Combine($root,'inert-source.ron')
    [IO.File]::WriteAllText($inertSource,'inert synthetic input - never loaded')
    $pastOrigin=[Diagnostics.Stopwatch]::GetTimestamp()-27001*[Diagnostics.Stopwatch]::Frequency
    $workerCommand=(@($powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',
        [IO.Path]::Combine($PSScriptRoot,'run_ppo_e_continuation_8h.ps1'),'-Worker','-SessionDirectory',$expiredDirectory,
        '-SessionStartTicks',[string]$pastOrigin,'-ResumeCheckpoint',$inertSource) | ForEach-Object { [PpoSessionJob]::Quote($_) }) -join ' '
    $workerStart=[Diagnostics.Stopwatch]::GetTimestamp()
    $jobA=[PpoSessionJob]::new($workerCommand,$repoRoot,$workerStart,10.)
    while(-not $jobA.Wait(100)){}
    Assert (-not $jobA.Forced -and $jobA.ExitCode -eq 0) 'Expired worker did not stop normally.'
    $jobA.Dispose(); $jobA=$null
    $workerState=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText([IO.Path]::Combine($expiredDirectory,'worker_status.json')))
    Assert ($workerState.stop_reason -ceq 'wall_time_budget') 'Expired worker lost the planned stop reason.'
    Assert (-not [IO.File]::Exists([IO.Path]::Combine($expiredDirectory,'build.stdout.log'))) 'Expired worker started compilation.'
    Assert (-not [IO.File]::Exists([IO.Path]::Combine($expiredDirectory,'run-path.txt'))) 'Expired worker started training.'
    $powershell=[IO.Path]::Combine($env:SystemRoot,'System32\WindowsPowerShell\v1.0\powershell.exe')
    $synthetic=[IO.Path]::Combine($root,'synthetic.ps1')
    $childReceipt=[IO.Path]::Combine($root,'synthetic-child.txt')
    [IO.File]::WriteAllText($synthetic,@'
param([string]$Receipt)
$child=Start-Process -FilePath ([IO.Path]::Combine($env:SystemRoot,'System32\WindowsPowerShell\v1.0\powershell.exe')) -WindowStyle Hidden -PassThru -ArgumentList '-NoProfile -Command "Start-Sleep -Seconds 30"'
$child.Id | Set-Content -LiteralPath $Receipt
Start-Sleep -Seconds 30
'@)
    $command=(@($powershell,'-NoProfile','-File',$synthetic,'-Receipt',$childReceipt) | ForEach-Object { [PpoSessionJob]::Quote($_) }) -join ' '
    # One separate owned synthetic job must survive termination of A.
    $other=(@($powershell,'-NoProfile','-Command','Start-Sleep -Seconds 30') | ForEach-Object { [PpoSessionJob]::Quote($_) }) -join ' '
    $start=[Diagnostics.Stopwatch]::GetTimestamp()
    $jobB=[PpoSessionJob]::new($other,$root,$start,20.)
    $jobA=[PpoSessionJob]::new($command,$root,$start,4.)
    $parentId=$jobA.ProcessId
    while (-not $jobA.Wait(100)) { Assert ([PpoSessionJob]::Remaining($start,9.) -gt 0) 'Synthetic watchdog did not stop process.' }
    Assert $jobA.Forced 'Missing forced-interruption flag.'
    Assert (-not $jobB.Wait(0)) 'A stopped an unrelated synthetic job.'
    Assert ([IO.File]::Exists($childReceipt)) 'Synthetic child was not started.'
    $childId=[int]([IO.File]::ReadAllText($childReceipt).Trim())
    $jobA.Dispose(); $jobA=$null
    $timer=[Diagnostics.Stopwatch]::StartNew()
    while ($null -ne (Get-Process -Id $childId -ErrorAction SilentlyContinue) -and $timer.Elapsed.TotalSeconds -lt 2) { Start-Sleep -Milliseconds 40 }
    Assert ($null -eq (Get-Process -Id $childId -ErrorAction SilentlyContinue)) 'Owned child survived Job Object termination.'
    Assert ($null -eq (Get-Process -Id $parentId -ErrorAction SilentlyContinue)) 'Owned worker survived termination.'
    $jobB.Dispose(); $jobB=$null
    $failed=$false
    try { $jobA=[PpoSessionJob]::new($other,$root,($start-30*[Diagnostics.Stopwatch]::Frequency),1.) } catch { $failed=$true }
    Assert $failed 'Exhausted deadline launched work.'
    # Simulate time consumed by compilation/initialization before training.
    $past=[Diagnostics.Stopwatch]::GetTimestamp()-3600*[Diagnostics.Stopwatch]::Frequency
    $remaining=[PpoSessionJob]::Remaining($past,27000.)
    Assert ($remaining -le 23400. -and $remaining -gt 23390.) 'Earlier session stages were not deducted.'

    $run=[IO.Path]::Combine($root,'run'); $report=[IO.Path]::Combine($root,'report')
    $null=[IO.Directory]::CreateDirectory($run); $null=[IO.Directory]::CreateDirectory($report)
    $index=@(); $rows=@(); $source=$null; $latest=$null
    foreach ($n in @(200,300,301)) {
        $path='\\?\'+[IO.Path]::Combine($run,('checkpoint-{0:D6}.ron' -f $n))
        [IO.File]::WriteAllText($path,'synthetic checkpoint: never loaded or evaluated')
        $role=if ($n -eq 200) {'source'} elseif ($n -eq 300) {'periodic'} else {'final'}
        $index += [pscustomobject]@{role=$role;rollout=$n;adam_step=($n*64);checkpoint=$path}
        if ($n -eq 200) {$source=$path}; if ($n -eq 301) {$latest=$path; continue}
        foreach ($track in @('monza','red_bull_ring','interlagos','catalunya','silverstone')) {
            $done=$track -ne 'silverstone'
            $rows += [pscustomobject]@{experiment='fixture';rollout=$n;checkpoint=$path;track=$track;
                completed=$done;collision=(-not $done);timeout=$false;progress=$(if($done){1.}else{0.5})}
        }
    }
    $index | Export-Csv -NoTypeInformation -LiteralPath ([IO.Path]::Combine($run,'checkpoint_index.csv'))
    $rows | Export-Csv -NoTypeInformation -LiteralPath ([IO.Path]::Combine($run,'periodic_evaluations.csv'))
    [pscustomobject]@{source_checkpoint=$source;best_checkpoint=$source;latest_checkpoint=$latest;source_rollout=200;latest_rollout=301;
        adam_step=19264;latest_evaluated=$false;stop_reason='wall_time_budget';environment_semantics='fixture'} |
        Export-Csv -NoTypeInformation -LiteralPath ([IO.Path]::Combine($run,'continuation_state.csv'))
    foreach($name in @('metrics.csv','update_metrics.csv','track_metrics.csv','run.log')) { [IO.File]::WriteAllText([IO.Path]::Combine($run,$name),'synthetic') }
    Write-PpoEContinuationReport -RunDirectory ('\\?\'+$run) -ReportDirectory ('\\?\'+$report)
    $manifest=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText([IO.Path]::Combine($report,'comparison_manifest.json')))
    Assert ($manifest.best_checkpoint.path -ceq $source) 'Source must win equal scores with a newer policy.'
    Assert ($manifest.latest_checkpoint.path -ceq $latest) 'Latest checkpoint lost.'
    Assert (-not $manifest.latest_checkpoint.evaluated) 'Unevaluated final snapshot was ranked.'
    Assert ($manifest.source_checkpoint.adam_step -eq 12800) 'Missing source Adam counter.'
    Assert ($manifest.source_checkpoint.sha256.Length -eq 64) 'Missing checkpoint SHA-256.'
    Write-Host "PASS (PowerShell $($PSVersionTable.PSVersion)): session accounting, expired start, owned process tree watchdog, unrelated process survival, exact receipts/paths, source tie-break, final unevaluated checkpoint and report hashes."
} finally {
    if($null -ne $jobA){$jobA.Dispose()}; if($null -ne $jobB){$jobB.Dispose()}
    $resolved=[IO.Path]::GetFullPath($root)
    $temp=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if(-not $resolved.StartsWith($temp,[StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($resolved).StartsWith('ppo-e-session-test-')) { throw 'Unsafe synthetic fixture cleanup.' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
