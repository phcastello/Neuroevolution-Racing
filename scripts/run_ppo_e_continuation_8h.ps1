[CmdletBinding()]
param(
    [string]$ResumeCheckpoint,
    [switch]$Worker,
    [string]$SessionDirectory,
    [long]$SessionStartTicks = 0
)
# Start before setup, manifest reads, helper compilation or any child process.
$entryTicks = [Diagnostics.Stopwatch]::GetTimestamp()
$ErrorActionPreference = 'Stop'
if (-not $Worker) { $SessionStartTicks = $entryTicks }
if ($SessionStartTicks -le 0) { throw 'Missing monotonic session origin.' }
$repoRoot = [IO.Path]::GetFullPath([IO.Path]::Combine($PSScriptRoot,'..'))
$selectionPath = [IO.Path]::Combine($repoRoot,'configs\rl\ppo_e_continue_8h_source.json')
$selection = ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($selectionPath))
# Prepared .NET Framework assembly avoids launching an unowned compiler before
# the supervisor exists. Its auditable C# source and binary must both match.
foreach ($name in @('supervisor_assembly','supervisor_assembly_sha256','supervisor_source_sha256')) {
    if ($null -eq $selection.PSObject.Properties[$name] -or [string]::IsNullOrWhiteSpace($selection.$name)) { throw "Missing preparation property: $name" }
}
$supervisorAssembly = [IO.Path]::Combine($repoRoot,$selection.supervisor_assembly)
if (-not [IO.File]::Exists($supervisorAssembly)) { throw "Missing prepared supervisor assembly: $supervisorAssembly" }
if ((Get-FileHash -LiteralPath $supervisorAssembly -Algorithm SHA256).Hash -cne $selection.supervisor_assembly_sha256 -or
    (Get-FileHash -LiteralPath ([IO.Path]::Combine($PSScriptRoot,'ppo_session_job.cs')) -Algorithm SHA256).Hash -cne $selection.supervisor_source_sha256) { throw 'Supervisor source/assembly SHA-256 mismatch.' }
Add-Type -Path $supervisorAssembly
function Get-SessionElapsed { ([Diagnostics.Stopwatch]::GetTimestamp()-$SessionStartTicks)/[double][Diagnostics.Stopwatch]::Frequency }
function Write-NewSessionJson([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -InputObject $Value -Depth 12))
    $file = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $file.Write($bytes,0,$bytes.Length); $file.Flush($true) } finally { $file.Dispose() }
}
if (-not $Worker) {
    $sessionRoot = [IO.Path]::Combine($repoRoot,'results_ppo\e_continuation_8h\sessions')
    $SessionDirectory = [IO.Path]::Combine($sessionRoot,('session-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[Guid]::NewGuid().ToString('N')))
    $null = [IO.Directory]::CreateDirectory($SessionDirectory)
    Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'session_budget.json')) ([ordered]@{
        session_start_ticks=$SessionStartTicks; monotonic_frequency=[Diagnostics.Stopwatch]::Frequency;
        cooperative_deadline_seconds=27000; maximum_session_seconds=28800;
        external_cutoff_seconds=28790; supervisor_finalize_reserve_seconds=10;
        includes=@('compilation','initialization','training','evaluation','persistence','finalization','report')
    })
    $powershell = [IO.Path]::Combine($env:SystemRoot,'System32\WindowsPowerShell\v1.0\powershell.exe')
    $arguments = @($powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath,
        '-Worker','-SessionDirectory',$SessionDirectory,'-SessionStartTicks',[string]$SessionStartTicks)
    if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) { $arguments += @('-ResumeCheckpoint',$ResumeCheckpoint) }
    $command = ($arguments | ForEach-Object { [PpoSessionJob]::Quote([string]$_) }) -join ' '
    $job = $null; $exitCode = 1; $forced = $false
    Write-Host "Session directory: $SessionDirectory"
    try {
        # All worker descendants (cargo, rustc, trainer, report) inherit this job.
        # Assignment happens while suspended; no process-name or global tree kill.
        $job = [PpoSessionJob]::new($command,$repoRoot,$SessionStartTicks,28790.)
        while (-not $job.Wait(1000)) { }
        $forced = $job.Forced; $exitCode = $job.ExitCode
    } finally {
        if ($null -ne $job) { $forced = $forced -or $job.Forced; $job.Dispose() }
        $reason = if ($forced) { 'exceptional_forced_interruption' } elseif ($exitCode -eq 0) { 'normal_worker_completion' } else { 'worker_failure' }
        Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'supervision.json')) ([ordered]@{
            stop_reason=$reason; forced=$forced; worker_exit_code=$exitCode; session_elapsed_seconds=(Get-SessionElapsed);
            checkpoint_policy='Only published checkpoints listed in checkpoint_index.csv are candidates for recovery; ignore temporary files.'
        })
    }
    if ($forced) { throw "Exceptional forced interruption at the session deadline. Last published checkpoints are preserved. Session: $SessionDirectory" }
    if ($exitCode -ne 0) { throw "Session worker failed. Inspect worker_status.json and build/training logs: $SessionDirectory" }
    Write-Host "Session finished. Reports and exact run receipt: $SessionDirectory"
    exit 0
}

# Internal worker. The outer Job Object enforces the deadline on this whole block.
if (-not [IO.Directory]::Exists($SessionDirectory)) { throw 'Missing owned session directory.' }
$stage = 'preparation'; $reason = 'worker_failure'; $runPath = $null
try {
    Set-Location -LiteralPath $repoRoot
    function Invoke-SessionProcess([string]$Executable, [string[]]$Arguments, [string]$LogName) {
        $quoted = ($Arguments | ForEach-Object { [PpoSessionJob]::Quote($_) }) -join ' '
        $process = Start-Process -FilePath $Executable -ArgumentList $quoted -WorkingDirectory $repoRoot -WindowStyle Hidden `
            -PassThru -RedirectStandardOutput ([IO.Path]::Combine($SessionDirectory,"$LogName.stdout.log")) `
            -RedirectStandardError ([IO.Path]::Combine($SessionDirectory,"$LogName.stderr.log"))
        # Cache the handle before waiting. PS 5.1 Start-Process otherwise loses
        # ExitCode for a process with redirected streams; Refresh clears it too.
        $null = $process.Handle
        $process.WaitForExit()
        if ($null -eq $process.ExitCode) { throw "Missing process exit code: $Executable" }
        return $process.ExitCode
    }
    . ([IO.Path]::Combine($PSScriptRoot,'ppo_e_continuation_report.ps1'))
    $defaultSource = [string]::IsNullOrWhiteSpace($ResumeCheckpoint)
    if ($defaultSource) { $ResumeCheckpoint = [IO.Path]::Combine($repoRoot,$selection.source_checkpoint) }
    $original = Get-PpoReportAbsolutePath $ResumeCheckpoint 'ResumeCheckpoint'
    if (-not [IO.File]::Exists($original)) { throw "Missing source checkpoint: $original" }
    $sourceHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
    if ($defaultSource -and $sourceHash -cne $selection.sha256) { throw 'Selected source checkpoint SHA-256 mismatch.' }
    $source = [IO.Path]::Combine($SessionDirectory,'source_checkpoint.ron')
    # New session owns this immutable source copy. Never change its source file.
    $from = [IO.File]::OpenRead($original)
    try {
        $to = [IO.File]::Open($source,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $from.CopyTo($to); $to.Flush($true) } finally { $to.Dispose() }
    } finally { $from.Dispose() }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -cne $sourceHash) { throw 'Source preservation failed.' }
    $config = [IO.Path]::Combine($repoRoot,'configs\rl\ppo_e_continue_8h.ron')
    $environmentSource=[IO.Path]::Combine($repoRoot,'crates\app\src\simulation\rl_environment.rs')
    if ((Get-FileHash -LiteralPath $environmentSource -Algorithm SHA256).Hash -cne $selection.environment_source_sha256) {
        throw 'Environment implementation differs from the corrected source-selection implementation. Revalidate preparation before starting.'
    }
    $codePaths = @('crates/app/src/rl.rs','crates/app/src/rl/evaluation.rs','crates/app/src/rl/session_budget.rs',
        'crates/app/src/rl/persistence_policy.rs','crates/app/src/simulation/rl_environment.rs','crates/rl/src/persistence.rs',
        'scripts/run_ppo_e_continuation_8h.ps1','scripts/ppo_session_job.cs','scripts/ppo_e_continuation_report.ps1','Cargo.lock')
    $code = @($codePaths | ForEach-Object { $path=[IO.Path]::Combine($repoRoot,$_); [pscustomobject]@{ path=$path; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash } })
    Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'source_manifest.json')) ([ordered]@{
        original_source_checkpoint=$original; source_checkpoint=$source; sha256=$sourceHash;
        selection_manifest=$selectionPath; selected_preparation=$selection.preparation_manifest;
        selection_manifest_sha256=(Get-FileHash -LiteralPath $selectionPath -Algorithm SHA256).Hash;
        preparation_manifest_sha256=(Get-FileHash -LiteralPath ([IO.Path]::Combine($repoRoot,$selection.preparation_manifest)) -Algorithm SHA256).Hash;
        supervisor_assembly_sha256=$selection.supervisor_assembly_sha256;
        source_override=(-not $defaultSource); environment_semantics='rl-progress-connected-adjacent-v2';
        config=$config; config_sha256=(Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash; implementation=$code;
        resume_contract='Restore parameters, Adam moments/step, RNG states and accumulated counters; restart environments once; no rollback.'
    })
    if ((Get-SessionElapsed) -ge 27000) { $reason='wall_time_budget'; return }
    $stage='compilation'
    $buildExit = Invoke-SessionProcess (Get-Command cargo -ErrorAction Stop).Source @('build','--release','--locked','-p','neuroevolution-racing-app','--features','rl-gpu') 'build'
    if ($buildExit -ne 0) { throw "Release GPU build failed: $buildExit" }
    $executable = [IO.Path]::Combine($repoRoot,'target\release\neuroevolution-racing.exe')
    if (-not [IO.File]::Exists($executable)) { throw "Missing release executable: $executable" }
    Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'build_manifest.json')) ([ordered]@{
        executable=$executable; sha256=(Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash;
        compiled_at_session_elapsed_seconds=(Get-SessionElapsed); gpu_adapter=0; silent_cpu_fallback=$false
    })
    $elapsed = Get-SessionElapsed
    # Five seconds also cover process launch/CLI handoff. The Rust clock includes
    # configuration loading, GPU initialization and scene setup after entry.
    $remaining = 27000. - $elapsed - 5.
    if ($remaining -le 360.) { $reason='wall_time_budget'; return }
    $receipt = [IO.Path]::Combine($SessionDirectory,'run-path.txt')
    $stage='training'
    $invariant = [Globalization.CultureInfo]::InvariantCulture
    $trainExit = Invoke-SessionProcess $executable @('rl','train','--config',$config,'--resume',$source,
        '--save-every','100','--save-first','0','--save-last','0','--no-save-rollouts',
        '--evaluate-every','100','--experiment','E_continue_8h','--run-path-file',$receipt,
        '--wall-time-seconds',$remaining.ToString('R',$invariant),
        '--session-elapsed-seconds',$elapsed.ToString('R',$invariant)) 'training'
    if ($trainExit -ne 0) { throw "Continuation process failed: $trainExit" }
    if (-not [IO.File]::Exists($receipt)) { throw 'Training returned no exact run receipt.' }
    $runPath = Get-PpoReportAbsolutePath ([IO.File]::ReadAllText($receipt).Trim()) 'Run receipt'
    $stage='report'
    Write-PpoEContinuationReport -RunDirectory $runPath -ReportDirectory $SessionDirectory
    $state = Import-Csv -LiteralPath ([IO.Path]::Combine($runPath,'continuation_state.csv'))
    $reason = $state.stop_reason
    if ($reason -notin @('wall_time_budget','rollout_limit')) { throw "Unexpected normal stop reason: $reason" }
    $stage='complete'
} catch {
    Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'worker_error.json')) ([ordered]@{ stage=$stage; error=$_.Exception.Message; session_elapsed_seconds=(Get-SessionElapsed) })
    throw
} finally {
    Write-NewSessionJson ([IO.Path]::Combine($SessionDirectory,'worker_status.json')) ([ordered]@{
        stage=$stage; stop_reason=$reason; run_directory=$runPath; session_elapsed_seconds=(Get-SessionElapsed)
    })
}
