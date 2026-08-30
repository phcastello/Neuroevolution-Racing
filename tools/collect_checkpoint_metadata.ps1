<#
.SYNOPSIS
Collects metadata from versioned Neuroevolution Racing checkpoints.

.DESCRIPTION
Scans .ron files, writes one CSV row per valid checkpoint, and generates a
self-contained HTML report with experimental plots. Invalid files are reported
without stopping the remaining collection.

.EXAMPLE
.\tools\collect_checkpoint_metadata.ps1

.EXAMPLE
.\tools\collect_checkpoint_metadata.ps1 -LatestPerGeneration -PassThru
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string] $CheckpointDirectory = (Join-Path $PSScriptRoot '..\checkpoints'),

    [Parameter()]
    [string] $OutputPath = (Join-Path $PSScriptRoot '..\checkpoint_metadata.csv'),

    [Parameter()]
    [string] $ReportPath = (Join-Path $PSScriptRoot '..\checkpoint_metadata.html'),

    [Parameter()]
    [switch] $LatestPerGeneration,

    [Parameter()]
    [switch] $PassThru,

    [Parameter()]
    [switch] $NoHtmlReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$floatStyle = [System.Globalization.NumberStyles]::Float

function Get-RonScalar {
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [string] $Field
    )

    $pattern = '(?m)^\s*' + [regex]::Escape($Field) + ':\s*([^,\r\n]+)'
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) {
        throw "campo '$Field' nao encontrado"
    }
    return $match.Groups[1].Value.Trim()
}

function Get-RonScalarOrDefault {
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [string] $Field,
        [Parameter(Mandatory)] [string] $Default
    )

    $pattern = '(?m)^\s*' + [regex]::Escape($Field) + ':\s*([^,\r\n]+)'
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) {
        return $Default
    }
    return $match.Groups[1].Value.Trim()
}

function Get-RonListBody {
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [string] $Field
    )

    $pattern = '(?ms)^\s*' + [regex]::Escape($Field) + ':\s*\[(.*?)^\s*\],'
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) {
        throw "lista '$Field' nao encontrada"
    }
    return $match.Groups[1].Value
}

function ConvertTo-Integer {
    param([Parameter(Mandatory)] [string] $Value)
    return [long]::Parse($Value, $invariantCulture)
}

function ConvertTo-Number {
    param([Parameter(Mandatory)] [string] $Value)
    return [double]::Parse($Value, $floatStyle, $invariantCulture)
}

function Get-IntegerList {
    param([Parameter(Mandatory)] [string] $Body)
    return @([regex]::Matches($Body, '-?\d+') | ForEach-Object {
        [int]::Parse($_.Value, $invariantCulture)
    })
}

function Get-IdentifierList {
    param([Parameter(Mandatory)] [string] $Body)
    return @([regex]::Matches($Body, '\b[A-Za-z_][A-Za-z0-9_]*\b') | ForEach-Object {
        $_.Value
    })
}

function Get-StringList {
    param([Parameter(Mandatory)] [string] $Body)
    return @([regex]::Matches($Body, '"((?:\\.|[^"\\])*)"') | ForEach-Object {
        [regex]::Unescape($_.Groups[1].Value)
    })
}

function Get-GenomeParameterCount {
    param([Parameter(Mandatory)] [string] $Body)
    return [regex]::Matches(
        $Body,
        '(?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    ).Count
}

function Read-CheckpointMetadata {
    param([Parameter(Mandatory)] [System.IO.FileInfo] $File)

    $text = [System.IO.File]::ReadAllText($File.FullName)
    $formatVersion = ConvertTo-Integer (Get-RonScalar $text 'format_version')
    if ($formatVersion -notin @(1, 2, 3)) {
        throw "format_version $formatVersion nao suportado"
    }

    $timestamp = ConvertTo-Integer (Get-RonScalar $text 'saved_at_unix_seconds')
    $generation = ConvertTo-Integer (Get-RonScalar $text 'generation')
    $layerSizes = Get-IntegerList (Get-RonListBody $text 'layer_sizes')
    $activations = Get-IdentifierList (Get-RonListBody $text 'activations')
    $trainingTracks = Get-StringList (Get-RonListBody $text 'training_tracks')
    $genomeParameterCount = Get-GenomeParameterCount (Get-RonListBody $text 'genome')

    $completionRate = ConvertTo-Number (Get-RonScalar $text 'completion_rate')
    $validationProgress = ConvertTo-Number (Get-RonScalar $text 'normalized_progress')
    $savedAt = [DateTimeOffset]::FromUnixTimeSeconds($timestamp).ToLocalTime()
    $speedHalfSaturation = ConvertTo-Number (
        Get-RonScalarOrDefault $text 'progress_speed_half_saturation' '0'
    )
    if ($speedHalfSaturation -le 0.0) {
        $speedHalfSaturation = ConvertTo-Number (
            Get-RonScalar $text 'progress_speed_normalization'
        )
    }
    $speedNormalization = if ($formatVersion -ge 3) {
        'AsymptoticHalfSaturation'
    }
    else {
        'LegacyHardClamp'
    }

    [pscustomobject][ordered]@{
        generation                              = $generation
        filename                                = $File.Name
        format_version                          = $formatVersion
        saved_at_unix_seconds                   = $timestamp
        saved_at_local                          = $savedAt.ToString('yyyy-MM-dd HH:mm:ss zzz')
        architecture                            = ($layerSizes -join ' -> ')
        activations                             = ($activations -join ' -> ')
        genome_parameter_count                  = $genomeParameterCount
        champion_training_fitness               = ConvertTo-Number (Get-RonScalar $text 'champion_training_fitness')
        population_average_fitness              = ConvertTo-Number (Get-RonScalar $text 'population_average_fitness')
        average_useful_progress_speed_u_s       = ConvertTo-Number (Get-RonScalar $text 'average_useful_progress_speed')
        completion_rate                         = $completionRate
        completion_rate_percent                 = $completionRate * 100.0
        completed                               = ConvertTo-Integer (Get-RonScalar $text 'completed')
        collision                               = ConvertTo-Integer (Get-RonScalar $text 'collision')
        stalled                                 = ConvertTo-Integer (Get-RonScalarOrDefault $text 'stalled' '0')
        laser_eliminated                        = ConvertTo-Integer (Get-RonScalarOrDefault $text 'laser_eliminated' '0')
        timeout                                 = ConvertTo-Integer (Get-RonScalar $text 'timeout')
        training_tracks                         = ($trainingTracks -join ';')
        validation_track                        = (Get-RonScalar $text 'track_id').Trim('"')
        validation_score                        = ConvertTo-Number (Get-RonScalar $text 'score')
        validation_normalized_progress          = $validationProgress
        validation_progress_percent             = $validationProgress * 100.0
        validation_useful_progress_speed_u_s    = ConvertTo-Number (Get-RonScalar $text 'useful_progress_speed')
        validation_elapsed_s                    = ConvertTo-Number (Get-RonScalar $text 'elapsed')
        validation_finish_reason                = Get-RonScalar $text 'finish_reason'
        maximum_episode_duration_s              = ConvertTo-Number (Get-RonScalar $text 'maximum_episode_duration')
        stall_timeout_s                         = ConvertTo-Number (Get-RonScalarOrDefault $text 'stall_timeout' '0')
        significant_progress_epsilon_u          = ConvertTo-Number (Get-RonScalarOrDefault $text 'significant_progress_epsilon' '0')
        laser_grace_period_s                    = ConvertTo-Number (Get-RonScalarOrDefault $text 'grace_period' '0')
        laser_acceleration_u_s2                 = ConvertTo-Number (Get-RonScalarOrDefault $text 'acceleration' '0')
        laser_maximum_speed_u_s                 = ConvertTo-Number (Get-RonScalarOrDefault $text 'maximum_speed' '0')
        sensor_max_distance_u                   = ConvertTo-Number (Get-RonScalarOrDefault $text 'sensor_max_distance' '0')
        progress_weight                         = ConvertTo-Number (Get-RonScalar $text 'progress_weight')
        speed_weight                            = ConvertTo-Number (Get-RonScalar $text 'speed_weight')
        collision_penalty                       = ConvertTo-Number (Get-RonScalar $text 'collision_penalty')
        completion_bonus                        = ConvertTo-Number (Get-RonScalar $text 'completion_bonus')
        useful_speed_normalization              = $speedNormalization
        progress_speed_half_saturation_u_s      = $speedHalfSaturation
        progress_speed_normalization_u_s        = $speedHalfSaturation
        training_track_selection                = Get-RonScalar $text 'training_track_selection'
    }
}

function Write-HtmlReport {
    param(
        [Parameter(Mandatory)] [object[]] $Rows,
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [string] $SourceDirectory
    )

    $json = $Rows | ConvertTo-Json -Depth 4 -Compress
    $template = @'
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuroevolution Racing &mdash; Checkpoint Metadata</title>
<style>
:root { color-scheme: light dark; --bg:#101614; --panel:#17201d; --fg:#e5eee9; --muted:#9fb0a8; --grid:#33403b; --border:#3d4a45; }
@media (prefers-color-scheme: light) { :root { --bg:#f5f8f6; --panel:#fff; --fg:#18201d; --muted:#5d6b65; --grid:#dce4e0; --border:#cad5cf; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }
main { max-width:1400px; margin:auto; }
h1 { margin:0 0 4px; font-size:24px; font-weight:600; }
h2 { margin:0 0 8px; font-size:16px; font-weight:600; }
.subtitle { color:var(--muted); margin-bottom:20px; overflow-wrap:anywhere; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.panel { min-width:0; padding:16px; background:var(--panel); border:1px solid var(--border); border-radius:8px; }
.wide { grid-column:1 / -1; }
svg { display:block; width:100%; height:auto; overflow:visible; }
.axis { stroke:var(--muted); stroke-width:1; }
.grid-line { stroke:var(--grid); stroke-width:1; }
.tick { fill:var(--muted); font-size:11px; }
.legend { display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); margin-bottom:6px; }
.legend span::before { content:""; display:inline-block; width:10px; height:3px; margin-right:5px; vertical-align:middle; background:var(--series); }
.summary { color:var(--muted); margin:0 0 18px; }
@media (max-width:800px) { body { padding:12px; } .grid { grid-template-columns:1fr; } .wide { grid-column:auto; } }
</style>
</head>
<body>
<main>
<h1>Neuroevolution Racing &mdash; checkpoints</h1>
<p class="summary" id="summary"></p>
<p class="subtitle">Fonte: __SOURCE_DIRECTORY__</p>
<div class="grid">
  <section class="panel"><h2>Fitness de treino</h2><div id="fitness"></div></section>
  <section class="panel"><h2>Velocidade &uacute;til de progresso</h2><div id="speed"></div></section>
  <section class="panel"><h2>Conclus&atilde;o e progresso de valida&ccedil;&atilde;o</h2><div id="rates"></div></section>
  <section class="panel"><h2>Score de valida&ccedil;&atilde;o</h2><div id="validation"></div></section>
  <section class="panel wide"><h2>Motivos de t&eacute;rmino do campe&atilde;o nas pistas de treino</h2><div id="reasons"></div></section>
</div>
</main>
<script>
const rows = __CHECKPOINT_DATA__;
const palette = { green:'#55c98a', blue:'#58a6e7', amber:'#e3b75c', red:'#dd6b72', violet:'#aa83e8', gray:'#9aa7a1' };
const reasonColors = { Completed:palette.green, Collision:palette.red, Stalled:palette.amber, Laser_eliminated:palette.blue, EliminatedByLaser:palette.blue, Timeout:palette.violet };
const ns = 'http://www.w3.org/2000/svg';
const fmt = value => Number(value).toLocaleString('pt-BR', { maximumFractionDigits:3 });
document.getElementById('summary').textContent = `${rows.length} checkpoints \u2022 gera\u00e7\u00f5es ${rows[0].generation}\u2013${rows[rows.length-1].generation}`;

function addLegend(host, items) {
  const legend = document.createElement('div'); legend.className = 'legend';
  items.forEach(item => { const span=document.createElement('span'); span.style.setProperty('--series',item.color); span.textContent=item.label; legend.appendChild(span); });
  host.appendChild(legend);
}
function lineChart(id, series, unit, options={}) {
  const host=document.getElementById(id); addLegend(host, series);
  const W=760,H=260,m={l:58,r:16,t:12,b:38},pw=W-m.l-m.r,ph=H-m.t-m.b;
  const all=series.flatMap(s=>rows.map(r=>Number(r[s.key]))).filter(Number.isFinite);
  let ymin=options.zeroBased===false ? Math.min(...all) : Math.min(0,...all), ymax=Math.max(...all);
  if (ymax===ymin) ymax=ymin+1;
  const xmin=Number(rows[0].generation), xmax=Number(rows[rows.length-1].generation), xr=Math.max(1,xmax-xmin), yr=ymax-ymin;
  const x=v=>m.l+(Number(v)-xmin)/xr*pw, y=v=>m.t+(ymax-Number(v))/yr*ph;
  const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('role','img'); svg.setAttribute('aria-label',`${id}: evolu\u00e7\u00e3o por gera\u00e7\u00e3o`);
  for(let i=0;i<=4;i++){const value=ymin+yr*i/4, py=y(value); svg.insertAdjacentHTML('beforeend',`<line class="grid-line" x1="${m.l}" x2="${W-m.r}" y1="${py}" y2="${py}"/><text class="tick" x="${m.l-8}" y="${py+4}" text-anchor="end">${fmt(value)}</text>`);}
  for(let i=0;i<=5;i++){const value=Math.round(xmin+xr*i/5),px=x(value); svg.insertAdjacentHTML('beforeend',`<text class="tick" x="${px}" y="${H-12}" text-anchor="middle">${value}</text>`);}
  svg.insertAdjacentHTML('beforeend',`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/><text class="tick" x="${W/2}" y="${H-1}" text-anchor="middle">gera\u00e7\u00e3o</text><text class="tick" transform="translate(13 ${H/2}) rotate(-90)" text-anchor="middle">${unit}</text>`);
  series.forEach(s=>{const points=rows.map(r=>`${x(r.generation)},${y(r[s.key])}`).join(' '); svg.insertAdjacentHTML('beforeend',`<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2"/>`); rows.forEach(r=>{const c=s.pointColor?s.pointColor(r):s.color; const circle=document.createElementNS(ns,'circle'); circle.setAttribute('cx',x(r.generation)); circle.setAttribute('cy',y(r[s.key])); circle.setAttribute('r','2.5'); circle.setAttribute('fill',c); const title=document.createElementNS(ns,'title'); title.textContent=`Gera\u00e7\u00e3o ${r.generation} \u2022 ${s.label}: ${fmt(r[s.key])}${s.suffix||''}`; circle.appendChild(title); svg.appendChild(circle);});});
  host.appendChild(svg);
}
function stackedReasons() {
  const host=document.getElementById('reasons'); const keys=['completed','collision','stalled','laser_eliminated','timeout']; addLegend(host,keys.map(k=>({label:k,color:reasonColors[k[0].toUpperCase()+k.slice(1)]})));
  const W=1200,H=250,m={l:55,r:16,t:12,b:38},pw=W-m.l-m.r,ph=H-m.t-m.b,xmin=rows[0].generation,xmax=rows[rows.length-1].generation,xr=Math.max(1,xmax-xmin),bar=Math.max(2,pw/rows.length*.75);
  const x=v=>m.l+(v-xmin)/xr*pw,y=p=>m.t+(1-p)*ph; const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('role','img'); svg.setAttribute('aria-label','Propor\u00e7\u00e3o dos motivos de t\u00e9rmino por gera\u00e7\u00e3o');
  [0,.25,.5,.75,1].forEach(p=>svg.insertAdjacentHTML('beforeend',`<line class="grid-line" x1="${m.l}" x2="${W-m.r}" y1="${y(p)}" y2="${y(p)}"/><text class="tick" x="${m.l-8}" y="${y(p)+4}" text-anchor="end">${p*100}%</text>`));
  rows.forEach(r=>{const total=keys.reduce((a,k)=>a+Number(r[k]),0)||1; let acc=0; keys.forEach(k=>{const part=Number(r[k])/total,top=acc+part,rect=document.createElementNS(ns,'rect'); rect.setAttribute('x',x(r.generation)-bar/2); rect.setAttribute('y',y(top)); rect.setAttribute('width',bar); rect.setAttribute('height',Math.max(0,y(acc)-y(top))); rect.setAttribute('fill',reasonColors[k[0].toUpperCase()+k.slice(1)]); const title=document.createElementNS(ns,'title'); title.textContent=`Gera\u00e7\u00e3o ${r.generation} \u2022 ${k}: ${r[k]}/${total}`; rect.appendChild(title); svg.appendChild(rect); acc=top;});});
  for(let i=0;i<=5;i++){const value=Math.round(xmin+xr*i/5),px=x(value); svg.insertAdjacentHTML('beforeend',`<text class="tick" x="${px}" y="${H-12}" text-anchor="middle">${value}</text>`);}
  svg.insertAdjacentHTML('beforeend',`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/><text class="tick" x="${W/2}" y="${H-1}" text-anchor="middle">gera\u00e7\u00e3o</text>`); host.appendChild(svg);
}
lineChart('fitness',[{key:'champion_training_fitness',label:'campe\u00e3o',color:palette.green},{key:'population_average_fitness',label:'m\u00e9dia da popula\u00e7\u00e3o',color:palette.blue}],'fitness');
lineChart('speed',[{key:'average_useful_progress_speed_u_s',label:'treino do campe\u00e3o',color:palette.green,suffix:' u/s'},{key:'validation_useful_progress_speed_u_s',label:'valida\u00e7\u00e3o',color:palette.amber,suffix:' u/s'}],'u/s');
lineChart('rates',[{key:'completion_rate_percent',label:'completion rate',color:palette.green,suffix:'%'},{key:'validation_progress_percent',label:'progresso de valida\u00e7\u00e3o',color:palette.blue,suffix:'%'}],'%');
lineChart('validation',[{key:'validation_score',label:'score',color:palette.gray,pointColor:r=>reasonColors[r.validation_finish_reason]||palette.gray}],'score',{zeroBased:false});
addLegend(document.getElementById('validation'),Object.entries(reasonColors).map(([label,color])=>({label,color})));
stackedReasons();
</script>
</body>
</html>
'@

    $html = $template.Replace('__CHECKPOINT_DATA__', $json)
    $html = $html.Replace('__SOURCE_DIRECTORY__', [System.Net.WebUtility]::HtmlEncode($SourceDirectory))
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $html, $utf8WithoutBom)
}

$resolvedCheckpointDirectory = [System.IO.Path]::GetFullPath($CheckpointDirectory)
if (-not [System.IO.Directory]::Exists($resolvedCheckpointDirectory)) {
    throw "Diretorio de checkpoints nao encontrado: $resolvedCheckpointDirectory"
}

$checkpointFiles = @(
    Get-ChildItem -LiteralPath $resolvedCheckpointDirectory -File -Filter '*.ron' |
        Sort-Object Name
)
if ($checkpointFiles.Count -eq 0) {
    throw "Nenhum arquivo .ron encontrado em: $resolvedCheckpointDirectory"
}

$metadata = [System.Collections.Generic.List[object]]::new()
$errors = [System.Collections.Generic.List[object]]::new()
foreach ($file in $checkpointFiles) {
    try {
        $metadata.Add((Read-CheckpointMetadata $file))
    }
    catch {
        $errors.Add([pscustomobject][ordered]@{
            filename = $file.Name
            error    = $_.Exception.Message
        })
    }
}

$rows = @($metadata | Sort-Object generation, saved_at_unix_seconds)
if ($LatestPerGeneration) {
    $rows = @(
        $rows |
            Group-Object generation |
            ForEach-Object { $_.Group | Sort-Object saved_at_unix_seconds -Descending | Select-Object -First 1 } |
            Sort-Object generation
    )
}

if ($rows.Count -eq 0) {
    throw "Nenhum checkpoint valido foi encontrado. Arquivos invalidos: $($errors.Count)"
}

$resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = [System.IO.Path]::GetDirectoryName($resolvedOutputPath)
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}
$rows | Export-Csv -LiteralPath $resolvedOutputPath -NoTypeInformation -Encoding UTF8

Write-Host "Metadados coletados: $($rows.Count) checkpoint(s) valido(s)."
Write-Host "CSV: $resolvedOutputPath"
if (-not $NoHtmlReport) {
    $resolvedReportPath = [System.IO.Path]::GetFullPath($ReportPath)
    $reportDirectory = [System.IO.Path]::GetDirectoryName($resolvedReportPath)
    if (-not [string]::IsNullOrWhiteSpace($reportDirectory)) {
        [System.IO.Directory]::CreateDirectory($reportDirectory) | Out-Null
    }
    Write-HtmlReport -Rows $rows -Path $resolvedReportPath -SourceDirectory $resolvedCheckpointDirectory
    Write-Host "Grafico HTML: $resolvedReportPath"
}
if ($errors.Count -gt 0) {
    Write-Warning "$($errors.Count) arquivo(s) invalido(s) foram ignorados:"
    $errors | Format-Table -AutoSize | Out-Host
}

if ($PassThru) {
    $rows
}
