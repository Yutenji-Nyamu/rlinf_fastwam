$ErrorActionPreference = 'Stop'

$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = 'local_scripts\remote_exec_autodl.py'
$destinationRoot = 'exports\ogpo_formal_90k_live_20260808_1807'
New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null

$runtime = '/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime'
$run = '/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2'
$event = "$run/tensorboard/events.out.tfevents.1786166872.autodl-container-nekaqbwt43-6ce5babb.99682.0"
$files = @(
  @("$runtime/driver.log", 'driver.log'),
  @("$runtime/resources_1s.csv", 'resources_1s.csv'),
  @("$runtime/resolved.yaml", 'resolved.yaml'),
  @("$runtime/source_config.yaml", 'source_config.yaml'),
  @("$runtime/exact_command.txt", 'exact_command.txt'),
  @("$runtime/run_provenance.tsv", 'run_provenance.tsv'),
  @("$runtime/stop_conditions.txt", 'stop_conditions.txt'),
  @("$run/metrics.log", 'metrics.log'),
  @("$run/tensorboard/config.yaml", 'tensorboard_config.yaml'),
  @($event, 'events.out.tfevents')
)

foreach ($entry in $files) {
  $remote = $entry[0]
  $local = Join-Path $destinationRoot $entry[1]
  & $python $helper get $remote $local
  if ($LASTEXITCODE -ne 0) {
    throw "download failed: $remote"
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLowerInvariant()
  $bytes = (Get-Item -LiteralPath $local).Length
  Write-Output "GET_OK`t$bytes`t$actual`t$remote`t$local"
}
