$ErrorActionPreference = 'Stop'

$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = 'local_scripts\remote_exec_autodl.py'
$destinationRoot = 'exports\ogpo_smoke_20260807_v1'
New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null

$runtime = '/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime'
$run = '/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1'
$checkpoint = "$run/robotwin_adjust_bottle_ogpo_ca_smoke_8env_1update_v1/checkpoints/global_step_1/actor/ogpo_components/complete.json"
$event = "$run/tensorboard/events.out.tfevents.1786103639.autodl-container-nekaqbwt43-6ce5babb.75167.0"

$files = @(
  @("$runtime/checkpoint_summary.json", 'checkpoint_summary.json', '864de33ae54befcee03ccfb855dc45f0a595875a530e0c400bbfb40c4ad2225a'),
  @("$runtime/driver.log", 'driver.log', 'b22d5cbb8ac7885088d0bd5d9ae79d6fc7ef05c76e0c7862ccadf0f40100f0a3'),
  @("$runtime/driver_pid.txt", 'driver_pid.txt', 'c4ed257b7f487fd18d0717d039347a8e601f3bab1849f32143a8b9f5e813a4ea'),
  @("$runtime/monitor_pid.txt", 'monitor_pid.txt', 'd06946a4ea5ca82055684c1da0e3d732ad357fe2cc40adfdb2566271d00c9963'),
  @("$runtime/exact_command.txt", 'exact_command.txt', '00e096f9cc20b2adac5a1d767f16e5a1cf6b3a40c5d60412c42b12756eac0621'),
  @("$runtime/exit_code.txt", 'exit_code.txt', '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa'),
  @("$runtime/started_at.txt", 'started_at.txt', 'e8dedc30cf140a3df164f765431c042a0b61240a370a8b1bd9236fbdbabef1a8'),
  @("$runtime/finished_at.txt", 'finished_at.txt', '6e1ec091c245b5cccd37d7004f593a2ee844dc40208a8f799d6db017b9217ae0'),
  @("$runtime/resolved.yaml", 'resolved.yaml', 'ebd163f647d2a9399fdca007099fac550f6f344392162adc36eede129671f4eb'),
  @("$runtime/resource_monitor.log", 'resource_monitor.log', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
  @("$runtime/resource_summary.json", 'resource_summary.json', '103f0ef54775c272a6e0db0cb39b5cfc4930537486449100c163f0480475c375'),
  @("$runtime/resources_1s.csv", 'resources_1s.csv', 'fc273b773ce6badafaf198ab23b4c5a0edee03fac2290a82c1db8cdcefe991a1'),
  @("$runtime/resources_after.txt", 'resources_after.txt', '63e7d368cb7f65ddf2ae2c21311d680909c20327b72294c59b179f7752ad3f9d'),
  @("$runtime/resources_before.txt", 'resources_before.txt', '57788cf32356748cb9a9e86069992fa01866924dabf84c91e81de07d3a75f748'),
  @("$runtime/run_provenance.tsv", 'run_provenance.tsv', 'f8e6857c36144e2de24760640740fed1a49acbac5750efd7b914c455c1422120'),
  @("$runtime/source_config.yaml", 'source_config.yaml', 'f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291'),
  @("$runtime/stop_conditions.txt", 'stop_conditions.txt', '8984fd3aa372bf23d12bc2853a1c0f50483b240db6f3142b9a4c7e325d4ca986'),
  @("$run/metrics.log", 'metrics.log', 'e83f6999c1598ee24ddf9c0e90c3cd57c18a80af9b94126d1892b87445e821a7'),
  @("$run/tensorboard/config.yaml", 'tensorboard_config.yaml', '7c03c8fb2ef916344303fd26f02bbe097a267d7c70e539f252f3ec19bd286259'),
  @($event, 'events.out.tfevents', '60060aee318eab1c956968c1f8d6de0f9309609981651d9678901fd400e9f6c5'),
  @($checkpoint, 'checkpoint_complete.json', 'b20cf427bdf9272267023d2e69444578a3eaefcc16ff1b6953b13e39ebcc5b60')
)

foreach ($entry in $files) {
  $remote = $entry[0]
  $local = Join-Path $destinationRoot $entry[1]
  $expected = $entry[2]
  & $python $helper get $remote $local
  if ($LASTEXITCODE -ne 0) {
    throw "download failed: $remote"
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLowerInvariant()
  if ($actual -ne $expected) {
    throw "SHA mismatch: $remote expected=$expected actual=$actual"
  }
  $bytes = (Get-Item -LiteralPath $local).Length
  Write-Output "GET_OK`t$bytes`t$actual`t$remote`t$local"
}
