$ErrorActionPreference = 'Stop'

$root = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $root 'local_scripts\remote_exec_autodl.py'
$remote = '/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30/runtime'
$local = Join-Path $root 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-grpo-w0to2-final-step41-20260825'
New-Item -ItemType Directory -Force -Path $local | Out-Null

$files = @(
    'driver.log',
    'resource.csv',
    'resolved.yaml',
    'launch_manifest.txt',
    'resume_parity.json',
    'contract.json',
    'stopped_by_user_for_w0to5.txt',
    'tensorboard_scalars_snapshot.json'
)

foreach ($file in $files) {
    & $python $helper `
        --host 120.241.223.9 `
        --port 22 `
        --user chenyiteng `
        --host-key-sha256 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY' `
        --timeout 30 `
        get "$remote/$file" (Join-Path $local $file)
    if ($LASTEXITCODE -ne 0) { throw "download failed: $file" }
}

Write-Output $local
