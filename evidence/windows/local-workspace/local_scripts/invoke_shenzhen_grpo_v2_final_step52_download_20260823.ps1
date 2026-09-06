$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$run = '/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2'
$destination = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\grpo_v2_final_step52_20260823'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)
$files = [ordered]@{
    "$run/driver.log" = 'driver.log'
    "$run/metrics.log" = 'metrics.log'
    "$run/resource.csv" = 'resource.csv'
    "$run/resource_observer.log" = 'resource_observer.log'
    "$run/resolved.yaml" = 'resolved.yaml'
    "$run/launch_manifest.txt" = 'launch_manifest.txt'
    "$run/driver.exit" = 'driver.exit'
}

if (Test-Path -LiteralPath $destination) {
    throw "Refusing to reuse snapshot directory: $destination"
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    foreach ($entry in $files.GetEnumerator()) {
        $local = Join-Path $destination $entry.Value
        & $python $helper @connection get $entry.Key $local
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP get failed for $($entry.Key) with exit $LASTEXITCODE"
        }
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-ChildItem -LiteralPath $destination -File |
    Sort-Object Name |
    Select-Object Name, Length
