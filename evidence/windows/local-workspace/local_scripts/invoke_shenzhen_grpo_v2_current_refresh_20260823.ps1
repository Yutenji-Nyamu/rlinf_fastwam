$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$commandFile = Join-Path $workspace 'local_scripts\remote_commands\shenzhen_grpo_v2_current_refresh_20260823.sh'
$run = '/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2'
$destination = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\grpo_v2_live_current_20260823'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)
$files = [ordered]@{
    "$run/metrics.log" = 'metrics.log'
    "$run/resource.csv" = 'resource.csv'
    "$run/driver.log" = 'driver.log'
}

if (Test-Path -LiteralPath $destination) {
    throw "Refusing to reuse snapshot directory: $destination"
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    & $python $helper @connection run --command-file $commandFile
    if ($LASTEXITCODE -ne 0) {
        throw "Remote read-only refresh failed with exit $LASTEXITCODE"
    }
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

