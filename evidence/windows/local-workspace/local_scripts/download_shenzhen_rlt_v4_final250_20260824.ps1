$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$remoteRun = '/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix'
$destination = Join-Path $workspace 'docs\rlinf-shenzhen-rlt-dsrl-port\evidence\rlt-v4-final250-20260824'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)
$files = [ordered]@{
    "$remoteRun/runtime/driver.log" = 'rlt_v4_driver.log'
    "$remoteRun/runtime/resource.csv" = 'rlt_v4_resource.csv'
    "$remoteRun/metrics.log" = 'rlt_v4_metrics.log'
    "$remoteRun/tensorboard/events.out.tfevents.1787542941.admin.1389995.0" = 'rlt_v4_events.out.tfevents.1787542941.admin.1389995.0'
    "$remoteRun/runtime/resolved.yaml" = 'rlt_v4_resolved.yaml'
    "$remoteRun/runtime/command.txt" = 'rlt_v4_command.txt'
    "$remoteRun/runtime/launch_manifest.txt" = 'rlt_v4_launch_manifest.txt'
    "$remoteRun/runtime/started_at.txt" = 'rlt_v4_started_at.txt'
    "$remoteRun/runtime/finished_at.txt" = 'rlt_v4_finished_at.txt'
    "$remoteRun/runtime/exit_code.txt" = 'rlt_v4_exit_code.txt'
}

if (Test-Path -LiteralPath $destination) {
    throw "Refusing to reuse final snapshot directory: $destination"
}
New-Item -ItemType Directory -Path $destination | Out-Null

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
