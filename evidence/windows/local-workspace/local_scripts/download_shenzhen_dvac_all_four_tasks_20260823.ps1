$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$remoteArchive = '/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-four-tasks-v1.tar.gz'
$localArchive = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-analysis-all-four-tasks-20260823.tar.gz'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)

if (Test-Path -LiteralPath $localArchive) {
    throw "Refusing to overwrite local archive: $localArchive"
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    & $python $helper @connection get $remoteArchive $localArchive
    if ($LASTEXITCODE -ne 0) {
        throw "SFTP get failed with exit $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-Item -LiteralPath $localArchive | Select-Object FullName, Length
Get-FileHash -Algorithm SHA256 -LiteralPath $localArchive
