$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$remote = '/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1.tar.gz'
$local = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-analysis-fastwam-move-stapler-phase-20260823.tar.gz'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)

if (Test-Path -LiteralPath $local) {
    throw "Refusing to overwrite local archive: $local"
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    & $python $helper @connection get $remote $local
    if ($LASTEXITCODE -ne 0) {
        throw "SFTP get failed with exit $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-Item -LiteralPath $local | Select-Object FullName, Length
Get-FileHash -Algorithm SHA256 -LiteralPath $local
