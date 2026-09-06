$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$target = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\fastwam-move-stapler-phase-candidates-20260822'
$remoteRoot = '/data/chenyiteng/results/dvac-observation/phase-candidates/fastwam-move_stapler_pad-p2-v1'
$names = @(
    'success_episode_id4_seed4300003_contact_sheet_12f.png',
    'failure_episode_id6_seed4300005_contact_sheet_12f.png'
)
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)

if (Test-Path -LiteralPath $target) {
    throw "Refusing to reuse local target: $target"
}
New-Item -ItemType Directory -Path $target | Out-Null

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    foreach ($name in $names) {
        & $python $helper @connection get "$remoteRoot/$name" (Join-Path $target $name)
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP get failed for $name with exit $LASTEXITCODE"
        }
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-ChildItem -LiteralPath $target -File | ForEach-Object {
    [pscustomobject]@{
        Name = $_.Name
        Length = $_.Length
        SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    }
}
