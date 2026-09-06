$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$csv = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-analysis-all-four-tasks-20260823\query_metrics.csv'
$destination = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-detailed-action-timeline-20260823\pi0-query-images'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)

$episodes = @('ep000033_reset100100005', 'ep000050_reset100100017')
$rows = Import-Csv -LiteralPath $csv |
    Where-Object { $_.episode_uid -in $episodes } |
    Sort-Object episode_uid, @{ Expression = { [int]$_.query_idx } }

if ($rows.Count -ne 8) {
    throw "Expected 8 representative pi0 query rows, got $($rows.Count)"
}

$items = foreach ($row in $rows) {
    foreach ($camera in @('head', 'left', 'right')) {
        $column = if ($camera -eq 'head') { 'head_image_path' } elseif ($camera -eq 'left') { 'left_image_path' } else { 'right_image_path' }
        $remote = $row.$column
        $name = '{0}__q{1:d2}__{2}.png' -f $row.episode_uid, [int]$row.query_idx, $camera
        [pscustomobject]@{ Remote = $remote; Local = Join-Path $destination $name }
    }
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($item in $items) {
    if (Test-Path -LiteralPath $item.Local) {
        throw "Refusing to overwrite existing pi0 query image: $($item.Local)"
    }
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    foreach ($item in $items) {
        & $python $helper @connection get $item.Remote $item.Local
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP get failed with exit $LASTEXITCODE for $($item.Remote)"
        }
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-ChildItem -LiteralPath $destination -File | Sort-Object Name | Select-Object Name, Length
