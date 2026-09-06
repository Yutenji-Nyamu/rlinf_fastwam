$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $PSScriptRoot '..\local_scripts\remote_exec_autodl.py'
$localRoot = Join-Path $PSScriptRoot 'rlt_dvac_pure_live_refresh_20260828\raw'
New-Item -ItemType Directory -Force -Path $localRoot | Out-Null

$items = @(
    @('/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/metrics.log', (Join-Path $localRoot 's0p5_metrics.log')),
    @('/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/metrics.log', (Join-Path $localRoot 's2p0_metrics.log')),
    @('/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1/paired_resources.csv', (Join-Path $localRoot 'paired_resources.csv'))
)

$securePassword = [REDACTED] 'AutoDL SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    foreach ($item in $items) {
        & $python $helper get $item[0] $item[1]
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}
