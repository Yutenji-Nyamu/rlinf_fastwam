$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $helper = 'local_scripts\remote_exec_autodl.py'
    $raw = 'tmp\rlt_dvac_pure_live_refresh_20260828\raw'
    New-Item -ItemType Directory -Force -Path $raw | Out-Null

    & $python $helper run --command-file 'tmp\autodl_health_rlt_pure_dual_formal480_compact_20260828.sh'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $helper run --command-file 'tmp\autodl_eta_rlt_pure_dual_20260828.sh'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python $helper get '/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/metrics.log' "$raw\s0p5_metrics.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $helper get '/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/metrics.log' "$raw\s2p0_metrics.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $helper get '/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1/paired_resources.csv' "$raw\paired_resources.csv"
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
