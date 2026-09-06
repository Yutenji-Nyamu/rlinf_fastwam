$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $out = 'tmp\rlt_dvac_pure03_pure04_live_refresh_20260829\raw'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    & $python 'local_scripts\remote_exec_autodl.py' run --command-file 'tmp\autodl_refresh_rlt_pure03_pure04_20260829.sh'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python 'local_scripts\remote_exec_autodl.py' get '/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/metrics.log' "$out\pure03_metrics.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python 'local_scripts\remote_exec_autodl.py' get '/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/metrics.log' "$out\pure04_metrics.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python 'local_scripts\remote_exec_autodl.py' get '/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_dual_single_gpu_formal480_20260829_v1/paired_resources.csv' "$out\paired_resources.csv"
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
