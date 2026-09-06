$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $helper = 'local_scripts\remote_exec_autodl.py'
    $uploads = @(
        @('tmp\autodl_run_one_rlt_pure_formal480_20260828.sh','/tmp/autodl_run_one_rlt_pure_formal480_20260828.sh'),
        @('tmp\autodl_start_shared_ray_pure_formal480_20260828.sh','/tmp/autodl_start_shared_ray_pure_formal480_20260828.sh'),
        @('tmp\autodl_launch_rlt_pure03_pure04_dual_single_gpu_formal480_20260829.sh','/tmp/autodl_launch_rlt_pure03_pure04_dual_single_gpu_formal480_20260829.sh'),
        @('tmp\autodl_health_rlt_pure03_pure04_compact_20260829.sh','/tmp/autodl_health_rlt_pure03_pure04_compact_20260829.sh')
    )
    foreach ($item in $uploads) {
        & $python $helper put $item[0] $item[1]
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $python $helper run --command-file 'tmp\autodl_invoke_rlt_pure03_pure04_formal480_20260829.sh'
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
