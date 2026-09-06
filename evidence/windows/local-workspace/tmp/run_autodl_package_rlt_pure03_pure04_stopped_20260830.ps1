$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $helper = 'local_scripts\remote_exec_autodl.py'
    $remote = '/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_stopped_high_info_20260830_v2.tar.gz'
    $local = 'tmp\rlt_dvac_pure03_pure04_stopped_high_info_20260830_v2.tar.gz'

    & $python $helper run --command-file 'tmp\autodl_package_rlt_pure03_pure04_stopped_20260830.sh'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $helper get $remote $local
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
