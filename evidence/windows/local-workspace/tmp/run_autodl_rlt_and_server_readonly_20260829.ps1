$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $helper = 'local_scripts\remote_exec_autodl.py'

    & $python $helper run --command-file 'tmp\autodl_health_rlt_pure_dual_formal480_compact_20260828.sh'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $helper run --command-file 'tmp\autodl_server_overall_readonly_20260829.sh'
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
