param(
    [Parameter(Mandatory = $true)]
    [string]$LocalPath,
    [Parameter(Mandatory = $true)]
    [string]$RemotePath
)

$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $PSScriptRoot '..\local_scripts\remote_exec_autodl.py'
$securePassword = [REDACTED] 'AutoDL SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    & $python $helper put $LocalPath $RemotePath
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

