$ErrorActionPreference = 'Stop'

$securePassword = [REDACTED] 'AutoDL password' -AsSecureString
$passwordPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPtr)
    $python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $helper = 'local_scripts\remote_exec_autodl.py'
    $localConfig = 'tmp\rlt_dvac_impl_source\examples\embodiment\config'
    $remoteConfig = '/root/autodl-tmp/RLinf_rlt_dvac_pure/examples/embodiment/config'
    $names = @(
        'robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p0_mb256_warm20k_replay80k_fresh480.yaml',
        'robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p5_mb256_warm20k_replay80k_fresh480.yaml',
        'robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p5_mb256_warm20k_replay80k_gpu1_fresh480.yaml'
    )
    foreach ($name in $names) {
        & $python $helper put "$localConfig\$name" "$remoteConfig/$name"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $python $helper run --command-file 'tmp\autodl_add_pure03_pure04_configs_commit_20260829.sh'
    exit $LASTEXITCODE
}
finally {
    $env:SEETA_SSH_PASSWORD = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPtr)
}
