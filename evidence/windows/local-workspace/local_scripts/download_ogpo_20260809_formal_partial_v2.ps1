$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$destination = Join-Path $workspace 'exports\ogpo_formal_90k_partial_64078_20260809_v2'
$credentialSource = 'E:\Codex\home\attachments\a0e17fb0-c2f9-4800-b353-c3a3d2090c83\pasted-text.txt'
$runtime = '/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime'
$runRoot = '/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2'
$experiment = "$runRoot/robotwin_adjust_bottle_ogpo_ca_formal_90k_utd005_v2"
$eventName = 'events.out.tfevents.1786166872.autodl-container-nekaqbwt43-6ce5babb.99682.0'

$nonEmpty = @(Get-Content -LiteralPath $credentialSource | Where-Object { $_.Trim().Length -gt 0 })
$sshIndex = [Array]::IndexOf($nonEmpty, 'ssh -p 36406 root@connect.bjb1.seetacloud.com')
if ($sshIndex -lt 0 -or $sshIndex + 1 -ge $nonEmpty.Count) {
    throw 'Credential source format mismatch'
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
$downloads = @(
    @("$runtime/driver.log", 'driver.log'),
    @("$runtime/resources_1s.csv", 'resources_1s.csv'),
    @("$runtime/resolved.yaml", 'resolved.yaml'),
    @("$runtime/source_config.yaml", 'source_config.yaml'),
    @("$runtime/exact_command.txt", 'exact_command.txt'),
    @("$runtime/run_provenance.tsv", 'run_provenance.tsv'),
    @("$runtime/stop_conditions.txt", 'stop_conditions.txt'),
    @("$runtime/exit_code.txt", 'exit_code.txt'),
    @("$runtime/started_at.txt", 'started_at.txt'),
    @("$runtime/finished_at.txt", 'finished_at.txt'),
    @("$runtime/resources_before.txt", 'resources_before.txt'),
    @("$runtime/resources_after.txt", 'resources_after.txt'),
    @("$runRoot/metrics.log", 'metrics.log'),
    @("$runRoot/tensorboard/config.yaml", 'tensorboard_config.yaml'),
    @("$runRoot/tensorboard/$eventName", $eventName),
    @("$experiment/checkpoints/global_step_22/actor/ogpo_components/complete.json", 'checkpoint_30k_complete.json'),
    @("$experiment/checkpoints/global_step_43/actor/ogpo_components/complete.json", 'checkpoint_60k_complete.json')
)

try {
    $env:SEETA_SSH_PASSWORD = $nonEmpty[$sshIndex + 1].Trim()
    $env:PYTHONIOENCODING = 'utf-8'
    foreach ($item in $downloads) {
        $remote = $item[0]
        $local = Join-Path $destination $item[1]
        & $python $helper get $remote $local
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP get failed: $remote"
        }
        $file = Get-Item -LiteralPath $local
        $hash = (Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLowerInvariant()
        "GET_OK`t$($file.Length)`t$hash`t$remote`t$local"
    }
} finally {
    Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
}
