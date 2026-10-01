$ErrorActionPreference = 'Stop'
$taskRoot = 'C:/Users/86136/Documents/rl'
$stepRoot = Join-Path $taskRoot 'local_logs/wan-goal-20261001/steps'
$evidenceRoot = Join-Path $taskRoot 'docs/experiments/wan-goal-sz3-20260930/resume-20261001/monitor-repair-r6'
$labels = @('w157-user-wm-mechanism-status','w158-user-wm-card-reward-audit','w159-user-wm-current-status','w160-user-wm-role-algorithm-source')
$receipts = [ordered]@{}
$reports = [ordered]@{}
foreach ($label in $labels) {
    $receipt = Get-Content -Raw -LiteralPath (Join-Path $stepRoot ($label+'/receipt.json')) | ConvertFrom-Json
    if ($receipt.exit_code -ne 0 -or -not $receipt.identity_verified -or -not $receipt.host_key_verified) { throw 'Unverified step' }
    $receipts[$label] = $receipt
    $reports[$label] = Get-Content -Raw -LiteralPath (Join-Path $stepRoot ($label+'/stdout.log')) | ConvertFrom-Json
}
$current = $reports[$labels[2]]
if ($current.completed_runner_epochs -ne 10 -or $current.effective_update_steps.Count -ne 7 -or -not $current.owner_alive -or $current.pipeline_phase -ne 'RUNNING_WM') { throw 'Unexpected snapshot' }
$metrics = [ordered]@{}
foreach ($property in $current.metrics.PSObject.Properties) {
    $metrics[$property.Name] = @($property.Value | Select-Object -Last 1)
}
$snapshots = @()
foreach ($index in @(1,3)) {
    $report = $reports[$labels[$index]]
    $cards = [ordered]@{}
    foreach ($property in $report.cards.PSObject.Properties) {
        $card = $property.Value
        if ($card.processes.Count -ne 3) { throw 'Unexpected process count' }
        $roles = @()
        foreach ($process in $card.processes) {
            if (-not $process.owned_catalog -or $process.uid -ne 20001) { throw 'Unverified context identity' }
            $role = if ($process.title -match 'EmbodiedFSDPActor') {'actor'} elseif ($process.title -match 'MultiStepRolloutWorker') {'rollout'} elseif ($process.title -match 'EnvWorker') {'env'} else {throw 'Unknown role'}
            $roles += [ordered]@{role=$role;memory_used_mib=[int]$process.memory_used_mib;current_catalog_identity_verified=$true}
        }
        $cards[$property.Name] = [ordered]@{memory_total_mib=$card.memory_total_mib;memory_used_mib=$card.memory_used_mib;utilization_pct=$card.utilization_pct;env_slots=16;group_size=8;concurrent_groups=2;roles=$roles}
    }
    $snapshots += [ordered]@{step=$labels[$index];time=$report.time;phase=$(if($index -eq 1){'actor_training'}else{'rollout'});cards=$cards;read_only=$true;resource_actions=@()}
}
$sourceHashes = [ordered]@{}
foreach ($property in $reports[$labels[3]].sources.PSObject.Properties) {
    $relative = $property.Name.Substring('/data/chenyiteng/projects/wan-goal-sz3/'.Length)
    $sourceHashes[$relative] = $property.Value.sha256
}
$proof = [ordered]@{
    schema=1; description='Fixed-source mechanism and phase-specific four-GPU snapshots; not peak measurements or physical LIBERO evaluation'
    host='sz3';run='wan-goal-sz3-20261001-r6/pi05-formal';time=$current.time
    source_revision='d34d4c320d08cb982de034aa9a011f08dc0fa217';physical_gpus=@(4,5,6,7)
    owner_alive=$true;pipeline_phase=$current.pipeline_phase;completed_runner_epochs=10;epochs_with_effective_grpo_gradient=$current.effective_update_steps;all_filtered_runner_epochs=@(3,4,5)
    latest_metrics=$metrics;recent_rollout_progress=$current.recent_rollout_progress
    protocol=$current.protocol;resolved_config_sha256=$current.resolved_config_sha256;source_yaml_sha256=$current.source_yaml_sha256
    per_runner_epoch=[ordered]@{trajectory_slots=512;action_slots=163840;chunk_slots=20480;groups_per_wave=8;waves=8;optimizer_calls=10;rank_chunk_slots=5120;rank_update_batch=512;micro_batch=128;gradient_accumulations=4;slot_counts_are_not_valid_sample_counts=$true}
    source_hashes=$sourceHashes;phase_snapshots=$snapshots
    checkpoint_directories=$current.checkpoint_directories;monitor_diagnostics=$current.monitor_diagnostics;recent_primary_error_lines=$current.recent_primary_error_lines
    restoration=[ordered]@{after_wm='RLT_DIRECT';resume_dojo=$false;existing_unique_owner=$true}
    receipts=$receipts;training_or_owner_source_changes=@();resource_actions=@()
}
New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null
$target = Join-Path $evidenceRoot 'wm-mechanism-parallel-live.json'
if (Test-Path -LiteralPath $target) { throw 'Evidence already exists; do not replace snapshot' }
[IO.File]::WriteAllText($target, ($proof | ConvertTo-Json -Depth 24)+"`n", [Text.UTF8Encoding]::new($false))
Write-Output ([ordered]@{path=$target;bytes=(Get-Item -LiteralPath $target).Length;identity_checks_passed=$true;training_changed=$false} | ConvertTo-Json -Compress)
