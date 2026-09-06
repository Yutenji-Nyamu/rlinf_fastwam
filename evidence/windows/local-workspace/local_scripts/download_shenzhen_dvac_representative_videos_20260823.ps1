$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\86136\Documents\rl'
$python = 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$helper = Join-Path $workspace 'local_scripts\remote_exec_autodl.py'
$destination = Join-Path $workspace 'docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-detailed-action-timeline-20260823\videos'
$connection = @(
    '--host', '120.241.223.9',
    '--port', '22',
    '--user', 'chenyiteng',
    '--host-key-sha256', 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'
)

$items = @(
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2/adjust_bottle/episode1_randomized-false_success-true.mp4', 'fastwam_adjust_bottle_success_episode0002_reset1.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode3_randomized-false_success-true.mp4', 'fastwam_move_stapler_success_episode0004_reset3.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode5_randomized-false_success-false.mp4', 'fastwam_move_stapler_failure_episode0006_reset5.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-turn_switch-p2-16ep-c63dc9b5-v1/turn_switch/episode0_randomized-false_success-false.mp4', 'fastwam_turn_switch_failure_episode0001_reset0.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-turn_switch-p2-16ep-c63dc9b5-v1/turn_switch/episode8_randomized-false_success-true.mp4', 'fastwam_turn_switch_success_episode0009_reset8.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1/pick_diverse_bottles/episode3_randomized-false_success-true.mp4', 'fastwam_pick_bottles_success_episode0004_reset3.mp4'),
    @('/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1/pick_diverse_bottles/episode4_randomized-false_success-false.mp4', 'fastwam_pick_bottles_failure_episode0005_reset4.mp4')
)

New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($item in $items) {
    $local = Join-Path $destination $item[1]
    if (Test-Path -LiteralPath $local) {
        throw "Refusing to overwrite existing representative video: $local"
    }
}

$securePassword = [REDACTED] 'SZ-H100 SSH password' -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $env:SEETA_SSH_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    foreach ($item in $items) {
        $local = Join-Path $destination $item[1]
        & $python $helper @connection get $item[0] $local
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP get failed with exit $LASTEXITCODE for $($item[0])"
        }
    }
}
finally {
    Remove-Item Env:\SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}

Get-ChildItem -LiteralPath $destination -File | Sort-Object Name | Select-Object Name, Length
