"""Synchronize the reviewed OGPO closure fixes and their narrow fixtures."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml": (
        "fe08e0e8bacd91359f2ee35947d64a63f8d3ad0c216a31262c4ac7323909c193"
    ),
    "rlinf/algorithms/ogpo/core.py": (
        "f392aeb6233c4c6f18c46f9fa44b5ebed607dc17190ac756bafa7fc54d5ddcf7"
    ),
    "rlinf/config.py": (
        "ec0e4d4424ede823225663ac0a0ce873059d7833ed3c35d29fa87f2fbb5f83dc"
    ),
    "rlinf/runners/embodied_runner.py": (
        "d5a94aa982e2e0281634fcb9b4252648327136e592b57d6eb9c9992c79c37903"
    ),
    "rlinf/workers/actor/fsdp_ogpo_policy_worker.py": (
        "6f9f47de2224b60db02d5a3de4148033644ee4b84c86bca306d4a4542f14f9ce"
    ),
    "rlinf/workers/env/env_worker.py": (
        "902cbef322cd1e559534f80928fce63ef7eb68b62826e723f9e92512c53d290f"
    ),
    "tests/workers/test_ogpo_checkpoint_sidecar.py": (
        "8cc19a6508292a58d484bc1dcf74b0ff78de6319d01d2720db880674c8d9fc57"
    ),
    "tests/workers/test_ogpo_env_trace.py": "ABSENT",
    "tests/workers/test_ogpo_row_schedule.py": "ABSENT",
}


if __name__ == "__main__":
    sync.main()
