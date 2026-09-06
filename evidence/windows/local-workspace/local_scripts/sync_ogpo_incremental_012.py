"""Align the full-update probe with production FSDP strategy and optimizer warmup."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/embodiment/ogpo_real_fsdp_update_probe.py": (
        "445d8ea3f41128d2bb34fd18e5512f4097fa6cfb2e31c1a3582a542228ca4d12"
    ),
}


if __name__ == "__main__":
    sync.main()
