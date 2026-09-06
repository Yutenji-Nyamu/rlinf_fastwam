"""Upload the bounded real two-rank production full-update probe."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/embodiment/ogpo_real_fsdp_update_probe.py": "ABSENT",
}


if __name__ == "__main__":
    sync.main()
