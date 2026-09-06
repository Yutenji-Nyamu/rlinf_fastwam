"""Apply Ruff's import ordering to the real full-update probe."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/embodiment/ogpo_real_fsdp_update_probe.py": (
        "e412eb52be401ea7716c6efb6793d7d696282a1950728ada8fc867d34c85df1b"
    ),
}


if __name__ == "__main__":
    sync.main()
