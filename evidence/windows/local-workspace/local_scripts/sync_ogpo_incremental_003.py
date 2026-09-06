"""Restore the minimal EMA module and correct the FSDP fixture wrap policy."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "rlinf/models/embodiment/modules/ogpo_modules.py": (
        "1e9247644f0bf59da683ec5280881b660ee37914e4b1a64007c540bd8539919b"
    ),
    "tests/embodiment/ogpo_fsdp_ema_fixture.py": (
        "ec07b52685da98ba6797299a4b9102f4d19bd73a606d2029f1a8f6e07650f733"
    ),
}


if __name__ == "__main__":
    sync.main()
