"""Remove the unrelated clean-50 label from the OGPO source fingerprint."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml": (
        "c94e9dc9295ab9e2218108af18c779dd346d932dc8cd4db75d95ce8299aec2fc"
    ),
}


if __name__ == "__main__":
    sync.main()
