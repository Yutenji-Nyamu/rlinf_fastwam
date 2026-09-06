"""Add the critic optimizer contract fixture after the closure sync."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/workers/test_ogpo_checkpoint_sidecar.py": (
        "2f73064b840c33fe9d1961f3700831550beb67034b1c0c4cce8b1ede94514fdd"
    ),
}


if __name__ == "__main__":
    sync.main()
