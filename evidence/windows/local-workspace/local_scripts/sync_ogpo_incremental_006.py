"""Assert the real FSDP ownership split for PaliGemma's tied weight."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/embodiment/ogpo_real_fsdp_ema_probe.py": (
        "c30543de5a2717aad7d84e8e4f68670711a0a23c4c84e0bb6f00648b7b1b2845"
    ),
}


if __name__ == "__main__":
    sync.main()
