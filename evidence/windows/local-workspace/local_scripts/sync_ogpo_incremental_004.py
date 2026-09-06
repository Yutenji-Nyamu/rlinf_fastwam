"""Add the configured B4 x G8 candidate microbatch to the real FSDP probe."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "tests/embodiment/ogpo_real_fsdp_ema_probe.py": (
        "cae423c0b792f800bc40f908ceac8b496ba745ad7d4dc1c17044d0da77f258f5"
    ),
}


if __name__ == "__main__":
    sync.main()
