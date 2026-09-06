"""Synchronize the FSDP-aligned EMA correction and its two-rank fixture."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "rlinf/models/embodiment/modules/ogpo_modules.py": (
        "da807488e373f9c524659f34ed2df33cd4c243720c13b56c635d052ebe253bb9"
    ),
    "tests/embodiment/ogpo_fsdp_ema_fixture.py": (
        "2d23c892ca580eefdacea6402f90f44dfbb70f0b4678ec9e5809a72137d4e5b5"
    ),
}


if __name__ == "__main__":
    sync.main()
