"""Sync the FSDP-safe pi0 success-BC route and its focused fixture."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "rlinf/models/embodiment/openpi/openpi_ogpo.py": (
        "4eeccbd49f2a85ec94d87acea36440579b6289adf34a73269edc7b4c8a4a42f3"
    ),
    "rlinf/models/embodiment/openpi/openpi_action_model.py": (
        "d4d433cab8131076acc9415bb78458b381eff1e1b7aba37f96ded68abb8371af"
    ),
    "tests/embodiment/test_openpi_ogpo_adapter.py": (
        "802febe6b9a8b16f47aad751fe8ef54e657df7a3de267bb703328778e5d8b089"
    ),
}


if __name__ == "__main__":
    sync.main()
