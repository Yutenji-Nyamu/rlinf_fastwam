"""Keep PaliGemma's tied token weight in one FSDP ownership unit."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "rlinf/models/embodiment/modules/ogpo_modules.py": (
        "da807488e373f9c524659f34ed2df33cd4c243720c13b56c635d052ebe253bb9"
    ),
    "rlinf/models/embodiment/openpi/openpi_action_model.py": (
        "ade9d601a97858f4aa7fbf0409d6b38bbe6a08087a4e950d2bd3f787db0b6778"
    ),
    "tests/embodiment/test_openpi_ogpo_adapter.py": (
        "5d92b35f8b0412c999c13aee0ea1fa828fd9dfc4744e8527a56a78e51317367f"
    ),
}


if __name__ == "__main__":
    sync.main()
