"""Apply the three Ruff import-spacing corrections."""

import sync_ogpo_incremental_001 as sync


sync.UPDATES = {
    "rlinf/data/ogpo_replay.py": (
        "ea6d263156738a1218d96ca71a64eea1c641877ed5f89f6972ffa7b703114dea"
    ),
    "rlinf/models/embodiment/openpi/openpi_action_model.py": (
        "05e84672088c220d385efa8f8eede96d2c055017625e311971c87984bc80a8c5"
    ),
    "rlinf/models/embodiment/openpi/openpi_ogpo.py": (
        "27403d007623a5e7e4ebd02cbdab4d09fdb5151ef586a2ed8a0f010269ba0aee"
    ),
}


if __name__ == "__main__":
    sync.main()
