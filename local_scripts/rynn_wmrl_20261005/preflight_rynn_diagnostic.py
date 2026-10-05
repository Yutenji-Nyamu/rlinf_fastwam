"""CPU-only entrypoint: same arguments as rynn_diagnostic.py, CUDA hidden.

Loads AutoConfig/AutoProcessor and fixed source metadata; never model weights.
The shared diagnostic implementation supplies the exact cases and assertions.
"""

import os
import sys


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Processor preflight requires CUDA_VISIBLE_DEVICES empty")
    from rynn_diagnostic import main
    if "--cpu-preflight" not in sys.argv:
        sys.argv.append("--cpu-preflight")
    main()
