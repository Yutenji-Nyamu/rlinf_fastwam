"""Run the existing RLinf converter in Dojo's JAX environment, CPU only.

Dojo's inference venv has Orbax but not the RL training dependencies. Loading
only the converter avoids importing RLinf's unrelated package initializers.
"""

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("JAX_PLATFORMS") == "cpu"
root = Path(__file__).resolve().parents[2]
for name in [
    "rlinf",
    "rlinf.utils",
    "rlinf.utils.ckpt_convertor",
    "rlinf.utils.ckpt_convertor.openpi",
]:
    module = types.ModuleType(name)
    module.__path__ = [str(root.joinpath(*name.split(".")))]
    sys.modules[name] = module
converter = importlib.import_module(
    "rlinf.utils.ckpt_convertor.openpi.jax_to_openpi_rlinf"
)
load_params = converter._load_jax_params


def unwrap_nnx(tree):
    # Dojo's Flax NNX snapshot wraps each parameter in {'value': array}.
    # Strip containers only; the RLinf tensor transforms remain unchanged.
    if isinstance(tree, dict):
        if set(tree) == {"value"}:
            return unwrap_nnx(tree["value"])
        return {key: unwrap_nnx(value) for key, value in tree.items()}
    return tree


converter._load_jax_params = lambda path: unwrap_nnx(load_params(path))
parser = argparse.ArgumentParser()
converter.add_arguments(parser)
converter.run(parser.parse_args())
print("CONVERSION_COMPLETE", flush=True)
