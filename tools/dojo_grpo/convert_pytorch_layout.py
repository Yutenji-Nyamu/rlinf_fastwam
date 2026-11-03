"""Finish conversion for the proven pre-OpenPI_RLinf training runtime.

The native embedding is tied to the language head; explicitly serialize both
aliases for this Transformers version. The action-expert language head is not
called by continuous action prediction (the upstream converter uses a reference
only for that unused module). All active weights come from the Dojo checkpoint.
"""

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch

from rlinf.utils.ckpt_convertor.openpi.openpi_rlinf_to_openpi_pytorch import (
    ACTION_EXPERT_LM_HEAD,
    new_to_old_state_dict,
)


def main():
    p = argparse.ArgumentParser()
    for key in ["source", "reference", "output"]:
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    source = Path(a.source)
    out = Path(a.output)
    original = safetensors.torch.load_file(str(source / "model.safetensors"))
    weights = new_to_old_state_dict(original, action_expert_uses_adarms=True)
    weights[
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    ] = original["llm.embedder.embedding.weight"].clone()
    ref = safetensors.torch.load_file(str(Path(a.reference) / "model.safetensors"))
    weights[ACTION_EXPERT_LM_HEAD] = ref[ACTION_EXPERT_LM_HEAD].clone()
    assert set(weights) == set(ref), (set(ref) - set(weights), set(weights) - set(ref))
    assert all(weights[k].shape == ref[k].shape for k in weights)
    out.mkdir(parents=True, exist_ok=True)
    # Match the accepted inference runtime's BF16 backbone and FP32 action/time
    # projections; don't unnecessarily quantize the small action projections.
    converted = {k: v.contiguous().to(dtype=ref[k].dtype) for k, v in weights.items()}
    safetensors.torch.save_file(converted, str(out / "model.safetensors"))
    shutil.copytree(source / "assets", out / "assets", dirs_exist_ok=True)
    shutil.copy2(source / "config.json", out / "config.json")
    (out / "conversion-receipt.json").write_text(
        json.dumps(
            {
                "active_source": str(source),
                "key_count": len(weights),
                "missing": [],
                "shape_mismatches": [],
                "reference_only_unused_module": ACTION_EXPERT_LM_HEAD,
                "tied_embedding_serialized": True,
            },
            indent=2,
        )
    )
    print("PYTORCH_LAYOUT_COMPLETE", len(weights), flush=True)


if __name__ == "__main__":
    main()
