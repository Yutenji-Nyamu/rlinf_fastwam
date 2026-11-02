"""Supplement K3 teacher predictions from accepted saved observations/noises.

Uses the exact deployed model source/config. Does not create a simulator or
alter the old batch. Each GPU owns disjoint queries. Main-noise replay is the
smoke gate before the three additional predictions.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_observation(path):
    import torch

    result = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            # The accepted writer serializes this known unused None as object.
            if key == "extra_view_images":
                result[key] = None
            elif key == "task_descriptions":
                result[key] = z[key].tolist()
            else:
                result[key] = torch.from_numpy(z[key].copy())
    return result


def main(config):
    c = json.loads(Path(config).read_text())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(c["gpu"])
    source = Path(c["model_source"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "tools/pi05_u_video"))
    from noise_samples import numpy_exact, repeated_observation
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    batch, output = Path(c["input_batch"]), Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((batch / "config.json").read_text())
    model = get_model(OmegaConf.create(original["model"]))
    state = torch.load(
        original["strong_weights"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval()
    timings = []

    def predict(obs, noise):
        with torch.inference_mode():
            env_action, result = model.predict_action_batch(
                obs,
                mode="eval",
                compute_values=False,
                num_steps=original["num_steps"],
                noise=noise,
                return_dvac_telemetry=True,
            )
            trace = result["dvac_telemetry"]
            np.testing.assert_array_equal(
                numpy_exact(noise), numpy_exact(trace["initial_noise_full"])
            )
            return numpy_exact(trace["final_model_action"]), numpy_exact(env_action)

    for q in c["queries"]:
        name = f"query_{q:03d}.npz"
        obs = load_observation(batch / f"observation_{q:03d}.npz")
        with np.load(batch / name, allow_pickle=False) as z:
            noise = z["noise4_initial_noise"].copy()
            reference = z["teacher_model_actions"].copy()
            main_env = z["teacher_env_actions"].copy()
        torch.cuda.synchronize()
        start = time.perf_counter()
        main, _ = predict(obs, torch.from_numpy(noise[:, 0]).cuda())
        if not np.array_equal(main, reference):
            delta = float(np.abs(main - reference).max())
            raise RuntimeError(
                f"Original teacher replay differs; max={delta}; no supplement accepted"
            )
        b = len(noise)
        flat = np.concatenate([noise[:, k] for k in (1, 2, 3)])
        repeated = repeated_observation(obs, 3)
        am, ae = [], []
        for lo in range(0, len(flat), c.get("extra_batch", 96)):
            hi = min(lo + c.get("extra_batch", 96), len(flat))
            chunk = {
                k: v[lo:hi] if v is not None else None for k, v in repeated.items()
            }
            m, e = predict(chunk, torch.from_numpy(flat[lo:hi]).cuda())
            am.append(m)
            ae.append(e)
        model4 = np.stack(
            [reference, *np.concatenate(am).reshape(3, b, 50, 14)], axis=1
        )
        env4 = np.stack([main_env, *np.concatenate(ae).reshape(3, b, 50, 14)], axis=1)
        torch.cuda.synchronize()
        row = dict(
            query=q,
            seconds=time.perf_counter() - start,
            exact_original=True,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            extra_batch=c.get("extra_batch", 96),
        )
        temporary = output / (name + ".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(
                f, teacher4_model_actions=model4, teacher4_env_actions=env4
            )
        temporary.replace(output / name)
        timings.append(row)
        (output / f"progress-g{c['gpu']}.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)
    receipt = dict(
        passed=True,
        gpu=c["gpu"],
        queries=timings,
        input_batch=str(batch),
        checkpoint=original["strong_weights"],
        model_source=str(source),
        noise_source="unchanged noise4_initial_noise in input batch",
        completed=time.time(),
    )
    (output / f"receipt-g{c['gpu']}.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
