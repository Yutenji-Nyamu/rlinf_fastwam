"""Frozen pi0.5, fixed observations, two independent K8 banks and K costs."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from .common import Timer, cuda_guard, entropy, seed_for, write_json, write_npz


def main(args):
    torch = cuda_guard()
    sys.path.insert(0, args.model_source)
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model
    from tools.pi05_u_video.noise_samples import numpy_exact

    root = Path(args.data)
    out = root / "rt-k8"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "rt-pool/config.json").read_text())
    model = get_model(OmegaConf.create(cfg["model"])).cuda().eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    with np.load(root / "rt-pool/observations.npz", allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}

    def pack(ids, source=obs):
        result = {
            k: torch.from_numpy(v[ids].copy())
            for k, v in source.items()
            if k != "task_descriptions"
        }
        result["task_descriptions"] = source["task_descriptions"][ids].tolist()
        result["extra_view_images"] = None
        return result

    def predict(ob, noise):
        with torch.inference_mode():
            ae, r = model.predict_action_batch(
                ob,
                noise=noise.cuda().to(model.action_in_proj.weight.dtype),
                mode="eval",
                compute_values=False,
                num_steps=10,
                return_dvac_telemetry=True,
                record_signals=False,
            )
        trace = r["dvac_telemetry"]
        return numpy_exact(trace["final_model_action"]), numpy_exact(ae)

    # Replay the original native B16 path once before supplementary inference.
    batch = Path(json.loads((root / "rt-pool/queries.json").read_text())[0]["batch"])
    with np.load(batch / "query_000_obs.npz", allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    qm = json.loads((batch / "queries.json").read_text())
    if isinstance(qm, dict):
        qm = qm["queries"]
    old["task_descriptions"] = np.asarray(qm[0]["descriptions"])
    with np.load(batch / "query_000.npz", allow_pickle=False) as z:
        old_noise = z["noise4_initial_noise"][:, 0].copy()
        target = z["noise4_model_actions"][:, 0].copy()
        target_env = z["noise4_env_actions"][:, 0].copy()
    replay, physical = predict(
        pack(np.arange(len(old_noise)), old), torch.from_numpy(old_noise)
    )
    delta = float(np.max(np.abs(replay - target)))
    env_delta = float(np.max(np.abs(physical - target_env)))
    write_json(
        out / "replay-gate.json",
        dict(
            model_max_abs=delta,
            physical_max_abs=env_delta,
            exact=bool(np.array_equal(replay, target)),
            batch=16,
        ),
    )
    np.testing.assert_allclose(replay, target, rtol=0, atol=1e-5)
    np.testing.assert_allclose(physical, target_env, rtol=0, atol=1e-5)
    del replay, physical
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2**30
    with Timer() as probe:
        predict(
            pack(np.arange(8)),
            torch.randn(8, 50, 32, generator=torch.Generator().manual_seed(1701)),
        )
    per = max((probe.result["allocated_peak_gib"] - base) / 8, 0.05)
    free, total = torch.cuda.mem_get_info()
    target_gib = min(68.0, total / 2**30 - 12.0)
    micro = max(8, min(256, int((target_gib - base) / per * 0.90) // 8 * 8))
    write_json(
        out / "memory-plan.json",
        dict(
            base_gib=base,
            probe=probe.result,
            estimated_gib_per_sample=per,
            microbatch=micro,
            target_gib=target_gib,
        ),
    )
    cost = []
    for k in [1, 2, 4, 8]:
        noise = torch.randn(
            k, 50, 32, generator=torch.Generator().manual_seed(1900 + k)
        )
        predict(pack(np.zeros(k, dtype=int)), noise)
        for repeat in range(3):
            with Timer() as timer:
                predict(pack(np.zeros(k, dtype=int)), noise)
            cost.append(dict(kind="batched_K", K=k, B=1, repeat=repeat, **timer.result))
        if k > 1:
            for repeat in range(3):
                with Timer() as timer:
                    predict(pack(np.zeros(1, dtype=int)), noise[:1])
                    predict(pack(np.zeros(k - 1, dtype=int)), noise[1:])
                cost.append(
                    dict(
                        kind="preserved_main_then_extras",
                        K=k,
                        B=1,
                        repeat=repeat,
                        **timer.result,
                    )
                )
    write_json(out / "costs.json", cost)
    for bank in range(2):
        path = out / f"bank_{bank}.npz"
        if path.exists():
            continue
        ids = np.repeat(np.arange(64), 8)
        seeds = np.asarray(
            [seed_for(bank, int(o), s) for o in range(64) for s in range(8)]
        )
        noise = torch.stack(
            [
                torch.randn(50, 32, generator=torch.Generator().manual_seed(int(s)))
                for s in seeds
            ]
        )
        am, ae, timing = [], [], []
        lo = 0
        retried = False
        while lo < len(ids):
            hi = min(lo + micro, len(ids))
            try:
                with Timer() as timer:
                    m, e = predict(pack(ids[lo:hi]), noise[lo:hi])
            except torch.cuda.OutOfMemoryError:
                if retried or micro <= 8:
                    raise
                retried = True
                micro = max(8, micro // 2)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            am.append(m)
            ae.append(e)
            timing.append(dict(lo=lo, hi=hi, **timer.result))
            lo = hi
            write_json(
                out / "progress.json",
                dict(bank=bank, done=lo, total=len(ids), microbatch=micro),
            )
        models = np.concatenate(am).reshape(64, 8, 50, 14)
        write_npz(
            path,
            model_actions=models,
            env_actions=np.concatenate(ae).reshape(64, 8, 50, 14),
            initial_noise=noise.numpy().reshape(64, 8, 50, 32),
            seeds=seeds.reshape(64, 8),
            entropy8=entropy(models),
        )
        write_json(out / f"bank_{bank}-timing.json", timing)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            observations=64,
            banks=2,
            K=8,
            NFE=10,
            model_source=args.model_source,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-source", required=True)
    main(p.parse_args())
