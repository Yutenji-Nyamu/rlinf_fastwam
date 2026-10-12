"""Export existing final-denoising action scalars, without loading raw Q/K."""

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = ("visual", "instruction", "state", "action")
SCALARS = [f"{prefix}_{group}" for prefix in ("E", "M") for group in GROUPS]
SCALARS += ["M_other", "E_obs", "dv", "action_amplitude", "gripper_left", "gripper_right"]


def export(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    files, valid_batches, samples = [], [], []
    manifest = json.loads((root / "manifest.json").read_text())
    for config_path in manifest["configs"]:
        cfg = json.loads(Path(config_path).read_text())
        batch = Path(cfg["output"])
        validation_path = batch / "validation.json"
        if not validation_path.exists():
            continue
        validation = json.loads(validation_path.read_text())
        if not (validation.get("passed") and validation.get("video_passed")):
            continue
        rows = json.loads((batch / "atlas-data.json").read_text())
        episodes = json.loads((batch / "episodes.json").read_text())
        arrays = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in SCALARS}
        for key in ("query", "slot", "requested_seed", "actual_seed"):
            arrays[key] = np.asarray([row[key] for row in rows], dtype=np.int64)
        for key in ("native_retry", "success", "executed_known", "submitted"):
            arrays[key] = np.asarray([row[key] for row in rows], dtype=bool)
        entropy = np.stack([arrays[f"E_{g}"] for g in GROUPS], axis=-1)
        mass = np.stack([arrays[f"M_{g}"] for g in GROUPS], axis=-1)
        counts = np.asarray([[row["source_token_counts"][g] for g in GROUPS] for row in rows])
        valid = np.isfinite(entropy) & (counts[:, None, :] > 1) & (mass > 0)
        valid_mass = np.where(valid, mass, 0).sum(-1)
        numerator = np.where(valid, mass * entropy, 0)
        contribution = np.divide(numerator, valid_mass[..., None], out=np.full_like(numerator, np.nan), where=valid_mass[..., None] > 0)
        arrays.update(source_token_counts=counts, valid_normalized_entropy=valid,
                      valid_mass=valid_mass, E_weighted=contribution.sum(-1))
        for i, group in enumerate(GROUPS):
            arrays[f"ME_{group}"] = numerator[..., i]
            arrays[f"contribution_{group}"] = contribution[..., i]
        assert np.nanmin(arrays["E_weighted"]) >= -1e-6
        assert np.nanmax(arrays["E_weighted"]) <= 1 + 1e-6
        name = f"{cfg['task']}_b{cfg['batch']}"
        dest = output / f"{name}.npz"
        np.savez_compressed(dest, **arrays)
        record = dict(task=cfg["task"], batch=cfg["batch"], file=dest.name,
                      bytes=dest.stat().st_size, rows=len(rows), episodes=episodes,
                      validation=validation, source=str(batch))
        files.append(record)
        valid_batches.append(dict(task=cfg["task"], batch=cfg["batch"], path=str(batch),
                                  episodes=episodes, validation=validation))
        known = arrays["executed_known"][:, None] & arrays["submitted"]
        samples.append({key: arrays[key][known] for key in SCALARS + ["E_weighted", "valid_mass"]})
    population = {}
    for key in SCALARS + ["E_weighted", "valid_mass"]:
        values = np.concatenate([s[key] for s in samples])
        population[key] = dict(finite_fraction=float(np.isfinite(values).mean()),
                               q05_q50_q95=np.nanquantile(values, [.05, .5, .95]).tolist())
    count = sum(len(b["episodes"]) for b in files)
    index = dict(schema=1, denoising_round=10, action_horizon=50,
                 precision='E and modal M reuse the accepted atlas rounded to 6 decimals; original FP32 all-round scores remain on the server.',
                 groups=GROUPS, episodes=count, tasks=len({b['task'] for b in files}),
                 unique_task_seed_pairs=len({(b['task'],e['actual']) for b in files for e in b['episodes']}),
                 task_successes=sum(e['success'] for b in files for e in b['episodes']),
                 known_action_positions=sum(len(s['E_weighted']) for s in samples),
                 bytes=sum(f['bytes'] for f in files), population=population, files=files,
                 deferred_tasks=['move_stapler_pad', 'place_fan', 'place_mouse_pad'],
                 definitions=dict(E='within-source H/log(N), N>1; invalid is NaN',
                                  M='read mass under the original full attention distribution',
                                  ME='M*E, zero only for invalid aggregation contribution',
                                  contribution='valid M*E / sum(valid M)',
                                  E_weighted='sum(contribution); not joint token entropy',
                                  row='one active query/episode; each curve has 50 predicted action positions',
                                  execution='submitted and executed_known are separate; terminal prefix is unknown',
                                  video='query q is bracketed by frames q and q+1, not action-resolved video'))
    (output / 'index.json').write_text(json.dumps(index, indent=2, allow_nan=False))
    return index, valid_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    index, _ = export(args.root, args.output)
    print(json.dumps({k: v for k, v in index.items() if k not in ['files', 'population']}))
