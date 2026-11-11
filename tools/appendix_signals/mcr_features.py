"""Official MCR representation and GR-BC gates on frozen-state light pairs."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .common import Timer, cuda_guard, sha256, write_json, write_npz


def main(args):
    torch = cuda_guard()
    import pandas as pd
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet50

    out = Path(args.data) / "grbc"
    nodes = json.loads((out / "nodes.json").read_text())
    model = resnet50(weights=None)
    model.fc = torch.nn.Identity()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False).cuda()
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    tensors = []
    grips = []
    for node in nodes:
        with np.load(node["raw"], allow_pickle=False) as z:
            images = z["images"][node["appearance"]]
            state = z["state"]
        tensors.extend(transform(Image.fromarray(rgb)) for rgb in images)
        grips.append([bool(state[6] < 0.5), bool(state[13] < 0.5)])
    encoded = []
    timings = []
    for lo in range(0, len(tensors), 96):
        with Timer() as timer, torch.inference_mode():
            value = (
                model(torch.stack(tensors[lo : lo + 96]).cuda()).float().cpu().numpy()
            )
        encoded.append(value)
        timings.append(timer.result)
    raw = np.concatenate(encoded).reshape(len(nodes), 3, 2048)
    per_view = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-8)
    flat = raw.reshape(len(nodes), 6144).copy()
    flat /= np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-8)
    write_npz(
        out / "embeddings.npz",
        raw=raw,
        per_view=per_view,
        concat_normalized=flat,
        gripper_closed=np.asarray(grips),
    )
    spec = importlib.util.spec_from_file_location(
        "upstream_grbc_graph", args.graph_source
    )
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    reports = []
    triplets = []
    for task in sorted({n["task"] for n in nodes}):
        ids = [i for i, n in enumerate(nodes) if n["task"] == task]
        ep_keys = sorted({(nodes[i]["seed"], nodes[i]["appearance"]) for i in ids})
        table = pd.DataFrame(
            [
                dict(
                    node_id=j,
                    episode_index=ep_keys.index(
                        (nodes[i]["seed"], nodes[i]["appearance"])
                    ),
                    frame_index=nodes[i]["query"] * 50,
                )
                for j, i in enumerate(ids)
            ]
        )
        edge_sets = []
        stats = []
        for hand in range(2):
            edges, report = graph.build_multi_view_adaptive_similarity_edges(
                flat[ids],
                table,
                np.asarray(grips)[ids, hand],
                mutual_k=20,
                stride=15,
                adaptive_alpha=1.0,
                adaptive_base_scale=1e-4,
                cost_scale=1.0,
                camera_count=3,
                min_camera_hits=3,
            )
            edge_sets.append(
                {(min(ids[e[0]], ids[e[1]]), max(ids[e[0]], ids[e[1]])) for e in edges}
            )
            stats.append(report)
        both = edge_sets[0] & edge_sets[1]
        reports.append(
            dict(
                task=task,
                both_gripper_gate_edges=[list(x) for x in sorted(both)],
                per_hand=stats,
            )
        )
        for i in ids:
            if nodes[i]["appearance"] != 0:
                continue
            j = next(
                x
                for x in ids
                if nodes[x]["pair"] == nodes[i]["pair"] and nodes[x]["appearance"] == 1
            )
            negatives = [
                x
                for x in ids
                if nodes[x]["seed"] == nodes[i]["seed"]
                and nodes[x]["appearance"] == 0
                and nodes[x]["query"] != nodes[i]["query"]
                and grips[x] == grips[i]
            ]
            dpos = np.linalg.norm(per_view[i] - per_view[j], axis=-1)
            for k in negatives:
                dneg = np.linalg.norm(per_view[i] - per_view[k], axis=-1)
                triplets.append(
                    dict(
                        anchor=i,
                        same_state_other_light=j,
                        other_time_same_light=k,
                        positive_per_view=dpos.tolist(),
                        negative_per_view=dneg.tolist(),
                        positive_mean=float(dpos.mean()),
                        negative_mean=float(dneg.mean()),
                        positive_farther=bool(dpos.mean() > dneg.mean()),
                        paired_edge_retained=(min(i, j), max(i, j)) in both,
                        negative_event="different time, semantic event requires review",
                    )
                )
    write_json(out / "graph-gates.json", reports)
    write_json(out / "triplets.json", triplets)
    write_json(
        out / "features-complete.json",
        dict(
            nodes=len(nodes),
            timings=timings,
            mcr_sha256=sha256(args.checkpoint),
            graph_sha256=sha256(args.graph_source),
            preprocess="Resize256-CenterCrop224-ToTensor, no ImageNet normalization",
            graph_scope="small sparse diagnostic pool; actual upstream gates, not a full training graph or a GR-BC success-rate test",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph-source", required=True)
    main(p.parse_args())
