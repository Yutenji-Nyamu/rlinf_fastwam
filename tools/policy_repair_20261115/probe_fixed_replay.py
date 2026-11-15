"""CPU-only candidate-coverage response on real checkpoint/replay inputs.

Both branches start with the same Q and replay samples. Actor, encoders, samples,
and replay-Q anchor are frozen; only Q heads receive 40 short updates. This tests
whether adding actual-policy candidates corrects their relative score, independently
of rollout noise or changing bootstrap targets. No training checkpoint is overwritten.
"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

os.environ.update(
    CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", USE_TF="0"
)

import torch
import torch.nn.functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cards", default="4,5,7")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--weight", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    snapshot = json.loads(Path(args.snapshot).read_text())
    loss_fn = load_module(args.helper, "coverage_helper").conservative_q_loss
    result = {
        "method": "frozen-encoder/actor/replay-anchor CPU Q-head response",
        "steps": args.steps,
        "weight": args.weight,
        "runs": {},
    }
    for card in args.cards.split(","):
        torch.manual_seed(1511)
        row = snapshot["runs"][card]
        root = Path(row["source"])
        checkpoint = Path(row["checkpoints"][-1]["path"])
        modules = root / "rlinf/models/embodiment/modules"
        compact = load_module(modules / "compact_encoders.py", "compact_encoders")
        gaussian = load_module(modules / "gaussian_policy.py", "gaussian_policy")
        shift = load_module(
            modules / "gaussian_shift_policy.py", "gaussian_shift_policy"
        )
        actor = (
            shift.GaussianShiftPolicy(128, 1600, shift_bound=0.5)
            if card == "7"
            else gaussian.GaussianPolicy(128, 1600, low=-2, high=2)
        )
        actor.probability_float32 = True
        parts = {
            "actor_image_encoder": compact.LightweightImageEncoder64(),
            "actor_state_encoder": compact.CompactStateEncoder(14, 64),
            "dsrl_action_noise_net": actor,
            "critic_image_encoder": compact.LightweightImageEncoder64(),
            "critic_state_encoder": compact.CompactStateEncoder(14, 64),
            "q_head": compact.CompactMultiQHead(action_dim=1600),
        }
        saved = torch.load(
            checkpoint / "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for name, part in parts.items():
            part.load_state_dict(
                {
                    k[len(name) + 1 :]: v
                    for k, v in saved["model"].items()
                    if k.startswith(name + ".")
                },
                strict=True,
            )
            part.eval()
        del saved
        replay = torch.load(
            checkpoint
            / "actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        storage = replay["storage"]
        ids = torch.linspace(0, int(replay["resident_size"]) - 1, 64).long()
        images = storage["curr_obs"]["main_images"][ids].float().unsqueeze(1)
        states = storage["curr_obs"]["states"][ids].float()
        data_actions = storage["actions"][ids].float()
        with torch.no_grad():
            features = torch.cat(
                (
                    parts["actor_state_encoder"](states),
                    parts["actor_image_encoder"](images),
                ),
                -1,
            )
            state_features = parts["critic_state_encoder"](states)
            image_features = parts["critic_image_encoder"](images)
            actual_actions = actor(features, sample=True)[0].flatten(1)
            heldout_actions = actor(features, sample=True)[0].flatten(1)
            broad_actions = actor(features, sample=True, noise_scale=1.0)[0].flatten(1)
            anchor = parts["q_head"](
                state_features, image_features, data_actions
            ).detach()

        def score(head):
            with torch.no_grad():
                data_q = head(state_features, image_features, data_actions)
                actual_q = head(state_features, image_features, actual_actions)
                heldout_q = head(state_features, image_features, heldout_actions)
                broad_q = head(state_features, image_features, broad_actions)
                return {
                    "data_q": float(data_q.mean()),
                    "actual_q": float(actual_q.mean()),
                    "actual_gap": float((actual_q - data_q).mean()),
                    "heldout_actual_gap": float((heldout_q - data_q).mean()),
                    "broad_q": float(broad_q.mean()),
                    "anchor_rmse": float((data_q - anchor).square().mean().sqrt()),
                }

        report = {"checkpoint": str(checkpoint), "before": score(parts["q_head"])}
        for name, include_policy in [("legacy", False), ("actual_and_broad", True)]:
            head = copy.deepcopy(parts["q_head"])
            optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
            trace = [{"step": 0, **score(head)}]
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                q_data = head(state_features, image_features, data_actions)
                q_broad = head(state_features, image_features, broad_actions)
                q_actual = (
                    head(state_features, image_features, actual_actions)
                    if include_policy
                    else None
                )
                loss = F.mse_loss(q_data, anchor) + args.weight * loss_fn(
                    q_data, q_broad, q_policy=q_actual
                )
                loss.backward()
                optimizer.step()
                if step in {1, 5, 10, args.steps}:
                    trace.append({"step": step, **score(head)})
            report[name] = score(head)
            report[name]["trace"] = trace
        assert all(
            parameter.grad is None
            for part in parts.values()
            for parameter in part.parameters()
        ), "Probe must leave loaded model parameters untouched"
        report["actual_gap_reduction_vs_legacy"] = (
            report["legacy"]["actual_gap"] - report["actual_and_broad"]["actual_gap"]
        )
        report["heldout_gap_reduction_vs_legacy"] = (
            report["legacy"]["heldout_actual_gap"]
            - report["actual_and_broad"]["heldout_actual_gap"]
        )
        result["runs"][card] = report
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({"card": card, **report}), flush=True)


if __name__ == "__main__":
    main()
