"""Closed-loop iid/repeated-noise comparison through the accepted Dojo bridge.

No policy or simulator changes. Both conditions use the frozen original model,
matched layout requests, and matching first-row noise. Reset replacements and
their actual layout identities are retained rather than called paired blindly.
"""

import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from openpi.models.model import Observation

from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv
from rlinf.models.embodiment.openpi import get_model


def main():
    d = Path(os.environ["DOJO_GRPO_DEPLOYMENT"])
    plan = json.loads((d / "plan.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
    torch.set_num_threads(4)
    cfg = OmegaConf.load(d / "config/formal.yaml")
    cfg.actor.model.openpi.use_dsrl = False
    cfg.actor.model.openpi.noise_method = "flow_ode"
    cfg.actor.model.add_q_head = False
    model = get_model(cfg.actor.model).eval().cuda()
    env = RoboDojoEnv(cfg.env.eval, plan["num_envs"], 0, 1, None)
    n = plan["num_envs"]
    episodes = int(plan.get("baseline_episodes", 32))
    limit = int(cfg.env.eval.max_episode_steps)
    assert episodes % n == 0
    rows = []
    for mode in ("independent", "repeated"):
        for wave in range(episodes // n):
            started = time.monotonic()
            requested = list(range(wave * n, (wave + 1) * n))
            obs, info = env.reset(env_seeds=requested)
            generator = torch.Generator(device="cuda").manual_seed(110400 + wave)
            done = torch.zeros(n, dtype=torch.bool)
            success = torch.zeros(n, dtype=torch.bool)
            for _ in range((limit + 49) // 50):
                with torch.inference_mode():
                    batch = model.input_transform(model.obs_processor(obs), transpose=False)
                    observation = Observation.from_dict(model.precision_processor(batch))
                    noise = torch.randn(n, 50, 32, device="cuda", generator=generator)
                    if mode == "repeated":
                        noise = noise[:, :1].expand(-1, 50, -1).contiguous()
                    # Same bf16 noise precision for both modes; this is the
                    # accepted DSRL path's precision, isolated from correlation.
                    output = model.sample_actions(
                        observation, noise=noise.bfloat16(), mode="eval", compute_values=False
                    )
                    actions = model.output_transform(
                        {"actions": output["actions"], "state": observation.state}
                    )["actions"]
                observations, _, terminated, truncated, infos = env.chunk_step(actions)
                obs = observations[-1]
                done |= (terminated | truncated).any(1)
                success |= infos[-1]["success"]
                if done.all():
                    break
            row = dict(
                task=plan["task"], mode=mode, wave=wave, episodes=n,
                successes=int(success.sum()), success=success.tolist(),
                steps=env.elapsed_steps.tolist(), reset_audit=info["reset_audit"],
                seconds=time.monotonic() - started, time=time.time(),
            )
            rows.append(row)
            with (d / "noise-baseline.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "reset_audit"}), flush=True)
    summary = {
        mode: {"successes": sum(x["successes"] for x in rows if x["mode"] == mode),
               "episodes": episodes}
        for mode in ("independent", "repeated")
    }
    (d / "noise-baseline-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
