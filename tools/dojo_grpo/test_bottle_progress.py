"""CPU smoke of native step accounting and the production GRPO actor."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from bottle_progress import BottleProgress
from native_entry import NativeLoop
from rlinf.algorithms.advantages import compute_grpo_advantages
from rlinf.algorithms.utils import calculate_scores, preprocess_embodied_advantages_inputs
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def native_case(mode, directory):
    # Four lanes: 2 bottles at timeout, 4 without home, full success, no bottles.
    stages = [[0, 1, 0, 0], [1, 2, 1, 0], [1, 3, 2, 0], [2, 4, 3, 0]]
    env = SimpleNamespace(end_flag=[False] * 4, success=[False] * 4,
                          take_action_cnt=[0] * 4, robot_action_dim_info={},
                          unstable_envs=set(), physx_monitor_enabled=False,
                          reward_manager=SimpleNamespace(score_completed_count=[0] * 4))
    def step(controls, env_idx_list):
        for i in env_idx_list:
            t = env.take_action_cnt[i]
            env.reward_manager.score_completed_count[i] = stages[t][i]
            env.take_action_cnt[i] += 1
            if t == 3:
                env.end_flag[i] = True
                env.success[i] = i == 2

    env.take_action_batch = step
    with TemporaryDirectory(dir=directory) as temp:
        loop = NativeLoop.__new__(NativeLoop)
        loop.env, loop.n, loop.mode = env, 4, mode
        loop.cfg = {"capture_every_action":True}
        loop.bottles = BottleProgress(4)
        loop.train_renderer = "chunk"
        loop.totals = {"action_seconds":0.0,"observation_seconds":0.0}
        loop.output, loop.trial_profile = Path(temp), {}
        loop.wave_layouts, loop.wave_recorded = list(range(4)), False
        loop.observations = lambda: {}
        loop.progress = lambda *_args, **_kwargs: None
        loop.unpack = lambda *_args, **_kwargs: None
        actions = np.zeros((4, 3, 14), dtype=np.float32)
        batches = [loop.step({"mode": mode,"actions":actions}) for _ in range(3)]
        rewards = np.concatenate([b["rewards"] for b in batches], axis=1)
        rows = json.loads((loop.output/"completed-waves.jsonl").read_text())["rows"]
        np.testing.assert_array_equal(rewards.sum(axis=1), [.5, 1, 1, 0] if mode=="train" else [0,0,1,0])
        assert not rewards[:,4:].any()
        assert [r["bottles"] for r in rows] == [2,4,4,0]
        assert [r["success"] for r in rows] == [False,False,True,False]
        assert rows[0]["first_bottle_steps"] == [2,4,-1,-1]
        assert rows[1]["first_bottle_steps"] == [1,2,3,4]
        assert all(r["episode_return"] == float(total) for r,total in zip(rows,rewards.sum(axis=1)))


def actor_case():
    # The first group has just one quarter-point across eight episodes.
    # The second has identical partial scores and the third spans the full range.
    episode_scores = torch.tensor([.25]+[0.]*7+[.50]*8+[0,.25,.50,.75,1,0,0,0])
    rewards = torch.zeros(3,24,50)
    rewards[0,:,3] = torch.minimum(episode_scores, torch.full_like(episode_scores,.25))
    rewards[1,:,10] = episode_scores - rewards[0,:,3]
    dones = torch.zeros(4,24,50,dtype=torch.bool)
    dones[2,:,10:]=True
    dones[3]=True
    worker = SimpleNamespace(cfg=OmegaConf.create(dict(
        env=dict(train=dict(rollout_epoch=1, auto_reset=False, ignore_terminations=False)),
        algorithm=dict(filter_rewards=True, group_size=8, reward_type="chunk_level",
                       rewards_lower_bound=.1, rewards_upper_bound=.9, filter_by_reward_variation=True))))
    batch=EmbodiedFSDPActor._process_received_rollout_batch(worker,{"rewards":rewards,"dones":dones})
    expected=torch.tensor([True]*8+[False]*8+[True]*8)
    assert torch.equal(batch["loss_mask"][0,:,0],expected)
    assert not batch["loss_mask"][2].any()
    inp=preprocess_embodied_advantages_inputs(**batch, reward_type="chunk_level", adv_type="grpo",group_size=8)
    inp=calculate_scores(**inp)
    torch.testing.assert_close(inp["rewards"].flatten(),episode_scores)
    adv,_=compute_grpo_advantages(**inp)
    assert adv[0,0]>0 and adv[0,1]<0 and not adv[:,8:16].any()
    assert bool(torch.all(adv[0,16:20] < adv[0,17:21]))
    assert not adv[2].any()


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        native_case("train",temp)
        native_case("eval",temp)
    actor_case()
    print("PASS native train and eval, repeated and terminal steps, GRPO filters, scores and advantages")
