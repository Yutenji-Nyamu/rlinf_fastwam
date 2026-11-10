# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
import copy
from pathlib import Path

from omegaconf import OmegaConf
from prepare import BASE, read, save

for e in read(BASE / "prepared-dsrl.json")["entries"].values():
    d = Path(e["deployment"])
    p = read(d / "plan.json")
    assert not (d / "identity.json").exists()
    if e["method"] != "dsrl":
        continue
    c = OmegaConf.to_container(OmegaConf.load(d / "config/formal.yaml"), resolve=True)
    c["runner"].update(
        max_steps=2, val_check_interval=0, save_interval=1, resume_dir=None
    )
    c["runner"]["logger"].update(log_path=e["run"] + "/smoke", experiment_name="smoke")
    c["algorithm"]["replay_buffer"]["warmup_size"] = 8
    c["algorithm"]["dsrl_critic_warmup_updates"] = 4
    c["env"]["train"].update(max_steps_per_rollout_epoch=100, max_episode_steps=100)
    OmegaConf.save(OmegaConf.create(c), d / "config/smoke.yaml")
    phase = copy.deepcopy(p["phases"][0])
    phase.update(name="smoke", timeout=2400)
    phase["command"][-1] = "smoke"
    assert p["phases"][0]["name"] == "formal"
    p["phases"].insert(0, phase)
    save(d / "plan.json", p)
print("SMOKE_PREPARED")
