"""One scripted-expert RoboTwin environment probe for the joint venv."""

from __future__ import annotations

import os

import yaml

from envs import CONFIGS_PATH
from envs.adjust_bottle import adjust_bottle


with open("./task_config/demo_clean.yml", encoding="utf-8") as stream:
    args = yaml.safe_load(stream)

with open(
    os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8"
) as stream:
    embodiment_table = yaml.safe_load(stream)

embodiment = args["embodiment"]
assert len(embodiment) == 1, embodiment
robot_file = embodiment_table[embodiment[0]]["file_path"]
with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as stream:
    robot_cfg = yaml.safe_load(stream)

args.update(
    {
        "task_name": "adjust_bottle",
        "task_config": "demo_clean",
        "seed": 0,
        "now_ep_num": 0,
        "eval_mode": True,
        "save_data": False,
        "render_freq": 0,
        "left_robot_file": robot_file,
        "right_robot_file": robot_file,
        "left_embodiment_config": robot_cfg,
        "right_embodiment_config": robot_cfg,
        "dual_arm_embodied": True,
    }
)

environment = adjust_bottle()
try:
    environment.setup_demo(**args)
    observation = environment.get_obs()
    print("observation_keys", sorted(observation))
    environment.play_once()
    print("play_once_complete", bool(environment.check_success()))
finally:
    environment.close_env(clear_cache=True)
