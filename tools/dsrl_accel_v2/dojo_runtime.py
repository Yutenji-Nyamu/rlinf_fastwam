# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
# Adapted from MINT-SJTU/RoboScope-Kit and RoboDojo control_robot.
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY.md.
"""Opt-in native adapters, based on pinned RoboScope Dojo patches.

Source: MINT-SJTU/RoboScope-Kit@30158a77, benchmarks/robodojo/patches.
Physics cadence, rendering count, current-frame fences and targets are preserved.
"""

import math
from typing import Any

_STATUS = {}


def startup(options: dict[str, Any]) -> None:
    """Set only this process's optional startup CPU settings."""
    import os
    import sys

    if options.get("cpu_affinity"):
        os.sched_setaffinity(0, set(options["cpu_affinity"]))
    count = options.get("physics_threads")
    if count:
        assert count in (1, 4, 8)
        idx = sys.argv.index("--kit_args") + 1
        sys.argv[idx] += (
            f" --/persistent/physics/numThreads={count} --/app/settings/persistent=false"
        )


def install(options: dict[str, Any]) -> Any:
    """Install after AppLauncher, before constructing the benchmark environment."""
    from env.environment.base_env import BaseEnv
    from env.robot_manager.robot_manager import RobotManager

    _STATUS.update(options=options)
    if options.get("unpaced_render"):
        original = BaseEnv.render

        def render(self) -> Any:
            if self.sim is None:
                return original(self)
            import carb.settings
            import omni.kit.loop._loop as omni_loop

            sim = self.sim.sim
            loop = omni_loop.acquire_loop_interface()
            dt = sim.get_rendering_dt()
            if not loop.get_manual_mode() or not math.isclose(
                loop.get_manual_step_size(), dt, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise RuntimeError(
                    "Unpaced render requires unchanged manual rendering timestep"
                )
            settings = carb.settings.get_settings()
            key = "/app/runLoops/main/rateLimitEnabled"
            previous = settings.get(key)
            if not isinstance(previous, bool):
                raise RuntimeError("Missing boolean Kit rate limiter")
            physics = sim.current_time_step_index
            try:
                settings.set_bool(key, False)
                original(self)
            finally:
                settings.set_bool(key, previous)
            if sim.current_time_step_index != physics or not math.isclose(
                sim.get_rendering_dt(), dt, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise RuntimeError("Render changed physics or rendering cadence")
            _STATUS["rate_limit_was_enabled"] = previous

        BaseEnv.render = render
    if options.get("control_state_reuse"):
        import hashlib
        import inspect
        from pathlib import Path

        actual = hashlib.sha256(
            Path(inspect.getfile(RobotManager)).read_bytes()
        ).hexdigest()
        expected = "1306cb6d17b58a04ad8fc298edb6eb9122469048dbc9717dcf246f4f985d9974"
        if actual != expected:
            raise RuntimeError("Native RobotManager source drift: " + actual)
        _STATUS["robot_manager_source_sha256"] = actual

        def wrap(original: Any, index_name: str) -> Any:
            def get(self, robot: Any, env_idx_list: Any = None) -> Any:
                live = self.robot_key[self.robot_list.index(robot)].data.joint_pos
                _STATUS["control_state_device"] = str(live.device)
                if live.device.type != "cpu":
                    return original(self, robot, env_idx_list)
                ids = range(self.num_envs) if env_idx_list is None else env_idx_list
                indices = getattr(robot, index_name)
                return {
                    i: live[i][indices].detach().numpy().copy() if i in ids else None
                    for i in range(self.num_envs)
                }

            return get

        RobotManager.get_joint = wrap(RobotManager.get_joint, "arm_joint_indices")
        RobotManager.get_end_effector_real_val = wrap(
            RobotManager.get_end_effector_real_val, "gripper_joint_indices"
        )
        RobotManager.control_robot = control_robot


def after_construct(env: Any, options: dict[str, Any]) -> None:
    """Read effective simulator settings after the first native reset."""
    import carb.settings
    import omni.kit.loop._loop as omni_loop
    from omni.kit.viewport.utility import get_active_viewport

    settings = carb.settings.get_settings()
    loop = omni_loop.acquire_loop_interface()
    sim = env.sim.sim
    _STATUS.update(
        rate_limit_enabled=settings.get("/app/runLoops/main/rateLimitEnabled"),
        manual_mode=loop.get_manual_mode(),
        manual_step=loop.get_manual_step_size(),
        rendering_dt=sim.get_rendering_dt(),
    )
    _STATUS["physics_threads"] = settings.get("/persistent/physics/numThreads")
    if (
        options.get("physics_threads")
        and _STATUS["physics_threads"] != options["physics_threads"]
    ):
        raise RuntimeError("Requested physics threads did not take effect")
    viewport = get_active_viewport()
    _STATUS["viewport_present"] = viewport is not None
    _STATUS["viewport_was_enabled"] = (
        bool(viewport.updates_enabled) if viewport is not None else False
    )
    if options.get("disable_viewport"):
        if viewport is not None:
            viewport.updates_enabled = False
    _STATUS["device"] = str(env.device)


def status() -> Any:
    """Return a copy of observed runtime settings."""
    return dict(_STATUS)


def control_robot(self, meta_control_list: Any = None) -> None:
    """Pinned native control_robot, caching only one call's exact MetaControl."""
    import torch
    from env.robot_manager.control_manager import MetaControl

    sim = self.sim.sim
    plan_lst = [
        i
        for i in range(self.scene.num_envs)
        if isinstance(meta_control_list[i], MetaControl)
    ]
    plan_num = len(plan_lst)
    expanded = {}
    for robot in self.robot_list:
        if robot.robot_type != "arm":
            continue
        arm = self.robot_key[self.robot_list.index(robot)]
        arm_position = torch.zeros(
            (plan_num, len(robot.arm_joint_indices)),
            device=sim.device,
            dtype=torch.float32,
        )
        arm_velocity = torch.zeros_like(arm_position)
        gripper_position = torch.zeros(
            (plan_num, len(robot.gripper_joint_indices)),
            device=sim.device,
            dtype=torch.float32,
        )
        for row, env_idx in enumerate(plan_lst):
            meta = meta_control_list[env_idx]
            if type(meta) is MetaControl:
                if env_idx not in expanded:
                    expanded[env_idx] = meta.get_action(self, env_idx=env_idx)
                action = expanded[env_idx]
            else:
                action = meta.get_action(self, env_idx=env_idx)
            arm_position[row] = torch.tensor(
                action[self.process_name(robot.arm_name)]["position"],
                device=sim.device,
                dtype=torch.float32,
            )
            arm_velocity[row] = torch.tensor(
                action[self.process_name(robot.arm_name)]["velocity"],
                device=sim.device,
                dtype=torch.float32,
            )
            gripper_position[row][:] = torch.tensor(
                action[self.process_name(robot.gripper_name)]["position"],
                device=sim.device,
                dtype=torch.float32,
            )
        ids = torch.tensor(plan_lst, dtype=torch.int32, device=arm_velocity.device)
        arm.set_joint_position_target(
            arm_position, joint_ids=robot.arm_joint_indices, env_ids=ids
        )
        arm.set_joint_velocity_target(
            arm_velocity, joint_ids=robot.arm_joint_indices, env_ids=ids
        )
        arm.set_joint_position_target(
            gripper_position, joint_ids=robot.gripper_joint_indices, env_ids=ids
        )
