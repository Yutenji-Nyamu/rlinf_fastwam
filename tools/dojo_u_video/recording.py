"""Read-only callbacks around native Dojo actions and native video streams."""

import os
from pathlib import Path

from common import append, episode_dir, progress, read, save, serial


def instrument(env):
    plan = None
    if os.environ.get("DOJO_U_CONFIG"):
        from layouts import install_plan

        plan = read(os.environ["DOJO_U_CONFIG"])["layout_plan"]
        install_plan(env, plan)
    tick = [0]
    executing = [False]
    current = {}
    native_reset, native_sim = env.reset, env.sim_step
    native_obs, native_stream = env.get_obs_batch, env._stream_vision
    native_action, native_save = env.take_action_batch, env.save_video

    def meta(i):
        episode = int(env.env_seeds[i])
        source = plan[episode] if plan else {}
        return dict(
            task=env.task_name,
            layout=episode,
            slot=i,
            model_seed=int(env.eval_seed),
            run_id=env.run_id,
            action_before=int(env.take_action_cnt[i]),
            episode_id=episode,
            source_layout=source,
        )

    def reset(*args, **kwargs):
        executing[0] = False
        current.clear()
        result = native_reset(*args, **kwargs)
        tick[0] = 0
        executing[0] = True
        progress("reset_complete", layouts=env.current_env_seed_map)
        return result

    def sim_step(*args, **kwargs):
        result = native_sim(*args, **kwargs)
        if executing[0]:
            tick[0] += 1
        return result

    def stream(i, frame):
        before = {k: w.n_frames for k, w in env.video_writers.get(i, {}).items()}
        result = native_stream(i, frame)
        for cam, writer in env.video_writers.get(i, {}).items():
            if writer.n_frames > before.get(cam, 0):
                append(
                    episode_dir(meta(i)) / "frames.jsonl",
                    dict(
                        **meta(i),
                        camera=cam,
                        frame=writer.n_frames - 1,
                        control_tick=tick[0],
                        sim_seconds=tick[0] * float(env.dt),
                        **current.get(i, dict(query=None, h=None)),
                        terminal=bool(env.end_flag[i]),
                    ),
                )
        return result

    def get_obs(*args, **kwargs):
        result = native_obs(*args, **kwargs)
        for obs in result:
            obs["_u_meta"] = meta(int(obs["env_idx"]))
        return result

    def action(actions_list, env_idx_list=None):
        ids = list(range(env.num_envs)) if env_idx_list is None else list(env_idx_list)
        before = [int(env.take_action_cnt[i]) for i in ids]
        for i, n in zip(ids, before):
            current[i] = dict(query=n // 50, h=n % 50)
        start = tick[0]
        result = native_action(actions_list, env_idx_list)
        for i, n, command in zip(ids, before, actions_list):
            if env.take_action_cnt[i] == n:
                continue
            append(
                episode_dir(meta(i)) / "actions.jsonl",
                dict(
                    **meta(i),
                    query=n // 50,
                    h=n % 50,
                    action_index=n,
                    start_tick=start,
                    end_tick=tick[0],
                    command=serial(command),
                    terminal=bool(env.end_flag[i]),
                    success=bool(env.success[i]),
                ),
            )
        progress("action", counts=env.take_action_cnt, control_tick=tick[0])
        return result

    def save_video(i, video_path, tag):
        streams = {
            cam: dict(
                frames=w.n_frames,
                width=w.width,
                height=w.height,
                fps=w.fps,
                path=os.path.abspath(video_path.replace(".mp4", f"_{cam}_{tag}.mp4")),
            )
            for cam, w in env.video_writers.get(i, {}).items()
        }
        result = native_save(i, video_path, tag)
        save(
            episode_dir(meta(i)) / "episode.json",
            dict(
                **meta(i),
                success=tag == "success",
                native_tag=tag,
                executed_actions=int(env.take_action_cnt[i]),
                videos=streams,
                native_result=os.path.abspath(Path(env.save_dir) / "_result.json"),
            ),
        )
        return result

    env.reset, env.sim_step = reset, sim_step
    env.get_obs_batch, env._stream_vision = get_obs, stream
    env.take_action_batch, env.save_video = action, save_video
    return env
