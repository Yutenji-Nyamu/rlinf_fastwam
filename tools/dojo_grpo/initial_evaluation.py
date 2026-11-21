"""Measure the untouched Dojo policy, then return control to the formal runner."""
import json
import time


def write_json(path, value):
    tmp = path.with_suffix(".next")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def run_baseline(runner, evaluate, directory, plan):
    config = runner.cfg
    size = int(config.env.eval.total_num_envs)
    target = plan["phases"][0]["episodes"]
    assert runner.global_step == 0 and config.runner.resume_dir is None
    assert size == 32 and target == 128 and config.env.eval.rollout_epoch == 1
    runner.update_rollout_weights()
    successes = 0
    batch_log = directory / "baseline-batches.jsonl"
    started = time.time()
    for batch in range(target // size):
        metrics = evaluate(runner)
        assert int(metrics["num_trajectories"]) == size, metrics
        wins = round(float(metrics["success_once"]) * size)
        successes += wins
        completed = (batch + 1) * size
        with batch_log.open("a") as f:
            f.write(json.dumps(dict(batch=batch + 1, successes=wins, episodes=size, time=time.time())) + "\n")
        write_json(directory / "status.json", dict(phase="INITIAL_EVALUATION", task=plan["task"],
                   completed=completed, episodes=target, successes=successes, time=time.time()))
    result = dict(round=0, time=time.time(), task=plan["task"], model=plan["model"],
                  successes=successes, episodes=target, seconds=time.time() - started,
                  metrics=dict(success_once=successes / target, num_trajectories=target))
    write_json(directory / "initial-evaluation.json", result)
    with (directory / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    runner.metric_logger.log(data={"eval/success_once": successes / target}, step=0)
    write_json(directory / "status.json", dict(phase="FORMAL_TRAINING", task=plan["task"],
               rounds=plan["rounds"], time=time.time(), baseline_successes=successes, baseline_episodes=target))
