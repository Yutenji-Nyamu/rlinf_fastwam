# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner


def _runner(*, checkpoint_interval: int) -> tuple[EmbodiedRunner, list[int]]:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {
            "algorithm": {
                "ogpo": {
                    "eval_interval_rows": 0,
                    "checkpoint_interval_rows": checkpoint_interval,
                    "total_online_rows": 250,
                    "final_eval": False,
                }
            }
        }
    )
    runner._ogpo_row_schedule_initialized = False
    runner._ogpo_next_eval_rows = 0
    runner._ogpo_next_checkpoint_rows = checkpoint_interval
    runner._ogpo_last_eval_rows = 0
    runner._ogpo_last_checkpoint_rows = 0
    runner._ogpo_run_start_rows = 0
    saves: list[int] = []
    runner._save_checkpoint = lambda: saves.append(runner._current_rows)
    return runner, saves


def _schedule(runner: EmbodiedRunner, rows: int, previous: int) -> None:
    runner._current_rows = rows
    runner._maybe_ogpo_eval_and_checkpoint(rows, previous)


def test_final_row_saves_when_interval_does_not_divide_total() -> None:
    runner, saves = _runner(checkpoint_interval=100)

    _schedule(runner, 200, 150)
    _schedule(runner, 250, 200)

    assert saves == [200, 250]


def test_final_threshold_checkpoint_is_not_saved_twice() -> None:
    runner, saves = _runner(checkpoint_interval=50)

    _schedule(runner, 250, 200)

    assert saves == [250]
