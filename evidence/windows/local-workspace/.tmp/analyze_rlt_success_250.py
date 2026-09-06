from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar(accumulator: EventAccumulator, tag: str) -> pd.Series:
    return pd.Series(
        {
            int(event.step) + 1: float(event.value)
            for event in accumulator.Scalars(tag)
        },
        dtype=float,
    ).sort_index()


def aggregate(
    rates: pd.Series,
    denominators: pd.Series,
    selected_cycles: list[int],
) -> dict[str, float | int]:
    frame = pd.concat(
        [
            rates.loc[selected_cycles].rename("rate"),
            denominators.loc[selected_cycles].rename("episodes"),
        ],
        axis=1,
    )
    episodes = int(round(frame["episodes"].sum()))
    successes = int(round((frame["rate"] * frame["episodes"]).sum()))
    return {
        "cycles": len(selected_cycles),
        "episodes": episodes,
        "successes": successes,
        "rate": successes / episodes,
    }


event_file = Path(sys.argv[1])
accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
accumulator.Reload()

train_rate = scalar(accumulator, "env/success_once")
train_denominator = scalar(accumulator, "env/num_trajectories")
eval_rate = scalar(accumulator, "eval/success_once")
eval_denominator = scalar(accumulator, "eval/num_trajectories")
actor_switch = scalar(accumulator, "train/replay/actor_switch_rate")
ramp = scalar(accumulator, "train/actor/actor_weight_ramp_progress")
update_step = scalar(accumulator, "train/rlt/update_step")

latest = int(train_rate.index.max())
all_cycles = list(map(int, train_rate.index))
phases = {
    "P1_reference_collect": [cycle for cycle in all_cycles if 1 <= cycle <= 135],
    "P2_reference_plus_sac": [cycle for cycle in all_cycles if 136 <= cycle <= 154],
    "P3_student_ramp": [cycle for cycle in all_cycles if 155 <= cycle <= 191],
    "P4_stable_student": [cycle for cycle in all_cycles if 192 <= cycle <= 250],
}

result = {
    "latest_complete_cycle": latest,
    "train_all": aggregate(train_rate, train_denominator, all_cycles),
    "train_phases": {
        name: aggregate(train_rate, train_denominator, cycles)
        for name, cycles in phases.items()
    },
    "train_recent": {
        str(window): aggregate(
            train_rate,
            train_denominator,
            all_cycles[-window:],
        )
        for window in (5, 10, 20)
    },
    "eval_points": [
        {
            "cycle": int(cycle),
            "episodes": int(round(eval_denominator.loc[cycle])),
            "successes": int(
                round(eval_rate.loc[cycle] * eval_denominator.loc[cycle])
            ),
            "rate": float(eval_rate.loc[cycle]),
        }
        for cycle in eval_rate.index
    ],
    "markers": {
        "first_actor_switch_rate_one": (
            int(actor_switch[actor_switch >= 1.0 - 1e-8].index.min())
            if (actor_switch >= 1.0 - 1e-8).any()
            else None
        ),
        "first_ramp_progress_one": (
            int(ramp[ramp >= 1.0 - 1e-8].index.min())
            if (ramp >= 1.0 - 1e-8).any()
            else None
        ),
        "first_positive_update_step": (
            int(update_step[update_step > 0].index.min())
            if (update_step > 0).any()
            else None
        ),
        "latest_update_step_pre_cycle": int(round(update_step.iloc[-1])),
        "latest_actor_switch_rate": float(actor_switch.iloc[-1]),
        "latest_ramp_progress": float(ramp.iloc[-1]),
    },
    "available_success_cycles": [int(value) for value in train_rate.index],
}

print(json.dumps(result, indent=2, sort_keys=True))
