# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch

from rlinf.workers.env.env_worker import EnvWorker


def test_ogpo_final_observation_uses_each_envs_first_done_step() -> None:
    observations = [
        {
            "states": torch.tensor([[10.0], [11.0], [12.0]]),
            "pixels": np.asarray([[110], [111], [112]]),
        },
        {
            "states": torch.tensor([[20.0], [21.0], [22.0]]),
            "pixels": np.asarray([[120], [121], [122]]),
        },
        {
            "states": torch.tensor([[30.0], [31.0], [32.0]]),
            "pixels": np.asarray([[130], [131], [132]]),
        },
    ]
    terminations = torch.tensor(
        [[True, False, False], [False, False, False], [False, False, False]]
    )
    truncations = torch.tensor(
        [[False, False, False], [False, True, False], [False, False, False]]
    )

    final_obs = EnvWorker._ogpo_first_terminal_observation(
        observations, terminations, truncations
    )

    assert final_obs is not None
    torch.testing.assert_close(
        final_obs["states"], torch.tensor([[10.0], [21.0], [32.0]])
    )
    np.testing.assert_array_equal(
        final_obs["pixels"], np.asarray([[110], [121], [132]])
    )


def test_ogpo_final_observation_is_none_without_done() -> None:
    observations = [{"states": torch.zeros(2, 1)}]
    done = torch.zeros(2, 1, dtype=torch.bool)

    assert (
        EnvWorker._ogpo_first_terminal_observation(observations, done, done) is None
    )
