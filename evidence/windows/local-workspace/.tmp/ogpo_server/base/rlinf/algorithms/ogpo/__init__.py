# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical building blocks for OGPO."""

from .core import (
    clipped_ppo_loss,
    condot_score_from_velocity,
    conservative_group_advantages,
    diagonal_gaussian_log_prob,
    h_step_td_target,
    normalized_whole_chain_log_prob,
    tapered_noise_std,
    tapered_sde_drift_correction,
    tapered_sde_step_stats,
)

__all__ = [
    "clipped_ppo_loss",
    "condot_score_from_velocity",
    "conservative_group_advantages",
    "diagonal_gaussian_log_prob",
    "h_step_td_target",
    "normalized_whole_chain_log_prob",
    "tapered_noise_std",
    "tapered_sde_drift_correction",
    "tapered_sde_step_stats",
]
