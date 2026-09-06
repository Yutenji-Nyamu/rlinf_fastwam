#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a deterministic numerical fixture from the locked official QAM code.

This script is intentionally standalone and CPU-only. It imports the locked
upstream QAM checkout, executes its small-network Plain-QAM update, and writes a
numeric-only NPZ for production PyTorch parity tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "2726d767c9a0a7a46d49693f0391f73dc2cf58ac"
SCHEMA_VERSION = 1
BATCH_SIZE = 2
OBSERVATION_DIM = 3
ACTION_DIM = 4
FLOW_STEPS = 3
NUM_QS = 10
HIDDEN_DIMS = (8, 8)
SEED = 20260731
REQUIRED_SOURCE_FILES = (
    "agents/qam.py",
    "utils/flax_utils.py",
    "utils/networks.py",
)
DIRECT_DISTRIBUTIONS = (
    "numpy",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "distrax",
    "tensorflow-probability",
    "ml-collections",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Clean checkout of ColinQiyangLi/qam at the locked commit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npz file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination.",
    )
    return parser.parse_args()


def _run_git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Official QAM source is not a directory: {source}")

    commit = _run_git(source, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"Expected official QAM commit {EXPECTED_COMMIT}, found {commit}"
        )

    tracked_status = _run_git(source, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError(
            "Official QAM checkout has tracked modifications:\n" + tracked_status
        )

    source_hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required official source file missing: {path}")
        source_hashes[relative] = _sha256_file(path)

    return {
        "source_root": str(source),
        "source_commit": commit,
        "source_hashes": source_hashes,
    }


def _distribution_versions() -> tuple[dict[str, str], list[str]]:
    direct = {name: importlib.metadata.version(name) for name in DIRECT_DISTRIBUTIONS}
    resolved = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    return direct, resolved


def _path_component(component: Any) -> str:
    for attribute in ("key", "idx", "name"):
        if hasattr(component, attribute):
            return str(getattr(component, attribute))
    return str(component)


def _tree_path(path: tuple[Any, ...]) -> str:
    return "/".join(_path_component(component) for component in path)


class FixtureBuilder:
    """Collect numeric arrays and UTF-8-encoded metadata for an NPZ fixture."""

    def __init__(self, np_module: Any, jax_module: Any):
        self._np = np_module
        self._jax = jax_module
        self.arrays: dict[str, Any] = {}
        self.tree_manifest: dict[str, list[dict[str, Any]]] = {}

    def add(self, name: str, value: Any) -> None:
        if name in self.arrays:
            raise KeyError(f"Duplicate fixture key: {name}")
        array = self._np.asarray(self._jax.device_get(value))
        if array.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden in fixture array {name}")
        self.arrays[name] = array

    def add_json(self, name: str, value: Any) -> None:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.add(name, self._np.frombuffer(payload, dtype=self._np.uint8).copy())

    def add_tree(self, prefix: str, tree: Any) -> None:
        entries: list[dict[str, Any]] = []
        path_leaves, _ = self._jax.tree_util.tree_flatten_with_path(tree)
        for index, (path, leaf) in enumerate(path_leaves):
            array = self._np.asarray(self._jax.device_get(leaf))
            if array.dtype.hasobject:
                raise TypeError(
                    f"Object dtype is forbidden in tree {prefix}: {_tree_path(path)}"
                )
            key = f"{prefix}_{index:04d}"
            self.add(key, array)
            entries.append(
                {
                    "key": key,
                    "path": _tree_path(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )
        self.tree_manifest[prefix] = entries


def _tree_max_abs(jax: Any, jnp: Any, left: Any, right: Any) -> Any:
    differences = jax.tree_util.tree_map(
        lambda x, y: jnp.max(jnp.abs(x - y)), left, right
    )
    leaves = jax.tree_util.tree_leaves(differences)
    return jnp.max(jnp.stack([jnp.asarray(value) for value in leaves]))


def _make_batch(jnp: Any) -> dict[str, Any]:
    observations = jnp.asarray(
        [[-0.70, 0.20, 0.90], [0.40, -0.50, 0.10]], dtype=jnp.float32
    )
    actions = jnp.asarray(
        [
            [[-0.80, -0.10, 0.35, 0.90]],
            [[0.55, -0.65, 0.15, -0.25]],
        ],
        dtype=jnp.float32,
    )
    next_observations = jnp.asarray(
        [
            [[-0.55, 0.25, 0.75]],
            [[0.35, -0.40, 0.05]],
        ],
        dtype=jnp.float32,
    )
    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": jnp.asarray([[0.25], [1.00]], dtype=jnp.float32),
        "masks": jnp.asarray([[1.0], [0.0]], dtype=jnp.float32),
        "valid": jnp.asarray([[1.0], [0.0]], dtype=jnp.float32),
    }


def _derive_update_randoms(
    jax: Any,
    agent_rng: Any,
    observations: Any,
    action_dim: int,
    flow_steps: int,
) -> dict[str, Any]:
    new_agent_rng, loss_rng = jax.random.split(agent_rng)
    unused_total_rng, actor_rng, critic_rng = jax.random.split(loss_rng, 3)
    next_action_rng, edit_rng = jax.random.split(critic_rng)
    (
        unused_actor_rng,
        fm_x_rng,
        fm_t_rng,
        adj_rng,
        actor_edit_rng,
    ) = jax.random.split(actor_rng, 5)

    fm_x0 = jax.random.normal(fm_x_rng, (observations.shape[0], action_dim))
    fm_t = jax.random.uniform(fm_t_rng, (observations.shape[0], 1))
    adj_x0 = jax.random.normal(adj_rng, observations.shape[:-1] + (action_dim,))
    adj_step_keys = jax.random.split(adj_rng, flow_steps)
    adj_step_noises = jax.numpy.stack(
        [
            jax.random.normal(key, observations.shape[:-1] + (action_dim,))
            for key in adj_step_keys
        ],
        axis=0,
    )
    next_action_noise = jax.random.normal(
        next_action_rng, (observations.shape[0], 1, action_dim)
    )
    return {
        "new_agent_rng": new_agent_rng,
        "loss_rng": loss_rng,
        "unused_total_rng": unused_total_rng,
        "actor_rng": actor_rng,
        "critic_rng": critic_rng,
        "next_action_rng": next_action_rng,
        "edit_rng": edit_rng,
        "unused_actor_rng": unused_actor_rng,
        "fm_x_rng": fm_x_rng,
        "fm_t_rng": fm_t_rng,
        "adj_rng": adj_rng,
        "actor_edit_rng": actor_edit_rng,
        "adj_step_keys": adj_step_keys,
        "fm_x0": fm_x0,
        "fm_t": fm_t,
        "adj_x0": adj_x0,
        "adj_step_noises": adj_step_noises,
        "next_action_noise": next_action_noise,
    }


def _manual_adj_forward(
    jax: Any,
    jnp: Any,
    agent: Any,
    observations: Any,
    randoms: dict[str, Any],
) -> dict[str, Any]:
    h = 1.0 / FLOW_STEPS
    x = randoms["adj_x0"]
    states = [x]
    times = []
    sigmas = []
    applied_velocities = []

    actor_slow = agent.network.select("target_actor_slow")
    actor_fast = agent.network.select("actor_fast")
    for index in range(FLOW_STEPS):
        t = (index / FLOW_STEPS) * jnp.ones_like(x[..., 0:1])
        sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
        noise = randoms["adj_step_noises"][index]
        if index != FLOW_STEPS - 1:
            velocity = actor_fast(observations, x, t)
            x = x + h * (2 * velocity - x / (t + h)) + jnp.sqrt(h) * sigma * noise
        else:
            velocity = actor_slow(observations, x, t)
            x = x + h * velocity
        states.append(x)
        times.append(t)
        sigmas.append(sigma)
        applied_velocities.append(velocity)

    return {
        "states": jnp.stack(states, axis=0),
        "times": jnp.stack(times, axis=0),
        "sigmas": jnp.stack(sigmas, axis=0),
        "applied_velocities": jnp.stack(applied_velocities, axis=0),
    }


def _build_fixture(source_info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "false"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    jax.config.update("jax_enable_x64", False)

    source = Path(source_info["source_root"])
    sys.path.insert(0, str(source))
    try:
        from agents.qam import QAMAgent, get_config
    finally:
        sys.path.pop(0)

    imported_qam = Path(inspect.getfile(QAMAgent)).resolve()
    expected_qam = (source / "agents/qam.py").resolve()
    if imported_qam != expected_qam:
        raise RuntimeError(
            f"Imported QAMAgent from {imported_qam}, expected {expected_qam}"
        )

    config = get_config()
    config.batch_size = BATCH_SIZE
    config.actor_hidden_dims = HIDDEN_DIMS
    config.value_hidden_dims = HIDDEN_DIMS
    config.horizon_length = 1
    config.action_chunking = False
    config.num_qs = NUM_QS
    config.flow_steps = FLOW_STEPS
    config.best_of_n = 1
    config.inv_temp = 0.3
    config.fql_alpha = 0.0
    config.edit_scale = 0.0
    config.target_actor = True
    config.residual = False
    config.clip_adj = True
    config.clip_grad = True
    config.use_target_grad = True
    config.edit_target_entropy = None

    batch = _make_batch(jnp)
    example_observation = jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32)
    example_action = jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
    agent = QAMAgent.create(
        seed=SEED,
        ex_observations=example_observation,
        ex_actions=example_action,
        config=config,
    )
    params_before = agent.network.params
    create_root_rng = jax.random.PRNGKey(SEED)
    expected_agent_rng, create_init_rng = jax.random.split(create_root_rng, 2)
    if not np.array_equal(
        np.asarray(jax.device_get(expected_agent_rng)),
        np.asarray(jax.device_get(agent.rng)),
    ):
        raise RuntimeError("Official create() PRNG split no longer matches exporter")
    randoms = _derive_update_randoms(
        jax,
        agent.rng,
        batch["observations"],
        ACTION_DIM,
        FLOW_STEPS,
    )

    builder = FixtureBuilder(np, jax)
    builder.add("schema_version", np.asarray(SCHEMA_VERSION, dtype=np.int32))
    builder.add(
        "dimensions",
        np.asarray(
            [BATCH_SIZE, OBSERVATION_DIM, ACTION_DIM, FLOW_STEPS, NUM_QS],
            dtype=np.int32,
        ),
    )
    for name, value in batch.items():
        builder.add(f"batch_{name}", value)
    builder.add("random_create_root_rng", create_root_rng)
    builder.add("random_create_init_rng", create_init_rng)
    builder.add("agent_rng_before", agent.rng)
    for name, value in randoms.items():
        builder.add(f"random_{name}", value)

    # Official behavior-flow FM intermediates.
    batch_actions = batch["actions"][..., 0, :]
    fm_x0 = randoms["fm_x0"]
    fm_t = randoms["fm_t"]
    fm_xt = (1 - fm_t) * fm_x0 + fm_t * batch_actions
    fm_target_velocity = batch_actions - fm_x0
    fm_prediction = agent.network.select("actor_slow")(
        batch["observations"], fm_xt, fm_t, params=params_before
    )
    fm_per_example = jnp.square(fm_prediction - fm_target_velocity).mean(axis=-1)
    fm_loss = jnp.mean(fm_per_example * batch["valid"][..., -1])
    builder.add("fm_x0", fm_x0)
    builder.add("fm_t", fm_t)
    builder.add("fm_xt", fm_xt)
    builder.add("fm_target_velocity", fm_target_velocity)
    builder.add("fm_prediction", fm_prediction)
    builder.add("fm_per_example", fm_per_example)
    builder.add("loss_flow_matching", fm_loss)

    # Official AM SDE/last-step ODE path and reverse adjoints.
    manual_adj = _manual_adj_forward(jax, jnp, agent, batch["observations"], randoms)
    adj_xs, adjoints, adj_times, pre_adj_info = agent.adj_matching(
        batch["observations"], randoms["adj_rng"], flow_steps=FLOW_STEPS
    )
    forward_state_error = jnp.max(jnp.abs(manual_adj["states"][:-1] - adj_xs))
    time_error = jnp.max(jnp.abs(manual_adj["times"] - adj_times))
    if float(forward_state_error) > 1e-6 or float(time_error) > 1e-7:
        raise RuntimeError(
            "Manual extraction no longer matches official adj_matching: "
            f"state={float(forward_state_error)}, time={float(time_error)}"
        )

    endpoint = manual_adj["states"][-1]
    target_q_endpoint = agent.network.select("target_critic")(
        batch["observations"], jnp.clip(endpoint, -1.0, 1.0)
    )

    def endpoint_q_sum(action: Any) -> Any:
        values = agent.network.select("target_critic")(
            batch["observations"], jnp.clip(action, -1.0, 1.0)
        )
        return values.mean(axis=0).sum()

    endpoint_q_gradient = jax.grad(endpoint_q_sum)(endpoint)
    terminal_adjoint = -endpoint_q_gradient * config.inv_temp
    repeated_observations = jnp.repeat(batch["observations"][None], FLOW_STEPS, axis=0)
    fine_velocity = agent.network.select("actor_fast")(
        repeated_observations, adj_xs, adj_times, params=params_before
    )
    base_velocity = agent.network.select("target_actor_slow")(
        repeated_observations, adj_xs, adj_times
    )
    sigmas = manual_adj["sigmas"]
    am_residual = (fine_velocity - base_velocity) * 2 / sigmas + sigmas * adjoints
    am_per_step_batch = jnp.square(am_residual).sum(axis=-1)
    am_loss = jnp.mean(jnp.sum(am_per_step_batch, axis=0))

    for name, value in manual_adj.items():
        builder.add(f"adj_forward_{name}", value)
    builder.add("adj_xs_official", adj_xs)
    builder.add("adj_times_official", adj_times)
    builder.add("adj_reverse_states", adjoints)
    builder.add("adj_endpoint", endpoint)
    builder.add("adj_endpoint_target_qs", target_q_endpoint)
    builder.add("adj_endpoint_q_gradient", endpoint_q_gradient)
    builder.add("adj_terminal", terminal_adjoint)
    builder.add("adj_fine_velocity", fine_velocity)
    builder.add("adj_base_velocity", base_velocity)
    builder.add("adj_am_residual", am_residual)
    builder.add("adj_am_per_step_batch", am_per_step_batch)
    builder.add("loss_adjoint_matching", am_loss)
    for name, value in pre_adj_info.items():
        builder.add(f"adj_info_{name}", value)

    # Official critic target and loss intermediates.
    next_observations = batch["next_observations"][..., -1, :]
    next_actions = agent.sample_actions(next_observations, rng=randoms["critic_rng"])
    next_actions = jnp.clip(next_actions, -1.0, 1.0)
    next_qs = agent.network.select("target_critic")(next_observations, next_actions)
    pessimistic_next_q = next_qs.mean(axis=0) - config.rho * next_qs.std(axis=0)
    critic_target = (
        batch["rewards"][..., -1]
        + (config.discount**config.horizon_length)
        * batch["masks"][..., -1]
        * pessimistic_next_q
    )
    current_qs = agent.network.select("critic")(
        batch["observations"], batch_actions, params=params_before
    )
    critic_squared_error = jnp.square(current_qs - critic_target)
    critic_loss = jnp.mean(critic_squared_error * batch["valid"][..., -1])
    builder.add("critic_next_observations", next_observations)
    builder.add("critic_next_actions", next_actions)
    builder.add("critic_next_target_qs", next_qs)
    builder.add("critic_pessimistic_next_q", pessimistic_next_q)
    builder.add("critic_target", critic_target)
    builder.add("critic_current_qs", current_qs)
    builder.add("critic_squared_error", critic_squared_error)
    builder.add("loss_critic", critic_loss)

    # One exact combined loss/gradient/clip/Optax-Adam step.
    def loss_fn(candidate_params: Any) -> tuple[Any, Any]:
        return agent.total_loss(batch, candidate_params, rng=randoms["loss_rng"])

    (total_loss, loss_info), raw_gradients = jax.value_and_grad(loss_fn, has_aux=True)(
        params_before
    )
    expected_total_loss = critic_loss + fm_loss + am_loss
    loss_error = jnp.abs(total_loss - expected_total_loss)
    if float(loss_error) > 2e-5:
        raise RuntimeError(
            f"Extracted losses no longer match official total_loss: {float(loss_error)}"
        )

    clip_transform = optax.clip_by_global_norm(1.0)
    clipped_gradients, _ = clip_transform.update(
        raw_gradients,
        clip_transform.init(params_before),
        params_before,
    )
    raw_global_grad_norm = optax.global_norm(raw_gradients)
    clipped_global_grad_norm = optax.global_norm(clipped_gradients)
    adam_updates, opt_state_after = agent.network.tx.update(
        raw_gradients,
        agent.network.opt_state,
        params_before,
    )
    params_after_adam = optax.apply_updates(params_before, adam_updates)

    updated_agent, official_update_info = agent.update(batch)
    params_after_official_update = updated_agent.network.params
    tau = config.tau
    expected_target_critic = jax.tree_util.tree_map(
        lambda online, target: online * tau + target * (1 - tau),
        params_before["modules_critic"],
        params_before["modules_target_critic"],
    )
    expected_target_actor_slow = jax.tree_util.tree_map(
        lambda online, target: online * tau + target * (1 - tau),
        params_before["modules_actor_slow"],
        params_before["modules_target_actor_slow"],
    )

    online_critic_error = _tree_max_abs(
        jax,
        jnp,
        params_after_adam["modules_critic"],
        params_after_official_update["modules_critic"],
    )
    online_slow_error = _tree_max_abs(
        jax,
        jnp,
        params_after_adam["modules_actor_slow"],
        params_after_official_update["modules_actor_slow"],
    )
    online_fast_error = _tree_max_abs(
        jax,
        jnp,
        params_after_adam["modules_actor_fast"],
        params_after_official_update["modules_actor_fast"],
    )
    target_critic_error = _tree_max_abs(
        jax,
        jnp,
        expected_target_critic,
        params_after_official_update["modules_target_critic"],
    )
    target_slow_error = _tree_max_abs(
        jax,
        jnp,
        expected_target_actor_slow,
        params_after_official_update["modules_target_actor_slow"],
    )
    update_errors = jnp.stack(
        [
            online_critic_error,
            online_slow_error,
            online_fast_error,
            target_critic_error,
            target_slow_error,
        ]
    )
    if float(jnp.max(update_errors)) > 2e-6:
        raise RuntimeError(
            "Manual Adam/pre-update EMA extraction no longer matches "
            f"official update: {np.asarray(update_errors)}"
        )

    builder.add("loss_total", total_loss)
    builder.add("loss_expected_total", expected_total_loss)
    builder.add("gradient_raw_global_norm", raw_global_grad_norm)
    builder.add("gradient_clipped_global_norm", clipped_global_grad_norm)
    builder.add("update_consistency_max_abs", update_errors)
    for name, value in loss_info.items():
        builder.add(f"loss_info_{name.replace('/', '_')}", value)
    for name, value in official_update_info.items():
        builder.add(f"update_info_{name.replace('/', '_')}", value)

    builder.add_tree("params_before", params_before)
    builder.add_tree("gradients_raw", raw_gradients)
    builder.add_tree("gradients_clipped", clipped_gradients)
    builder.add_tree("adam_updates", adam_updates)
    builder.add_tree("params_after_adam", params_after_adam)
    builder.add_tree("optimizer_state_before", agent.network.opt_state)
    builder.add_tree("optimizer_state_after", opt_state_after)
    builder.add_tree("target_critic_expected_preupdate_ema", expected_target_critic)
    builder.add_tree(
        "target_actor_slow_expected_preupdate_ema",
        expected_target_actor_slow,
    )
    builder.add_tree("params_after_official_update", params_after_official_update)

    initial_slow_fast_difference = _tree_max_abs(
        jax,
        jnp,
        params_before["modules_actor_slow"],
        params_before["modules_actor_fast"],
    )
    initial_critic_target_error = _tree_max_abs(
        jax,
        jnp,
        params_before["modules_critic"],
        params_before["modules_target_critic"],
    )
    initial_slow_target_error = _tree_max_abs(
        jax,
        jnp,
        params_before["modules_actor_slow"],
        params_before["modules_target_actor_slow"],
    )
    if float(initial_slow_fast_difference) == 0.0:
        raise RuntimeError("Official actor_slow and actor_fast were not independent")
    if (
        float(initial_critic_target_error) != 0.0
        or float(initial_slow_target_error) != 0.0
    ):
        raise RuntimeError("Official online/target initialization sync changed")
    builder.add("init_actor_slow_fast_max_abs", initial_slow_fast_difference)
    builder.add("init_critic_target_max_abs", initial_critic_target_error)
    builder.add("init_actor_slow_target_max_abs", initial_slow_target_error)

    direct_versions, resolved_distributions = _distribution_versions()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": source_info,
        "dimensions": {
            "batch": BATCH_SIZE,
            "observation": OBSERVATION_DIM,
            "action": ACTION_DIM,
            "flow_steps": FLOW_STEPS,
            "num_qs": NUM_QS,
            "hidden_dims": list(HIDDEN_DIMS),
        },
        "seed": SEED,
        "dtype": "float32",
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "python": sys.version,
        "platform": platform.platform(),
        "direct_distributions": direct_versions,
        "resolved_distributions": resolved_distributions,
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in config.to_dict().items()
        },
        "tree_manifest": builder.tree_manifest,
        "semantics": {
            "plain_qam": True,
            "action_chunking": False,
            "last_am_forward_step": "target_actor_slow_ode",
            "terminal_gradient": "target_critic_ensemble_mean",
            "td_bootstrap": "target_mean_minus_rho_std",
            "optimizer": "global_norm_clip_1_then_combined_optax_adam",
            "ema_source": "pre_update_online_parameters",
        },
    }
    builder.add_json("metadata_json_utf8", metadata)
    return np, builder.arrays


def _write_fixture(np: Any, arrays: dict[str, Any], output: Path, force: bool) -> str:
    output = output.resolve()
    if output.suffix != ".npz":
        raise ValueError(f"Fixture output must use .npz suffix: {output}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to replace existing fixture without --force: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dict(sorted(arrays.items())))

    with np.load(output, allow_pickle=False) as fixture:
        if sorted(fixture.files) != sorted(arrays):
            raise RuntimeError("Written fixture keys do not match collected arrays")
        for key in fixture.files:
            if fixture[key].dtype.hasobject:
                raise TypeError(f"Written fixture contains object dtype: {key}")

    return _sha256_file(output)


def main() -> int:
    args = _parse_args()
    source_info = _validate_source(args.source)
    np, arrays = _build_fixture(source_info)
    fixture_sha256 = _write_fixture(np, arrays, args.output, args.force)
    summary = {
        "array_count": len(arrays),
        "fixture_sha256": fixture_sha256,
        "output": str(args.output.resolve()),
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_info["source_commit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
