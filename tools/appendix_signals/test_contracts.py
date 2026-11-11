"""CPU checks for the invariants that matter to these measurements."""

import numpy as np
import pytest
import torch

from .common import causal_prefix, entropy, seed_for, uniform_indices


def test_legacy_dojo_trace_preserves_native_rng_and_injects_requested_noise():
    from types import SimpleNamespace

    from .dojo_sampling import add_legacy_trace

    class LegacyModel:
        config = SimpleNamespace(num_steps=10, action_env_dim=2)

        def sample_actions(self, observation, noise=None):
            if noise is None:
                noise = torch.randn(2, 3, 4)
            return noise + 2, noise

        def predict_action_batch(self, obs, **kwargs):
            full, noise = self.sample_actions(obs)
            return full[..., :2], {
                "model_actions": full,
                "forward_inputs": {"chains": torch.stack([noise, full], dim=1)},
            }

    model = LegacyModel()
    initial_rng = torch.get_rng_state().clone()
    expected, _ = model.predict_action_batch(None)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    add_legacy_trace(model)
    actual, _ = model.predict_action_batch(None, num_steps=10)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_rng, torch.get_rng_state())
    requested = torch.full((2, 3, 4), 7.0)
    actual, result = model.predict_action_batch(None, noise=requested, num_steps=10)
    assert torch.equal(actual, (requested + 2)[..., :2])
    assert torch.equal(result["dvac_telemetry"]["initial_noise_full"], requested)
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_entropy_known_distribution_and_dimensions():
    a = np.array([[-1.0, -2.0], [1.0, 2.0]])[None, :, None, :]
    expected = 0.5 * np.log(2 * np.pi * np.e * (np.array([2.0, 8.0]) + 1e-6)).sum()
    assert entropy(a)[0, 0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        entropy(a[:, :1])
    with pytest.raises(ValueError):
        entropy(np.full_like(a, np.nan))


def test_causal_prefix_and_seed_uniqueness():
    for anchor in range(300):
        ids = causal_prefix(anchor)
        assert (
            len(ids) == 8
            and ids[0] == 0
            and ids[-1] == anchor
            and np.all(np.diff(ids) >= 0)
        )
    seeds = [seed_for(b, o, k) for b in range(2) for o in range(64) for k in range(8)]
    assert len(set(seeds)) == 1024
    assert len(uniform_indices(3, 12)) == 3


def test_accepted_noise_observer_preserves_main_and_rng():
    from tools.pi05_u_video.noise_samples import collect_noise_samples

    class FakeModel:
        def predict_action_batch(self, obs, *, noise, **kwargs):
            # An internal incidental RNG draw must not leak out of the observer.
            torch.randn(7)
            final = noise[:, :, :14] + obs["states"][:, None, :14]
            return final.double(), {"dvac_telemetry": {"final_model_action": final}}

    main_noise = torch.randn(2, 50, 32)
    main = main_noise[:, :, :14].clone()
    obs = dict(
        states=torch.zeros(2, 14),
        task_descriptions=["left", "right"],
        extra_view_images=None,
    )
    state = torch.get_rng_state().clone()
    z = collect_noise_samples(
        FakeModel(),
        obs,
        dict(initial_noise_full=main_noise, final_model_action=main),
        main.double(),
        173,
        max_extra_batch=2,
    )
    assert torch.equal(state, torch.get_rng_state())
    np.testing.assert_array_equal(z["noise4_env_actions"][:, 0], main.double().numpy())
    assert z["noise4_model_actions"].shape == (2, 4, 50, 14)
    assert not np.array_equal(
        z["noise4_initial_noise"][:, 1], z["noise4_initial_noise"][:, 2]
    )
    again = collect_noise_samples(
        FakeModel(),
        obs,
        dict(initial_noise_full=main_noise, final_model_action=main),
        main.double(),
        173,
        max_extra_batch=6,
    )
    np.testing.assert_array_equal(
        z["noise4_model_actions"], again["noise4_model_actions"]
    )


def test_binding_rejects_even_tiny_gpu0_context(monkeypatch):
    from . import pipeline

    monkeypatch.setattr(pipeline, "marked", lambda _: [{"pid": 99}])

    def snapshot(gpu_index):
        cards = []
        for i in range(8):
            process = (
                "<process_info><pid>99</pid><used_memory>4 MiB</used_memory><type>G</type></process_info>"
                if i == gpu_index
                else ""
            )
            cards.append("<gpu><processes>" + process + "</processes></gpu>")
        return ("<nvidia_smi_log>" + "".join(cards) + "</nvidia_smi_log>").encode()

    monkeypatch.setattr(
        pipeline.subprocess, "check_output", lambda *a, **k: snapshot(0)
    )
    assert pipeline.binding("test")[1][0]["gpu"] == 0
    monkeypatch.setattr(
        pipeline.subprocess, "check_output", lambda *a, **k: snapshot(4)
    )
    assert pipeline.binding("test")[1] == []


def test_optional_collection_failure_returns_to_formal(tmp_path):
    from . import pre_native

    class Process:
        def wait(self, timeout):
            return 1

    phase = dict(
        name="collection", command=["unused"], timeout=5, continue_on_failure=True
    )
    pre_native.run(
        tmp_path,
        {"pre_native_phases": [phase]},
        lambda *args: Process(),
        lambda *args, **kw: None,
        lambda _: {},
    )
    assert (tmp_path / "collection-finished.json").is_file()
    phase["continue_on_failure"] = False
    with pytest.raises(RuntimeError):
        pre_native.run(
            tmp_path,
            {"pre_native_phases": [phase]},
            lambda *args: Process(),
            lambda *args, **kw: None,
            lambda _: {},
        )


def test_pipeline_runs_child_with_inherited_cuda_override(tmp_path, monkeypatch):
    import json
    import sys
    import time
    from types import SimpleNamespace

    from . import pipeline

    ready = tmp_path / "ready.json"
    ready.write_text("{}")
    plan = dict(
        id="appendix-contract",
        gpu=5,
        output=str(tmp_path / "result"),
        ready_receipt=str(ready),
        absolute_deadline=time.time() + 30,
        stages=[
            dict(
                name="child",
                command=[
                    sys.executable,
                    "-c",
                    "import os; assert os.environ['CUDA_VISIBLE_DEVICES'] == '5'",
                ],
                environment={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"},
                timeout=20,
                cwd=str(tmp_path),
            )
        ],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(pipeline.os, "getuid", lambda: 1003)
    monkeypatch.setattr(pipeline, "binding", lambda _: ({}, []))
    monkeypatch.setattr(pipeline, "cleanup", lambda _: [])
    pipeline.main(SimpleNamespace(plan=str(path)))
    result = json.loads((tmp_path / "result/complete.json").read_text())
    assert result["results"][0]["state"] == "complete"


def test_cancelling_owner_cleans_inner_native_stage(tmp_path, monkeypatch):
    from . import pre_native

    events = []

    class Process:
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return 143

        def terminate(self):
            events.append("dispatcher_stopped")

    monkeypatch.setattr(pre_native, "cleanup", lambda tag: events.append(tag))
    phase = dict(
        name="appendix", command=["unused"], timeout=5, stage_identifiers=["inner"]
    )
    with pytest.raises(KeyboardInterrupt):
        pre_native.run(
            tmp_path,
            {"pre_native_phases": [phase]},
            lambda *args: Process(),
            lambda *args, **kwargs: None,
            lambda _: {},
        )
    assert events == ["dispatcher_stopped", "inner"]
