# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Numerical contract and bounded SAC-component timings; no environment changes."""

import argparse
import copy
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch


def load(root, name):
    path = root / "rlinf/models/embodiment/modules" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(root, device, output):
    torch.set_num_threads(2)
    torch.manual_seed(1135)
    enc = load(root, "compact_encoders")
    gauss = load(root, "gaussian_policy")
    dev = torch.device(device)
    result = {"torch": torch.__version__, "device": str(dev), "benchmarks": {}}
    for bench, state_dim in [("rt2", 14), ("dojo", 14)]:
        q = enc.CompactMultiQHead(state_dim=64, image_dim=64, action_dim=1600).to(dev)
        q2 = copy.deepcopy(q)
        q2.vectorized = True
        inputs = [
            torch.randn(32, n, device=dev, requires_grad=True) for n in (64, 64, 1600)
        ]
        inputs2 = [x.detach().clone().requires_grad_(True) for x in inputs]
        y = q(*inputs)
        y2 = q2(*inputs2)
        torch.testing.assert_close(y, y2, atol=2e-6, rtol=3e-5)
        y.square().mean().backward()
        y2.square().mean().backward()
        for x, z in zip(inputs, inputs2):
            torch.testing.assert_close(x.grad, z.grad, atol=2e-7, rtol=3e-4)
        for x, z in zip(q.parameters(), q2.parameters()):
            torch.testing.assert_close(x.grad, z.grad, atol=3e-7, rtol=3e-4)
        assert list(q.state_dict()) == list(q2.state_dict())
        qerr = float((y - y2).abs().max())
        policy = gauss.GaussianPolicy(128, 1600, low=-1.0, high=1.0).to(dev)
        scaled = copy.deepcopy(policy)
        scaled.low = -2.0
        scaled.high = 2.0
        features = torch.randn(32, 128, device=dev)
        torch.manual_seed(77)
        a, lp = policy(features, sample=True)
        torch.manual_seed(77)
        a2, lp2 = scaled(features, sample=True)
        torch.testing.assert_close(a2, 2 * a, atol=0, rtol=0)
        torch.testing.assert_close(lp2, lp - 1600 * math.log(2), atol=3e-4, rtol=1e-6)
        ad, _ = policy(features, sample=True, deterministic=True)
        ad2, _ = scaled(features, sample=True, deterministic=True)
        torch.testing.assert_close(ad2, 2 * ad, atol=0, rtol=0)
        evaluated = scaled(features).log_prob(a2[:, 0])
        assert evaluated.shape == lp2.shape
        torch.testing.assert_close(evaluated, lp2, atol=0.01, rtol=2e-5)
        assert float(a2.abs().max()) <= 2 and torch.isfinite(lp2).all()
        optim = torch.optim.Adam(q.parameters(), lr=3e-4)
        before = [p.detach().clone() for p in q.parameters()]
        optim.step()
        nonzero = unchanged = 0
        for p, prev in zip(q.parameters(), before):
            mask = p.grad.ne(0)
            nonzero += int(mask.sum())
            unchanged += int((mask & p.eq(prev)).sum())
        assert all(
            v.dtype == torch.float32
            for state in optim.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        assert unchanged / nonzero < 0.01
        record = {
            "q_max_abs": qerr,
            "q_input_and_parameter_gradients": "pass",
            "checkpoint_keys": "unchanged",
            "bound_and_logprob": "pass",
            "unchanged_nonzero_gradient_fraction": unchanged / nonzero,
        }
        cnn = enc.LightweightImageEncoder64().to(dev)
        cnn2 = copy.deepcopy(cnn)
        cnn2.channels_last = True
        x = torch.randn(8, 1, 3, 64, 64, device=dev, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        a = cnn(x)
        b = cnn2(x2)
        try:
            torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
            a.square().mean().backward()
            b.square().mean().backward()
            torch.testing.assert_close(x.grad, x2.grad, atol=2e-5, rtol=3e-3)
            for v, w in zip(cnn.parameters(), cnn2.parameters()):
                torch.testing.assert_close(v.grad, w.grad, atol=3e-5, rtol=3e-3)
            record["channels_last_output_gradient_contract"] = "pass"
        except AssertionError as error:
            # Optional candidate rejection must not erase the successful DSRL
            # precision contract or prevent testing unrelated accelerations.
            record["channels_last_output_gradient_contract"] = "rejected"
            record["channels_last_error"] = str(error)
        if dev.type == "cuda":
            data = [torch.randn(256, n, device=dev) for n in (64, 64, 1600)]
            ref = copy.deepcopy(q).cpu().state_dict()
            initial_rng = torch.cuda.get_rng_state()
            outputs = {}
            for vectorized, fused, compile_q in [
                (False, False, False),
                (False, True, False),
                (True, False, False),
                (True, True, False),
                (True, True, True),
            ]:
                label = (
                    f"vec{int(vectorized)}_fused{int(fused)}_compile{int(compile_q)}"
                )
                candidate = enc.CompactMultiQHead(
                    state_dim=64, image_dim=64, action_dim=1600
                ).to(dev)
                candidate.load_state_dict(ref)
                candidate.vectorized = vectorized
                candidate.compile_vectorized = compile_q
                opt = torch.optim.Adam(candidate.parameters(), lr=3e-4, fused=fused)

                def update():
                    opt.zero_grad(set_to_none=True)
                    loss = candidate(*data).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 10.0)
                    opt.step()

                torch.cuda.set_rng_state(initial_rng)
                t0 = time.perf_counter()
                try:
                    for _ in range(5):
                        update()
                    torch.cuda.synchronize()
                    startup = time.perf_counter() - t0
                    times = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(20):
                            update()
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) / 20)
                    outputs[label] = {
                        "median_seconds": statistics.median(times),
                        "startup_seconds": startup,
                    }
                    outputs[label]["state"] = {
                        k: v.detach().cpu() for k, v in candidate.state_dict().items()
                    }
                except Exception as exc:
                    outputs[label] = {"error": repr(exc)}
                print(
                    json.dumps(
                        {
                            "benchmark": bench,
                            "case": label,
                            **{k: v for k, v in outputs[label].items() if k != "state"},
                        }
                    ),
                    flush=True,
                )
            baseline = outputs["vec0_fused0_compile0"].pop("state")
            for label, res in outputs.items():
                if "state" in res:
                    state = res.pop("state")
                    res["max_param_delta_after_105_updates"] = max(
                        float((state[k] - baseline[k]).abs().max()) for k in baseline
                    )
            record["q_update_timings"] = outputs
        result["benchmarks"][bench] = record
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.root, a.device, a.output)
