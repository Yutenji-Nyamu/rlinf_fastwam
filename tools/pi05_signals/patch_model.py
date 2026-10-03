"""Apply the narrowly scoped signal-collection patch to an explicit wrapper path.

Default is a dry check. Use ``python patch_model.py PATH --apply`` on the
isolated checkout. This script never locates a checkout or connects to a server.
It accepts the reviewed DV50 wrapper only (normalizing CRLF for identity),
parses the transformed source before writing, and atomically replaces PATH.
No model/dependency imports, model instantiation, or inference occur here.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile


BASE_LF_SHA256 = "188c36b8dc6f6c29beff34aab990e6d706077a7024d6b11fbff554ee99623985"
PATCHED_LF_SHA256 = "57542745995cd1a3dd5be6693283f44086b410a0bb89fcb41f3c2c2ef75165a1"
PATCH_MARKER = "# pi05-signals-20261003: explicit eval-only collection"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one exact source anchor, found {count}")
    return text.replace(old, new, 1)


def _function_span(source: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one function named {name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(map(len, lines[: node.end_lineno]))


def build_patch(source: str) -> str:
    """Pure source transformation; caller must verify base identity first."""
    source = _replace_once(
        source,
        "from concurrent.futures import ThreadPoolExecutor\n",
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from contextlib import nullcontext\n" + PATCH_MARKER + "\n",
        "nullcontext import",
    )

    start, end = _function_span(source, "predict_action_batch")
    body = source[start:end]
    body = _replace_once(
        body,
        "        return_dvac_telemetry: bool = False,\n        **kwargs,\n",
        "        return_dvac_telemetry: bool = False,\n"
        "        record_signals: bool = False,\n"
        "        consistency_steps: int = 5,\n"
        "        noise=None,\n"
        "        num_steps: int | None = None,\n"
        "        **kwargs,\n",
        "predict signature",
    )
    body = _replace_once(
        body,
        "        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs\n",
        "        if mode != \"eval\" and (noise is not None or num_steps is not None or record_signals):\n"
        "            raise ValueError(\"Per-call inference overrides and signal collection require mode='eval'\")\n"
        "        if record_signals and not return_dvac_telemetry:\n"
        "            raise ValueError(\"record_signals requires return_dvac_telemetry=True\")\n"
        "        if self.config.use_dsrl and (noise is not None or record_signals):\n"
        "            raise ValueError(\"External noise/signal collection is not supported on the DSRL path\")\n"
        "        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs\n",
        "predict validation",
    )
    body = _replace_once(
        body,
        "                noise=noise_actions,\n                mode=\"eval\",\n",
        "                noise=noise_actions,\n                mode=\"eval\",\n"
        "                num_steps=num_steps,\n",
        "DSRL optional eval step forwarding",
    )
    body = _replace_once(
        body,
        "            if rtc_context is not None and mode == \"eval\" and self.config.rtc_enabled:\n"
        "                if return_dvac_telemetry:\n",
        "            if rtc_context is not None and mode == \"eval\" and self.config.rtc_enabled:\n"
        "                if noise is not None or num_steps is not None or record_signals:\n"
        "                    raise ValueError(\"Per-call inference overrides/signal collection require the native non-RTC path\")\n"
        "                if return_dvac_telemetry:\n",
        "RTC override guard",
    )
    body = _replace_once(
        body,
        "                outputs = self.sample_actions(\n"
        "                    observation,\n"
        "                    mode=mode,\n"
        "                    compute_values=compute_values,\n"
        "                    return_dvac_telemetry=return_dvac_telemetry,\n"
        "                )\n",
        "                outputs = self.sample_actions(\n"
        "                    observation,\n"
        "                    mode=mode,\n"
        "                    compute_values=compute_values,\n"
        "                    return_dvac_telemetry=return_dvac_telemetry,\n"
        "                    noise=noise,\n"
        "                    num_steps=num_steps,\n"
        "                    record_signals=record_signals,\n"
        "                    consistency_steps=consistency_steps,\n"
        "                )\n",
        "native predict forwarding",
    )
    source = source[:start] + body + source[end:]

    start, end = _function_span(source, "sample_actions")
    body = source[start:end]
    body = _replace_once(
        body,
        "        return_dvac_telemetry: bool = False,\n    ) -> torch.Tensor:\n",
        "        return_dvac_telemetry: bool = False,\n"
        "        num_steps: int | None = None,\n"
        "        record_signals: bool = False,\n"
        "        consistency_steps: int = 5,\n"
        "    ) -> torch.Tensor:\n",
        "sample signature",
    )
    body = _replace_once(
        body,
        "        bsize = observation.state.shape[0]\n",
        "        if num_steps is not None:\n"
        "            if mode != \"eval\":\n"
        "                raise ValueError(\"num_steps override is evaluation-only\")\n"
        "            if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:\n"
        "                raise ValueError(\"num_steps must be a positive integer\")\n"
        "        if record_signals:\n"
        "            if mode != \"eval\" or not return_dvac_telemetry:\n"
        "                raise ValueError(\"record_signals requires eval and return_dvac_telemetry=True\")\n"
        "            if isinstance(consistency_steps, bool) or not isinstance(consistency_steps, int) or consistency_steps < 1:\n"
        "                raise ValueError(\"consistency_steps must be a positive integer\")\n"
        "            main_steps = self.config.num_steps if num_steps is None else num_steps\n"
        "            if main_steps != 10:\n"
        "                raise ValueError(\"Signal collection is defined on the M10 main chain\")\n"
        "            if getattr(self, \"_signal_observer\", None) is None:\n"
        "                raise ValueError(\"Bind model._signal_observer before enabling record_signals\")\n"
        "        bsize = observation.state.shape[0]\n",
        "sample option guards",
    )
    body = _replace_once(
        body,
        "        return self._sample_actions_with_prefix_cache(\n",
        "        result = self._sample_actions_with_prefix_cache(\n",
        "sample result",
    )
    body = _replace_once(
        body,
        "            return_dvac_telemetry=return_dvac_telemetry,\n        )\n",
        "            return_dvac_telemetry=return_dvac_telemetry,\n"
        "            num_steps=num_steps,\n"
        "            record_signals=record_signals,\n"
        "        )\n"
        "        if record_signals:\n"
        "            telemetry = result[\"dvac_telemetry\"]\n"
        "            # This is the full noise AFTER the cached sampler's dtype cast.\n"
        "            # Reusing the original pre-cast random sample can change the path.\n"
        "            initial_noise = telemetry[\"initial_noise_full\"]\n"
        "            rng_devices = []\n"
        "            if initial_noise.device.type == \"cuda\":\n"
        "                rng_devices = [initial_noise.device.index]\n"
        "            # flow_ode still samples zero-weight noise each round. Restore\n"
        "            # CPU and this CUDA RNG so the next main query is unchanged.\n"
        "            with torch.random.fork_rng(devices=rng_devices, enabled=True):\n"
        "                side = self._sample_actions_with_prefix_cache(\n"
        "                    state, prefix_output, prefix_pad_masks, past_key_values,\n"
        "                    noise=initial_noise.clone(), mode=\"eval\",\n"
        "                    compute_values=False, return_dvac_telemetry=False,\n"
        "                    num_steps=consistency_steps, record_signals=False,\n"
        "                )\n"
        "            telemetry[\"side_action_model\"] = side[\"actions\"][\n"
        "                ..., : self.config.action_env_dim\n"
        "            ].detach()\n"
        "            telemetry[\"side_num_steps\"] = torch.tensor(\n"
        "                consistency_steps, device=initial_noise.device, dtype=torch.int64\n"
        "            )\n"
        "            side_grid = self._get_timesteps(consistency_steps, initial_noise.device)\n"
        "            telemetry[\"side_timesteps\"] = side_grid[:-1].detach()\n"
        "            telemetry[\"side_final_time\"] = side_grid[-1].detach()\n"
        "        return result\n",
        "same-cache side chain",
    )
    source = source[:start] + body + source[end:]

    start, end = _function_span(source, "_sample_actions_with_prefix_cache")
    body = source[start:end]
    body = _replace_once(
        body,
        "        return_dvac_telemetry: bool = False,\n    ) -> torch.Tensor:\n",
        "        return_dvac_telemetry: bool = False,\n"
        "        num_steps: int | None = None,\n"
        "        record_signals: bool = False,\n"
        "    ) -> torch.Tensor:\n",
        "cached sampler signature",
    )
    body = _replace_once(
        body,
        "        num_steps = self.config.num_steps\n",
        "        if num_steps is None:\n"
        "            num_steps = self.config.num_steps\n"
        "        else:\n"
        "            if mode != \"eval\":\n"
        "                raise ValueError(\"num_steps override is evaluation-only\")\n"
        "            if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:\n"
        "                raise ValueError(\"num_steps must be a positive integer\")\n"
        "        observer = None\n"
        "        if record_signals:\n"
        "            if mode != \"eval\" or not return_dvac_telemetry or num_steps != 10:\n"
        "                raise ValueError(\"record_signals requires native M10 eval telemetry\")\n"
        "            observer = getattr(self, \"_signal_observer\", None)\n"
        "            if observer is None:\n"
        "                raise ValueError(\"Bind model._signal_observer before enabling record_signals\")\n",
        "cached sampler eval override",
    )
    body = _replace_once(
        body,
        "        x_t = noise\n",
        "        if record_signals:\n"
        "            expected_shape = (bsize, self.config.action_horizon, self.config.action_dim)\n"
        "            if tuple(noise.shape) != expected_shape or noise.device != device:\n"
        "                raise ValueError(\"Signal collection requires full model-space initial noise on the state device\")\n"
        "        initial_noise_full = noise.detach().clone() if record_signals else None\n"
        "        velocity_trace = [] if record_signals else None\n"
        "        x_t = noise\n",
        "capture actual full initial noise",
    )
    old_call = (
        "            x_t_mean, x_t_std, value_t, v_t = self.sample_mean_var_val(\n"
        "                x_t,\n"
        "                idx,\n"
        "                state,\n"
        "                prefix_pad_masks,\n"
        "                past_key_values,\n"
        "                sample_method,\n"
        "                num_steps,\n"
        "                compute_values,\n"
        "            )\n"
    )
    body = _replace_once(
        body,
        old_call,
        "            with (observer.step(idx, num_steps) if observer is not None else nullcontext()):\n"
        + "".join("    " + line for line in old_call.splitlines(keepends=True))
        + "            if velocity_trace is not None:\n"
        "                velocity_trace.append(v_t[..., : self.config.action_env_dim].detach().clone())\n",
        "main step observer scope and direct velocity",
    )
    body = _replace_once(
        body,
        "        return result\n",
        "        if record_signals:\n"
        "            result[\"dvac_telemetry\"][\"velocity\"] = torch.stack(velocity_trace, dim=1)\n"
        "            result[\"dvac_telemetry\"][\"initial_noise_full\"] = initial_noise_full\n"
        "        return result\n",
        "signal telemetry extension",
    )
    source = source[:start] + body + source[end:]
    ast.parse(source)
    compile(source, "patched_openpi_action_model.py", "exec")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Explicit openpi_action_model.py in the isolated checkout")
    parser.add_argument("--apply", action="store_true", help="Write the validated patch (default: check only)")
    parser.add_argument("--diff", action="store_true", help="Print the unified source diff")
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    if not target.is_file() or target.name != "openpi_action_model.py":
        parser.error("target must be an existing openpi_action_model.py")
    raw = target.read_bytes()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    before = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if before == PATCHED_LF_SHA256:
        print(json.dumps({"status": "already_applied", "target": str(target), "sha256_lf": before}))
        return
    if before != BASE_LF_SHA256:
        raise SystemExit(f"Source identity mismatch: expected {BASE_LF_SHA256}, got {before}; no write")
    patched = build_patch(source)
    after = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    if after != PATCHED_LF_SHA256:
        raise SystemExit(f"Internal patched-source identity mismatch: {after}; no write")
    if args.diff:
        print("".join(difflib.unified_diff(
            source.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile=str(target), tofile=str(target) + " (signal patch)",
        )), end="")
    if args.apply:
        newline = "\r\n" if b"\r\n" in raw else "\n"
        encoded = patched.replace("\n", newline).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(json.dumps({
        "status": "applied" if args.apply else "check_passed",
        "target": str(target), "before_sha256_lf": before, "after_sha256_lf": after,
        "modified_functions": ["predict_action_batch", "sample_actions", "_sample_actions_with_prefix_cache"],
    }))


if __name__ == "__main__":
    main()
