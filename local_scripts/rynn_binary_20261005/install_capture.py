"""Patch a PRIVATE RoboTwinEnv source, preserving a hash-identified original."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def patch(source):
    if "native_binary_recorder" in source:
        raise ValueError("Capture already installed")
    reset = "        extracted_obs = self._extract_obs_image(raw_obs)\n\n        return extracted_obs, infos"
    step = "        infos = self._record_metrics(step_reward, infos)"
    if source.count(reset) != 1 or source.count(step) != 2:
        raise ValueError("RoboTwinEnv differs from audited reset/step/chunk_step contract")
    source = source.replace("class RoboTwinEnv(gym.Env):", "from .native_binary_recorder import capture_reset, capture_observe\n\n\nclass RoboTwinEnv(gym.Env):", 1)
    source = source.replace(reset, reset.replace("\n\n        return", "\n        capture_reset(self, extracted_obs, env_idx, env_seeds)\n\n        return"))
    source = source.replace(step, step + "\n        capture_observe(self, extracted_obs, infos)")
    ast.parse(source)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-env-file", required=True, type=Path)
    args = p.parse_args()
    target = args.private_env_file.resolve()
    raw = target.read_bytes()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    updated = patch(source)
    sha = hashlib.sha256(raw).hexdigest()
    backup = target.with_name(target.name + ".pre-binary-" + sha[:12])
    if backup.exists():
        raise ValueError("Capture backup already exists")
    backup.write_bytes(raw)
    shutil.copyfile(Path(__file__).with_name("native_binary_recorder.py"), target.with_name("native_binary_recorder.py"))
    target.write_text(updated, encoding="utf-8")
    print(json.dumps(dict(target=str(target), original_sha256=sha, backup=str(backup),
        updated_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
