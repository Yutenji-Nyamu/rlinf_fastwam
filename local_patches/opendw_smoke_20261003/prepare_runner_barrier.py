"""Prepare the proven env-offload wait on the pinned Sidney runner.

By default emits a review patch. --apply is only for the new isolated checkout;
it never launches anything. Preserve old formal worktrees.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


BASE_REF = "2151a08ee1bd75df1bef0d8190e594bd5c7f7977"
EXPECTED_CHECKOUT = Path("/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlinf")
EXPECTED_BRANCH = "codex/sz3-opendw-robotwin-smoke-20261003"
RELATIVE = "rlinf/runners/embodied_runner.py"
ANCHOR = (
    "                    self.actor.recv_rollout_trajectories(\n"
    "                        input_channel=self.actor_channel\n"
    "                    ).wait()\n"
    "                    rollout_handle.wait()\n"
    "                    if self.reward is not None:\n"
)
REPLACEMENT = ANCHOR.replace(
    "                    if self.reward is not None:\n",
    "                    # The final trajectory can arrive before WM offload finishes.\n"
    "                    # Colocated actor training must wait for actual release.\n"
    "                    env_handle.wait()\n"
    "                    if self.reward is not None:\n",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    before = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
         "show", f"{BASE_REF}:{RELATIVE}"])
    target = repo / RELATIVE
    current = target.read_bytes()
    # Git on Windows can convert checkout line endings. Compare normalized text.
    source_text = before.decode("utf-8").replace("\r\n", "\n")
    current_text = current.decode("utf-8").replace("\r\n", "\n")
    if source_text != current_text:
        raise RuntimeError("Runner differs from the pinned base; review it before patching")
    if source_text.count(ANCHOR) != 1:
        raise RuntimeError("Expected exactly one synchronous rollout barrier anchor")
    after_text = source_text.replace(ANCHOR, REPLACEMENT, 1)
    args.output.mkdir(parents=True, exist_ok=False)
    patch = "".join(difflib.unified_diff(source_text.splitlines(True),
                                       after_text.splitlines(True),
                                       fromfile="a/" + RELATIVE, tofile="b/" + RELATIVE))
    (args.output / "env-offload-barrier.patch").write_text(patch, encoding="utf-8")
    (args.output / "runner-reviewed.py").write_text(after_text, encoding="utf-8")
    if args.apply:
        if repo != EXPECTED_CHECKOUT.resolve():
            raise RuntimeError(f"Apply only to the intended isolated checkout: {EXPECTED_CHECKOUT}")
        branch = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
             "branch", "--show-current"], text=True).strip()
        if branch != EXPECTED_BRANCH:
            raise RuntimeError(f"Unexpected checkout branch: {branch}")
        target.write_text(after_text, encoding="utf-8")
    record = {"source_ref": BASE_REF, "file": RELATIVE,
              "before_sha256": hashlib.sha256(before).hexdigest(),
              "after_sha256": hashlib.sha256(after_text.encode()).hexdigest(),
              "applied": args.apply, "launched": False}
    (args.output / "patch-contract.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
