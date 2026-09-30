"""Apply only the explicit LIBERO head-only input adapter to pinned RLinf.

Default --check is read-only; --apply writes only two verified source files.
No model load, networking, branch creation, GPU, or process control.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

UPSTREAM = "d34d4c320d08cb982de034aa9a011f08dc0fa217"
PLAN = {
    "rlinf/models/embodiment/openpi/policies/libero_policy.py": {
        "sha256": "e060c68094f02bf90a456d1bfb7ab8cb5767aec620115770d519aadb3e7d2975",
        "replacements": [
            (
                "    model_type: _model.ModelType\n\n    def __call__(self, data: dict) -> dict:\n",
                "    model_type: _model.ModelType\n"
                "    # Explicit experiment choice; the default retains the pretrained two-view input.\n"
                "    wrist_mode: str = \"required\"\n\n"
                "    def __post_init__(self):\n"
                "        if self.wrist_mode not in (\"required\", \"disabled\"):\n"
                "            raise ValueError(f\"Unsupported wrist_mode={self.wrist_mode!r}\")\n"
                "        if self.wrist_mode == \"disabled\" and self.model_type != _model.ModelType.PI05:\n"
                "            raise ValueError(\"The head-only adapter is limited to the pi05_libero experiment\")\n\n"
                "    def __call__(self, data: dict) -> dict:\n",
            ),
            (
                '        wrist_image = _parse_image(data["observation/wrist_image"])\n',
                '        if self.wrist_mode == "disabled":\n'
                '            wrist_image = np.zeros_like(base_image)\n'
                '            wrist_valid = np.False_\n'
                '            # pi05_libero has discrete_state_input=False and no state projection.\n'
                '            # This is unused 8D LIBERO-shaped padding, not predicted proprioception.\n'
                '            raw_state = np.asarray(data["observation/state"])\n'
                '            state = np.zeros((*raw_state.shape[:-1], 8), dtype=np.float32)\n'
                '        else:\n'
                '            wrist_image = _parse_image(data["observation/wrist_image"])\n'
                '            wrist_valid = np.True_\n'
                '            state = data["observation/state"]\n',
            ),
            ('            "state": data["observation/state"],\n', '            "state": state,\n'),
            ('                "left_wrist_0_rgb": np.True_,\n', '                "left_wrist_0_rgb": wrist_valid,\n'),
        ],
    },
    "rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py": {
        "sha256": "80f5157a5d8c946f343c07bf1e31f782639203d86c91ede4a1b8af210c29e6a3",
        "replacements": [
            (
                "    extra_delta_transform: bool = False\n",
                "    extra_delta_transform: bool = False\n    wrist_mode: str = \"required\"\n",
            ),
            (
                "        # The repack transform is *only* applied to the data coming from the dataset,\n",
                '        if self.wrist_mode == "disabled" and (\n'
                '            model_config.model_type != _model.ModelType.PI05\n'
                '            or model_config.discrete_state_input\n'
                '            or self.extra_delta_transform\n'
                '        ):\n'
                '            raise ValueError("Head-only LIBERO requires PI05 with no state conditioning or extra delta transform")\n'
                '        # The repack transform is *only* applied to the data coming from the dataset,\n',
            ),
            (
                "            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],\n",
                "            inputs=[\n"
                "                libero_policy.LiberoInputs(\n"
                "                    model_type=model_config.model_type, wrist_mode=self.wrist_mode\n"
                "                )\n"
                "            ],\n",
            ),
        ],
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_plan(repo: Path):
    rows = []
    for relative, spec in PLAN.items():
        path = (repo / relative).resolve()
        path.relative_to(repo)
        original = subprocess.check_output(["git", "-C", str(repo), "show", f"{UPSTREAM}:{relative}"])
        if sha(original) != spec["sha256"]:
            raise RuntimeError(f"Pinned upstream content mismatch: {relative}")
        text = original.decode("utf-8")
        for old, new in spec["replacements"]:
            if text.count(old) != 1:
                raise RuntimeError(f"Expected one source anchor in {relative}: {old!r}")
            text = text.replace(old, new, 1)
        compile(text, relative, "exec")
        patched = text.encode("utf-8")
        current = path.read_bytes()
        if current not in (original, patched):
            raise RuntimeError(f"Unreviewed local changes in {relative}; refusing to overwrite")
        rows.append((path, original, patched, {
            "path": relative, "before_sha256": sha(original), "after_sha256": sha(patched),
            "status": "already_applied" if current == patched else "pending",
        }))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", UPSTREAM, head], check=True)
    rows = build_plan(repo)  # Verify every target before the first write.
    if args.diff:
        for _, original, patched, row in rows:
            print("".join(difflib.unified_diff(
                original.decode().splitlines(keepends=True), patched.decode().splitlines(keepends=True),
                fromfile="a/" + row["path"], tofile="b/" + row["path"],
            )), end="")
    if args.apply:
        for path, _, patched, row in rows:
            if row["status"] == "already_applied":
                continue
            mode_bits = path.stat().st_mode
            fd, temporary = tempfile.mkstemp(prefix=".wan-headonly-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(patched)
                    out.flush()
                    os.fsync(out.fileno())
                os.chmod(temporary, mode_bits)
                os.replace(temporary, path)
                row["status"] = "applied"
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        for path, _, patched, _ in rows:
            if path.read_bytes() != patched:
                raise RuntimeError(f"Post-write verification failed: {path}")
    result = {"upstream": UPSTREAM, "checkout_head": head, "repo": str(repo),
              "mode": "apply" if args.apply else "check", "files": [row for *_, row in rows]}
    payload = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
