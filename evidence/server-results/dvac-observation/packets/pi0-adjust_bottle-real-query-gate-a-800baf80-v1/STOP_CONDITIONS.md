# Gate A stop conditions

## Before launch

- Stop if RLinf HEAD/branch/upstream is not the locked reviewed commit or the worktree is dirty.
- Stop if the RoboTwin compatibility commit, pinned runtime, model, resolved config, or GPU 2 differs.
- Stop if the exact output path already exists, GPU 2 has another compute process, or live RAM/disk is insufficient.

## During launch

- Stop this packet-owned process on traceback, CUDA OOM/illegal instruction, action/RNG parity failure, trace shape/formula failure, or the 900-second timeout.
- Preserve failure evidence; do not overwrite the output path and do not stop PPO or another user's process.

## Success

- Natural exit 0 and marker `PI0_DVAC_REAL_PARITY_OK`.
- `parity.json` and `parity.npz` exist under the new exact output path.
- All action, post-RNG, endpoint, shape, and `z=x-t*v` checks are true.
