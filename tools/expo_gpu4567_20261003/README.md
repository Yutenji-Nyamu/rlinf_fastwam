# SZ2 EXPO graphics scope and maintenance owner

This is the reviewed, deployed 2026-10-03 repair for physical GPUs 4–7. See the [deployment report](../../docs/methods/expo-ft/GPU4567_BINDING_20261003.md) and its evidence for exact host, process identities, source hashes and acceptance.

`bootstrap/sitecustomize.py` and `graphics-profile.json` implement per-process graphics visibility. The bootstrap requires the validated server-side `scope.json` through `RLINF_EXPO_GPU_SCOPE_MANIFEST`; CUDA visibility must match its ordered GPU UUIDs. The profile mask uses NVIDIA device minor numbers, verified on this host.

`handoff_gpu4567.py` and `maintenance_owner.py` preserve the existing borrow cycle, lock, pending learning calls and RLT return chain. These are identity-pinned, one-time maintenance operations already executed on SZ2. Do not replay the handoff or start another owner against the active cycle. A later migration requires a fresh inventory and contract.

The diagnostic scripts are historical server-side probes. They require the pinned environment and a free authorized GPU slot; they are not production startup commands. CPU tests were run on the server, including the existing next-six queue gate. This directory contains no credentials, checkpoints or replay data.
