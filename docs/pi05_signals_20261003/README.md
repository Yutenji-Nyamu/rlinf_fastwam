# pi0.5 action-signal collection

Seven new scores plus DV, collected during unchanged B16/H50/M10 RoboTwin inference; U-GROW adds a same-noise M5 side solve. See [signal definitions](signal-plan.md), [execution notes](execution.md), and [tools](../../tools/pi05_signals/README.md).

Real smoke: two tasks, 32 episodes and 32 videos, exact main-action/full-chain/RNG parity, 18 expert hooks per forward, all scores recomputable with zero validation warnings. Two failed startup attempts are retained (host Vulkan path and raw-image-size validation). Formal inference has been dispatched; this is not a claim of training effectiveness or completion of all 1600 episodes.

100 original batch configs and compact acceptance/dispatch evidence are included. NPZ/video payloads remain on the server; their byte sizes and hashes are indexed. Private paths in evidence are replaced by generic roots.
