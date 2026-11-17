# DSRL-NA source alignment

This revision corrects three differences in the pi0.5 integration:

* The actor, action critic and noise critic now receive the frozen VLA's 2048-dimensional last valid prefix token, in addition to the existing image and proprioception encoders. Rollout reuses the prefix KV cache for action generation. Current features are stored with the executed transition; next-observation features are computed from the aligned next observation. Features enter the small MLP directly, without another compression bottleneck.
* Every gradient update draws a new replay minibatch. The previous implementation repeatedly resampled a 256-row decoded pool. That reduced coverage and is removed. VLA inference is only microbatched; gradients and targets remain independent across updates.
* pi0.5 includes discretized state in its prompt tokens. Reusing current tokens for `next_obs` left that state stale. Replay now preserves the original UTF-8 instruction and tokenizes each observation using its own normalized state. The tokenizer test verifies that a changed next state produces the expected new tokens.

Sources pinned for review:

1. [pi0 NA training loop](https://github.com/arayabrain/dsrl_pi0_na/blob/afcb58f2bd95a2e3771db359b8f98eb8f4f9ebee/examples/train_utils_aloha_sim_na.py): fresh replay draws, real-action Bellman targets, noise-value distillation, and prefix representation in the state.
2. [Prefix extraction](https://github.com/arayabrain/openpi_pub/blob/5a703af8579b7d6f5569d9de772637a99a08474e/src/openpi/models/pi0.py): `get_prefix_rep` selects the last valid token. The PyTorch port uses the same selection, including non-contiguous masks.
3. [Example NA configuration](https://github.com/arayabrain/dsrl_pi0_na/blob/afcb58f2bd95a2e3771db359b8f98eb8f4f9ebee/examples/scripts/run_aloha_sim_na.sh): action-Q/noise-Q/actor update ratio 15/5/5, batch 128, learning rates 3e-4/1e-4, ten critics.
4. [Official OpenPI tokenizer](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py): pi0.5 encodes discretized proprioception in the task prompt, so next-observation tokens must use the next state.

The RLinf port retains its environment-specific action conversion, 20-step execution chunk, 1600 independent noise coordinates, FP32 small networks and the two existing actor parameterizations. These are adaptations to pi0.5/RoboTwin, not a verbatim reproduction of the ALOHA configuration. The single-GPU budget is four 15/5/5 cycles per collection round; pretraining retains the previous 1000 action-Q update threshold. Actual replay samples and update counts are logged separately. Critic updates precede the actor within each cycle.

Validation: `test_contract.py` covers targets, action/transition alignment, replay resume, noise bounds and image packing. `test_source_alignment.py` covers exact prefix selection, independent replay sampling and state-token alignment. GPU smoke exercises rollout, feature replay, both critics, actor optimization and checkpoint resume before a fresh formal run.
