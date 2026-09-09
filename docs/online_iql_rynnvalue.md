# Online RynnValue IQL for the clean pi0.5 BC runner

Independent implementation based on clean `01d770db3988da7862454e97434d4ff08f726fa2`.
Keep `algorithm.loss_type=online_bc`; explicitly enable `algorithm.online_iql.enabled` to select the new worker and parallel full-transition collector.

- Rounds 1–10: ordinary success-only BC and independent full-replay Q/V training.
- Round 11 onward: sample successful and failed transitions, apply detached `min(exp(10*(min(Q1,Q2)-V)),100)` to the original per-query flow-matching loss.
- Keep 4 attempts, 5 actor update slots, global1024/micro32, original actor Adam/LR2.5e-5, and no success length filter. Q and V each update once per slot, critic batch `min(64,N)` without replacement within a batch.
- Frozen RynnValue-8B supplies raw remaining seconds at true query boundaries. Reward is success0/1 plus `0.1*(.99*Phi_next-Phi_now)` with Phi=-remaining seconds. Success and finite-budget failure terminate bootstrapping and use absorbing Phi_next=0; raw scorer outputs remain stored.
- A query action is the entire submitted 50x14 joint-target proposal. Do not crop its Q input using post-hoc execution outcomes. Current/next three-camera observations are preserved separately.
- Q/V use the locked upstream ResNet18 variant with stride8, GN4, spatial softmax and independent FP32 Adam. Q has hidden LayerNorm; V does not. Update order: V, Q, target Q, actor advantage from current Q/V. No SARM thresholds or DVAC weight normalization.
- Switching and restoring retain actor/Q/V/target/optimizers/replay and independent RNGs. Phase depends on completed rounds, not the count of successful actor updates.

Method-only defaults are in `examples/embodiment/config/online_iql/rynnvalue.yaml`. The smoke overrides warmup to1 solely to exercise BC round1, checkpoint, then a new process resuming into IQL round2; no formal training is launched by this packet.

Upstream source: [RynnValue](https://github.com/alibaba-damo-academy/RynnValue/tree/10e0d333f5f3811d0d130587e50f1faf48da49e5), commit `10e0d333f5f3811d0d130587e50f1faf48da49e5`. This is an online joint-action adaptation of its offline IQL path. The sparse0/1 reward anchor, per-query discount, success-only warmup and independent actor1024/critic64 batches are explicit adaptations.

Tests and smoke are executed on the Shenzhen server. Source packets are maintained on Windows. Existing BC, DVAC, SARM and GRPO branches are not merged or overwritten.
