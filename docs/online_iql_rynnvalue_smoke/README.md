# Online RynnValue IQL: implementation and GPU6 validation

Independent branch `codex/sz-pi05-online-iql-rynnvalue-20260909`, clean BC base `01d770db3988da7862454e97434d4ff08f726fa2`; tested production commit `4d7877ebdc58558636be33b06dcec11fdf6f17db`.

Rounds 1–10 retain successful-only BC while Q/V learn all outcomes. Round 11 automatically enables IQL weighting on full replay. Actor, critic, target, optimizers, replay and sampling RNGs continue across the switch. See [method implementation](../online_iql_rynnvalue.md).

Actor controls remain 4 attempts/round, 5 updates, batch1024/micro32, original Adam/LR2.5e-5/constant/grad clip1, original model and empty replay, no success length filter. Q/V use independent FP32 Adam3e-4, batch min(64,N), one update each per actor slot, expectile .8, beta10 and max weight100. Frozen RynnValue-8B provides Phi=-remaining seconds. Reward is success0/1 + .1*(.99*Phi_next-Phi_now), with absorbing terminal Phi_next=0. The online schedule, sparse anchor and per-query discount are documented adaptations of the upstream offline IQL path.

Server checks: 57 passed, including same-parameter forward parity with the locked official Flax encoder. The first real restart exposed CPU-offloaded FSDP parameters at checkpoint load; the worker now loads parameters and optimizer before strategy restore. Four offload-state tests cover the fix. Earlier v1 evidence remains on the server; no failed run is reported as passing.

The final GPU6 smoke uses warmup1 solely to exercise both stages in two separate processes. R1 exited0 and saved; a fresh process loaded that checkpoint and completed R2 with exit0. Each round kept 4/U5 and actor1024/32; fixed evaluation was disabled and saving occurred each round.

| Verified item | R1 success BC | R2 after restart, IQL |
|---|---:|---:|
| Cumulative attempts | 4 | 8 |
| Full replay query transitions | 15 | 30 |
| Cumulative successes | 1 | 2 |
| Actor updates | 5 | 10 |
| Q updates / V updates | 5 / 5 | 10 / 10 |

Independent CPU verification checked the exact replay prefix across restart, stored true terminal frames bound to scoring requests, reward recomputation, full 50x14 proposals, optimizer/target finiteness, phase and finite bounded weights. This validates the recorded training chain; it does not establish policy gains or simulator bitwise continuation across restart.

The owned scorer was stopped, GPU6 has no compute processes, both smoke namespaces are empty, and the protected GPU4/5/7 experiments and shared Ray remained available. No formal training was launched. Formal warmup remains10; do not copy smoke overrides into a formal run.

Evidence: [verification](smoke-verification.json), [phase scalars](smoke-scalars.json), [finish receipt](finish-receipt.json), [tests](tests.txt), [R1 config](bc-resolved.yaml), [R2 config](iql-resolved.yaml). Other users' raw audit records and large checkpoints are excluded.
