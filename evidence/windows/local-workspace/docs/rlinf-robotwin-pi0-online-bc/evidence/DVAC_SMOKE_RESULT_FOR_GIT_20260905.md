# Action-level online BC DVAC: completed two-round GPU7 smoke

Source `736b1416f37f32034efd3924a0fc5f5fca611012`, branch `codex/sz-pi0-online-bc-dvac`, baseline `385d4e75`. 22 tests plus native resolved-config validation passed. Model/task/training budget unchanged from the validated BC smoke; method fields, GPU and required paths differ.

Run: `/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1`. 2026-09-05 18:53:14–19:31:36 CST, exit0, 38m22s. GPU7 released. No formal DVAC run launched.

Two complete collection/update/sync/eval/save rounds: 64 train attempts, 20 Adam updates, 20480 query presentations, 64 fixed eval attempts, two checkpoint generations. Train successes24/32 then21/32; fixed24/32 then25/32; FM loss0.022524 then0.015727. Not evidence of learning improvement.

Same M4 sampler provides last L3 endpoint variance at each of H50/D14 actions without extra model/RNG calls. New action positions5200/5350; first success queries72 with unit weights, second63 queries calibrated using preceding-round stats. Second-round weights mean1/std0.15960/range0.60236–1.78022. CPU artifact readback reconstructed every new weight exactly and verified old weights and replay RNG restoration. Native/full/replay/learner/DVAC sidecars exist in both generations; learner steps10/20. No full production-worker/model/optimizer restart test was performed.

Sampled GPU peak79576MiB=77.71GiB, Env FD1003, Env RSS74.96GiB; host available RAM minimum1203.51GiB with other workloads. No searched errors. This does NOT establish long-run resource stability: original BC had previously OOMed on round6 after its first eval; its code was not modified by this DVAC run. Subsequently authorized baseline eval8x4 smoke runs separately on GPU6, so these two smokes are not a strict method-performance comparison.

Evidence: [verification](DVAC_SMOKE_VERIFICATION_20260905.json), [status](DVAC_SMOKE_STATUS_20260905.json), [22-test/config transcript](DVAC_TEST_CONFIG_RETEST_20260905.txt), [resolved config](DVAC_SMOKE_RESOLVED_20260905.yaml), [config delta](DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json), [contract](GPU7_DVAC_SMOKE_CONTRACT_20260905.md), [learning chart](success.png), [resource snapshot](resources.png). Large model/replay artifacts stay in the run directory and are not in Git. User's latest instruction is to finish smokes only; no automatic formal launch, cleanup or storage migration.
