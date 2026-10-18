# CP180 high/low threshold sweep on Shenzhen3

The previous single high/low points had different realized takeover rates. This sweep estimates success-versus-realized-rate curves without changing the policies or native control protocol.

- Authorized GPUs: Shenzhen3 physical6/7, B32 per GPU, four batches and128 paired episodes per condition. Inference takes priority over the two per-card RLT jobs; preserve their latest complete recovery files and restore them afterward.
- Fifteen new calibration target fractions per rule:5%,10%,15%,20%,25%,30%,40%,50%,60%,70%,75%,80%,85%,90%,95%. Thirty conditions,3840 new episodes. Existing one-third, full-weak, full-strong and random points are reused. The user clarified18h is a rest window, not a hard limit; measured prior batches imply22–24h for this queue.
- Use only the existing249 decisions from64 independent weak-policy calibration episodes. High gates use quantile1-p; low gates use quantile p. Freeze all thresholds before evaluating. Plot the realized rates, not these calibration targets. No success-based threshold selection or seed filtering.
- Preserve original/GRPO U tau1 CP180 weights, paired formal seeds/noise, N10/H50/C50, same-observation N10/N5 uncertainty,200 submitted actions, native success, full execution video, endpoint signals, and graphics scoping. CP180 is already protected at models/protected-eval/pillbottle/grpo_u_tau1_cp180/full_weights.pt.
- Use the previously accepted finite lane unchanged. Each batch retains its3600s ceiling and600s stall check, no automatic retries. A failed lane releases to RLT and the peer aborts at its next barrier. No18h cutoff was added.
- Range-first order:10,90,30,70,50,5,95,20,80,40,60,15,85,25,75 percent, alternating high/low at each level. This order is fixed before results.

Runtime: /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-cp180-threshold-sweep-20261018

Transaction: /data/chenyiteng/deployment-20261018/cp180-threshold-sweep-sz3-v1

Output: /data/chenyiteng/results/pi05-policy-eval/pillbottle-cp180-threshold-sweep-20261018-v1

Preparation: existing server venv runs tools/pi05_policy_eval/prepare_sz3_threshold_sweep.py SOURCE_COMMIT. The cutover script reuses the accepted identity/UID/start-time, owner lock, namespace cleanup, latest-complete-checkpoint, new RLT runtime, and per-card lane lifecycle. It changes only6/7 slots;4/5 and shared Ray remain running. The unique cutover-complete.json receipt prevents replay.

Validation: server-side CPU gate and native dispatch tests, paired configuration delta checks, then first batch progress plus compute/graphics placement. This is a diagnostic sweep on previously evaluated seeds; final paper confirmation should use fresh seeds.

Publication audit: prior evaluation source and execution notes exist on the published CP180 branch. Complete policy-evaluation result logs were not tracked there. This turn archives available compact records for original budget/CP140, RL/C20, failed v2, and CP180 takeover. Raw videos, endpoint arrays and model weights stay on servers; plot-ready counts and original text/JSON records are retained separately.
