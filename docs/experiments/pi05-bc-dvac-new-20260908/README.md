# Online BC DVAC new: complete-batch two-level weighting

Current BC-DVAC used a five-round calibration window and replay-frozen [0,5] weights. The new opt-in `+bc_dvac=two_level_batch` keeps behavior-policy variance V, computes detached weights over each complete sampled optimizer batch, then splits microbatches. Legacy config groups and the old mapping helper remain available.

Two branches start from log(V+1e-12): within each chunk, masked MinMax is centered by valid action-dimension counts; raw masked mean(logV) is separately MinMax-scaled and query-centered across the full batch. Their factors multiply. Both strengths default to 1 and accept [0,1]. Duplicate replay draws count with their sampled multiplicity. Final batch mean coefficient is 1; this does not imply equal gradient norms. No history, sigmoid or frozen replay W is used in new mode.

BC loss remains native masked flow-matching. Missing/invalid V and empty supervised queries fail explicitly. Only one actor rank is supported; independent `dvac_new.pt` metadata rejects incompatible resumes before model loading. Signal-tail/model/seed settings are recorded in resolved config. Formal debug defaults to 0.

## Verification

- Base: `ad3da329270960f4ad378842e7138bbde29c521e`; isolated branch `codex/sz-pi05-online-bc-dvac-new-20260908`.
- Smoked source: `57262ba2eb6a9b9567b1e547a10e3584c54d4866`; publication changes evidence only.
- Server: 86 tests passed (44 new mathematical/integration tests plus 42 legacy BC/DVAC regressions).
- GPU7 smoke: 2 collection rounds, 4 attempts/U5, full batch1024/micro32, M10/C50, LR2.5e-5, fixed original model and empty success pool. No fixed evaluation or large checkpoint saving.
- Runtime config matches prepared config. Actual training loss and gradient diagnostics are finite; both factors participate, full-batch mean coefficient is 1. Two recorded full-batch tensors were recomputed exactly. The smoke exits and releases its owned actors. GPU6 clean BC independently finished its original R100 budget with exit0 at 22:24:54 CST, so the protection check allows only its three exact former processes to exit; Prism and shared Ray remain intact.
- Exact numerical results, timestamps and protection checks: [smoke-verification.json](smoke-verification.json). Configuration, command, source hashes and tests are adjacent.

## Preserved old 4/U5 reference

User-authorized stop of GPU7 old DVAC at R94/100 on 2026-09-08 22:05 CST. GPU6 clean BC continued. Paired fixed R5-90: old DVAC wins15/ties2/losses1, checkpoint mean42.88% to49.48% (+6.60pp); same seed and fixed32 scenes, not independent checkpoint trials.

[Closeout ZIP](bc4-u5-closeout.zip) contains 8 PNG/SVG charts, full scalar/event/log/config/source evidence and checkpoint inventory. SHA256: `d0f51ae8518a080fff44f081514ef486ad3f400869d189b19eecf003bb21a73f`. Large weights/replay images are not copied. [Retention manifest](old-dvac-retention.json) adds R60/R90; the prior R60/R80 retention also remains in force. Old source branch and directory were not overwritten.

Formal training has not started. Initial model/empty pool, 4/U5, batch1024/micro32, LR2.5e-5 and length-filter-off are the proposed controlled baseline; formal round budget and weighting strengths remain for user discussion.
