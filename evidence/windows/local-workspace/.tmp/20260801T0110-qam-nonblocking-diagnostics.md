# QAM formal incident: heuristic diagnostics must not block training

On 2026-08-01, a RoboTwin/pi0 QAM formal run was interrupted because an
adaptation-only frozen-prefix similarity diagnostic was implemented as a hard
exception. The relative-L2/cosine thresholds had no requirement in the QAM
paper or locked official code. Treat this as a serious engineering incident.

Long-term rule requested by the user:

- Do not invent heuristic safety gates or let them block formal training.
- Keep implementations concise and trace every algorithm choice, threshold,
  and deviation to the paper, official code, an existing project contract, or
  explicit user approval.
- Definite invalid states such as missing/malformed fields, shape or checkpoint
  contract mismatches, and NaN/Inf may fail fast.
- Empirical diagnostics such as relative-L2, cosine similarity, gradient size,
  or action sensitivity must default to metrics plus at most a bounded warning;
  they must not stop, skip, or change training unless the user explicitly
  approves that gate after its basis and consequences are shown.
- Before formal launch, distinguish method requirements, structural contracts,
  and optional diagnostics. State which conditions can terminate the run.
- A requested completion time is an estimate unless the user explicitly calls
  it a hard deadline; prefer a justified step budget over an unapproved
  wall-clock kill.

Applied QAM repair: commit 9e2abc04 makes prefix relative-L2/cosine threshold
crossings nonblocking while retaining structural and non-finite failures.
