# First-rollout startup evidence

Started 2026-09-06 15:08:30 CST from the original Sidney SFT on GPU4/5. Source is 30afdf8c, not the later evidence-only commit. Both actor/rollout/environment ranks were independently inspected as using the new worktree, job ff010000 / namespace RLinf. Actual runtime configuration matches the prepared config except four framework defaults, all equal to the Control runtime. 14 CPU tests passed before launch.

This frozen snapshot confirms entry into the first rollout, not completion of a training round, the first nontrivial DVAC-weighted update (warmup=1), or long-term stability. The baseline and BC tasks were not restarted or modified. See snapshot.json for time/resources/errors and the run directory for newer live truth. The Control was intentionally stopped at168; checkpoint160 remains intact. Its ZIP and light archive are on codex/sz-sidney-pi05-current-rlinf commit1d015a2a.
