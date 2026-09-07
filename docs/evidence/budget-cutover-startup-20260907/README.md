# Authorized fresh-run startup 2026-09-07

bc started from original Sidney SFT. Runtime code/config and physical GPU checked; shared Ray and other jobs preserved. BC pair uses 8 attempts/U2, seed42, micro32/global1024; original per-chunk DVAC mapping [0,2] unchanged. GRPO uses original256 attempts/U2, baseline chunk clip with logprob_st and positive-only scope [0.5,1.5].

Snapshot 2026-09-07T05:19:51.234886+00:00. GPU/model smoke was not requested and was not run. GRPO CPU20 tests and actual prior-array scope check passed. Startup is not a long-run stability/recovery claim. Runtime source-head remains 064b990cb3d7daeb364e98a20e81ae10fd7d6061; this commit adds documentation only.

Old lightweight results are on the respective closeout branches; no model, video or replay deletion was performed.
