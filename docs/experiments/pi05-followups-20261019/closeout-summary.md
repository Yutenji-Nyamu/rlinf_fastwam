# EXPO Clean B64 closeout, 2026-10-19

User-authorized early closure at 57,892/60,000 physical actions, 460 episodes and1,408 complete updates. Task: turn_switch. Latest completed fixed20 evaluation is episode450,17/20=85%; recent5 mean69%, initial8/20=40%, historical peak19/20=95%. The repeated fixed20 curve supports improvement in this run, not an independently measured population gain.

The stop signal arrived during episode460 evaluation and was wrapped by the native environment as RuntimeError (exit1). That evaluation is incomplete and excluded. The prior complete learner checkpoint at update1408 is finite, with no pending learner calls; GPU cleanup succeeded. Do not assign episode450's85% to the final update1408 weights.

The final complete checkpoint is protected at /data/chenyiteng/models/protected-eval/turn_switch/expo_clean_b64_20261019_final/checkpoint.pt. It is an EXPO checkpoint on another task, not a pillbottle candidate. Original results remain; no weight, replay or video is uploaded.

The2.1MiB archive contains plot-ready evaluations, configuration, events and a bounded driver-log tail. Source branch also contains the new inference implementation, tests and execution plan. No per-file rehash was performed.
