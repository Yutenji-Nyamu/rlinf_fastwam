# RLinf π0.5 adjust_bottle: five-method smoke

2026-10-27 20:54 CST. All five methods passed 12-step save, full resume to step 14, native one-call N1 export and 64 real environment episodes. The original multitask FQL resumed on SZ1 GPUs 6/7; existing direct distillation stays on 4/5, followed by per-card RLT fallbacks.

|方法|每卡batch / 全局|训练峰值GiB/卡|秒/更新（短测）|CP14真实执行|
|---|---:|---:|---:|---:|
|直接端点|224 / 448|72.8|4.87|47/64|
|CP|80 / 160|74.3|1.95|47/64|
|DMD new|192 / 384|72.6|6.03|45/64|
|MFD|192 / 384|72.6|10.42|37/64|
|固定老师FQL|224 / 448|72.9|8.66|38/64|

This is a functional smoke, not a converged-method comparison. Protocol: native fixed RLinf teacher, H50/C50, N5 teacher versus N1 student, original normalization and delta-joint/absolute-gripper interface, clean adjust_bottle, 200 actions, no video. Frozen VLM; method-specific action expert, auxiliary or Q training. Native baselines: N5=111/128; N1=98/128.

- `summary.json` / `summary.csv`: compact results and measured batches.
- `deployment/`: exact configs, capacity decisions, smoke/export acceptances, allocator retry and return receipt.
- `runs/`: light training metrics and checkpoint metadata; no actual weights or replay images.
- `runtime/owner-live-external.py`: deployed RLT controller with live owner refresh.
- `restored-fql/`: original multitask FQL config and resume receipt.

FQL smoke uses only 32 complete episodes / 102 native C50 transitions; this is not the formal training dataset. Evaluation batch 64 failed in renderer initialization; 32 is the measured working setting. MFD B192 initially OOMed with large unused reserved memory; the same batch passed with expandable segments. The failed trial is retained. All 26 targeted CPU tests passed.

Full Chinese explanation: [result notes](../../small_flow_distillation/11_SINGLE_METHOD_SMOKE_RESULTS_20261027.md).
