# Dojo success-only evaluation: six tasks on one GPU

Share native JAX weights while keeping per-task observations, sampling parameters, random state, RPC ports and simulator processes separate. Preserve four paired episodes per task, H50, C10/50 and N1/10. Continue completed results and skip/clean ordinary failed jobs. GPU3 is released after the scan; only GPUs4–7 retain RLT fallbacks.

Two isolation/resume tests passed. All six tasks advanced with compute and graphics on physical GPU3; one four-episode task completed and automatically continued with N10 while the other lanes kept running. Measured memory 72.0 GiB and GPU utilization 81–100%; the short action-throughput sample is not a matched-task speedup benchmark. See receipt.json and ../../dojo_u_video/14_PARALLEL_EXECUTION_20261025.md.
