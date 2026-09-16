# RLT pair closeout before the 800-round cutover

Both old runs were stopped on explicit user request. GPU6 tau1.5/success-scale1 continues from a complete checkpoint in a separate output run; GPU7 tau1/success-scale2 is replaced by fresh tau2/success-scale1. Original stopped-run history is preserved separately, without concatenating discarded post-checkpoint observations into the continuation.

- [π0 RLT · DVAC τ=1.5 · 成功倍率1](rlt_t15/README.md): collection through R443; latest checkpoint R425.
  本包保留GPU6原τ1.5/倍率1运行的全部已记录历史；后续从完整CP425恢复到总轮数800，新run为 /data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t15-half4env-resume425to800-phys6-20260916-v1。CP425之后若有旧尾部记录仍保留作证据，但不拼入续训主曲线；续训主曲线应取旧run R1–425与新run R426起。模型、优化器、target-Q、回放及trainer状态的恢复验收以新run STARTUP_VERIFIED为准，本次归档仅检查marker及组件文件元数据。
- [π0 RLT · DVAC τ=1 · 成功倍率2](rlt_t1s2/README.md): collection through R425; latest checkpoint R425.
  本包为GPU7已停止的τ1/成功倍率2原运行；新实验τ2/倍率1从空Stage2状态fresh训练800轮，新run为 /data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t2-half4env-fresh800-phys7-20260916-v1。两组指标分别保留，不串接曲线；旧checkpoint与回放留存服务器。

Each ZIP includes raw events/logs, complete scalar CSV/JSON, five PNG/PDF plots (raw, MA10/20/50, fixed20), configuration, provenance, exact stop receipts and stat-only checkpoint inventory. Models, optimizer and replay payload are excluded and kept on the server. These ZIPs preserve evidence and cannot restore training by themselves.
