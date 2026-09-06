# RLT-DVAC-Pure 单卡 smoke v2

- 代码：`codex/rlt-dvac-pure-reference-bc@cb88e9c5d817a248fef6d6ee02127b874ab851a8`
- 结果：1/1 cycle自然完成，`exit_code=0`；8 trajectories、2 success、8 critic + 4 actor updates。
- 语义：成功episode仍模仿π0 reference；`executed_target_ratio=0`；C10内DVAC权重mean-one。
- 权重：p05/mean/p95=`.448/1.000/1.499`，ESS=`.912`，top20 mass=`.277`。
- 资源：RAM峰值82.151 GiB；GPU0峰值20.711 GiB；GPU1空闲；OOM/fatal为0。
- checkpoint：Step1完整保存。没有启动480-step formal。

原始证据位于本目录的 `metrics.log`、`smoke_summary.json`、`resources.csv`、resolved config、日志尾部和4份trace。
