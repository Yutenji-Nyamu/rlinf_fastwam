# GPU6 U10 最终运行合同

2026-09-05用户明确确认U10并授权GPU6正式100轮；沿既有授权先完整同容量smoke，再原SFT/空池正式。无须重复精度/FSDP/FD旧probe。

## 配置与预算

完整配置：[smoke resolved](U10_SMOKE_RESOLVED_20260905.yaml)、[formal resolved](U10_FORMAL_RESOLVED_20260905.yaml)。精确环境/命令：[统一wrapper](../../../local_scripts/bc_u10_wrapper_20260905.sh)。

| 项目 | 同容量smoke | 正式 |
|---|---|---|
| GPU | 物理6，单H100 80GB；7不占用 | 相同 |
| 模型/起点 | adjust_bottle π0原SFT @92684e50，resume=null、空成功池 | 独立重新从原SFT/空池，不承接smoke |
| 模型/训练范围 | M4、C50/H50、D14/内部32、3相机；冻结VLM，仅expert及相关投影；原生混合精度 | 相同 |
| 方法 | 累计成功query回放、均匀有放回、原生FM；demo_weight0、图像增强off、无teacher/Q/V/DVAC | 相同 |
| 每轮采集 | 32并行×1串行、horizon200，最多128query/6400指令action槽 | 相同 |
| 每轮更新 | micro32/global1024/U10，累积32；10240次query呈现 | 相同 |
| 优化器 | LR2.5e-5恒定，无warmup；Adam(.9,.95)、eps1e-8、wd1e-10、clip1 | 相同 |
| 外轮 | 2 | 100 |
| 总采集 | 64尝试、≤256新query、≤12800指令action槽 | 3200尝试、≤12800新query、≤640000指令action槽 |
| 总更新，池非空 | 20次Adam、640micro、20480次query呈现 | 1000次Adam、32000micro、1024000次query呈现 |
| 固定评估 | 每轮16×2＝原单worker32种子，总64 | 每5轮同32，总640 |
| 保存 | 每轮，共2代 | 每10轮，共10代 |
| 墙钟硬上限 | 90分钟/1.5 GPUh | 48小时/48 GPUh，不是预计完成时间 |

相对旧v6只改用户已确认项：U2→10及相应optimizer total；评估32×1→16×2，评估关闭auto-reset改用原生每批显式reset，并用opt-in固定种子循环保持原32不同初态。训练auto_reset、并发、采集量、模型、loss与所有学习参数不变。FD4096继续只作用本任务EnvWorker；不叠加环境offload，不更换OIDN/SAPIEN，不套Fast补丁。

种子基准是旧BC单worker32环境，不声称与旧GRPO双worker seed分区完全一致。实源测试两批拼回原32，并在第二次评估重新覆盖同一集合；其他环境默认固定种子行为不变。

## 精确命令与输出

实际入口（完整环境变量见wrapper）：

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=1000 \
  runner.logger.experiment_name=pi0-bc-u10-formal100-gpu6
```

smoke仅上述四个轮次/interval/total参数变为2/1/1/20、实验名换smoke、输出换以下smoke目录；源码及其余resolved学习/资源参数一致。

- smoke：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7`
- formal：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1`

精确launcher拒绝覆盖已存在run目录。wrapper独立PGID、run-scoped日志与数据，RLinf选独立namespace，连接现有Ray6389、代码工作树显式指定；不改CUDA全局编号、shared Ray或他人进程。

## 资源与停止条件

15:41只读：GPU6约11MiB且无compute进程，RAMavailable约1.4TiB、PSI0，/data余556GiB。显存未知峰值仍需本次smoke实测，不能用降低eval并发直接宣布通过；训练容量及既有精度/FSDP组件已测，不重跑孤立probe。

现有独立probe实文件：checkpoint约10.39GB、full_weights约8.07GB；按10代估计模型/优化器与完整权重约185GB（十进制），另需累计池checkpoint、成功archive等。smoke另留2代约37GB。本次没有清理旧产物；其他运行也消耗/data，启动前再核对余量。主机RAM/显存每5秒留资源记录，采样峰值不是连续精确峰值。

smoke必须完整两轮、20次真实更新、两次各32评估及两代非空checkpoint实体；确认有限loss/grad和正常退出，才启动正式。若错误/OOM/NaN/原生分配失败则不继续正式，不自动缩训练并发或反复重试；若评估仍不足，使用用户已允许的取消中评方案前仍明确报告该实际差异。只处理本run，不动共享服务。正常正式启动后回报，不新建长期监控或持续盯跑。

本轮不声称生产worker完整恢复或100轮原生稳定性已验证。提交/推送只限已验证本BC连贯改动；无升级、模型下载或跨分支改动。
