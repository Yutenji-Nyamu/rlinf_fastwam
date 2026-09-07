# GPU6/7 π0.5 BC seed42正式切换合同

用户2026-09-06明确授权两旧run停止、轻量产物ZIP/push，随后从头启动BC与DVAC [0,2]，其他参数不变。本次不额外增加训练前评估或重复模型/仿真smoke。执行入口与逐操作仍为[本轮账本](BC_SEED42_CUTOVER_LEDGER_20260906.md)。

## 实际完整配置与命令

- GPU6：[完整resolved](SEED42_BC_resolved.yaml)、[实际wrapper](SEED42_BC_wrapper.sh)、[逐叶preflight与展开命令](SEED42_BC_preflight.json)。
- GPU7：[完整resolved](SEED42_DVAC_resolved.yaml)、[实际wrapper](SEED42_DVAC_wrapper.sh)、[逐叶preflight与展开命令](SEED42_DVAC_preflight.json)。

wrapper逐字源自各旧run实际启动文件，只替换run/experiment名称、添加`+rollout.seed=42`，DVAC组`bounded_half→default`（只有alpha0.125→0.25的方法设置变化）。两者都保留原48h timeout，原venv/RoboTwin/模型路径，不使用Fast shim、不升级依赖。

```bash
# GPU6: root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
# GPU7: root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac
# 完整环境变量与绝对run路径以各实际wrapper为准。
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  +online_bc_model=pi05_sidney +rollout.seed=42 \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=1000 \
  runner.logger.experiment_name=<各wrapper中的精确名称>
# GPU7额外 +bc_dvac=default，包含placement7和DVAC字段。
```

输出共同根：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/`。

- BC新run：`pi05-pillbottle-bc32x1-b1024-u10-m10-seed42-eval8x4-gpu6-formal100-20260906-v2`。
- DVAC新run：`pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-seed42-w0to2-eval8x4-gpu7-formal100-20260906-v2`。

## 保持相同的预算与模型

| 项目 | 两项相同 |
|---|---|
| 起点 | Sidney pi05 e49e2ab native原SFT；不加载旧BC权重；resume_dir/ckpt_path=null，空成功池 |
| 任务/观察 | move_pillbottle_pad；aloha-agilex；三相机224；原绝对qpos/MEAN_STD/state tokens |
| 动作/采样 | M10/C50/14D，H200；标准高斯初始noise＋原ODE；不启用GRPO SDE探索 |
| 训练采集 | 单卡32并行×1串行，每轮32条；100轮3200次尝试；最多12800新query |
| BC | 累计成功池、均匀有放回；demo_weight0；只训expert及相关投影/条件层，VLM冻结 |
| 优化 | micro32/global1024/U10；每轮最多320 micro、10240 chunk呈现、10 Adam；全程最多1000 Adam |
| 优化器 | LR2.5e-5 constant，warmup0，原AdamW/clip等逐叶不变 |
| 图像/精度 | 训练增强关闭，原生混合精度，不强制全BF16 |
| 评估 | 每5轮，8×4覆盖原同32固定初态；不额外增加Step0预算 |
| 保存 | 每10轮，共10代；原local_shard/full/replay/learner；无自动删除规则 |
| 原训练seed | actor1234、env0及原seed表保持；训练继续逐轮换题 |
| 新rollout seed | 两者42；按逻辑rank偏移（本次两者rank0，实际均42） |
| 评估RNG | 整轮评估重置42，返回时恢复训练Python/NumPy/Torch CPU/CUDA随机状态 |

固定的是序列，不是固定一张noise或用全零noise。seed在模型/采样设置初始化与offload后设置；每query沿现有调用继续生成新的高斯。默认不配置seed时，原worker行为不变。本次未增加rollout随机状态的精确断点续训保存；从头启动可配对，未来如要求逐位恢复须另核对恢复协议，不能宣称本轮已完成这一项。

## DVAC唯一方法增量

沿已验证action-level信号和FM误差加权，不改为chunk-level权重，不加reward/adv/critic。仅alpha从0.125变0.25：

```text
z = clip((log(V+1e-12)-past_mean) / max(past_std,1e-6), -2, 2)
w = stopgrad(1 + 0.25 * (z - valid_mask_weighted_mean_H(z)))
```

有效动作位置加权均值1，保证范围[0,2]，不强制拉满端点。tail3、过去5轮标定、先标注后更新统计、首轮w1、入池后固定w、仅成功数据训练均保持。V+w只为小附加张量，不增加模型forward。

## 资源与停止条件

16:17 compose /data余1358GiB。按两BC各约250GiB和GRPO余下约537GiB估计，预留总额约1037GiB；未授权移动/删除旧权重，也不保证其他用户不会继续写盘。启动前再次确认GPU6/7无compute进程、RAM余量及GRPO身份保持。

旧两BC已在同容量配置下完成69/68轮；它们的长程占用不能代替新run实测。单卡训练/评估既有峰值约74—79GiB边界，沿现有资源observer每5秒记录GPU/RAM/FD。只复用原no_shard/offload/FD4096布局，不改并发以绕过问题。

完成100轮或原48h上限结束；fatal/OOM/结构性错误不自动降预算或循环重启。精确owned driver/namespace才可操作，GRPO/shared Ray/其他用户保持不变。只确认正常启动后回报，不设置自动化、不长期等待学习结果。
