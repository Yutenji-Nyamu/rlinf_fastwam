# GPU6：原BC eval8×4容量smoke合同

用户09-05授权评估改8×4，GPU6与GPU7独立并行；随后明确“先完成smoke，暂不放正式”。本合同不授权formal、迁移存储或清理。GPU7现有DVAC、GPU4/5 Sidney与shared Ray保持。

## 变更与完整配置

原BC `385d4e75` → 已测试推送`a8764944`，只改基线YAML三个评估叶值及原有测试：eval16→8、串行2→4、固定初态分批2→4。没有生产Python改动。11/11服务器测试通过；真实配置校验及原32个固定ID有序覆盖通过。[resolved](BC_EVAL8_SMOKE_RESOLVED_20260905.yaml)、[逐叶差异](BC_EVAL8_DIFF_20260905.json)、[四批固定ID](BC_EVAL8_FIXED_SEEDS_20260905.json)。除上述三个叶值，只有输出路径和名称不同于原v7 smoke。

GPU6，原adjust_bottle π0 SFT，空池，非旧formal/smoke续训；train32×1，micro32/global1024/U10，M4/C50/D14，expert-only、无图像增强、demo0、原LR2.5e-5/Adam/FSDP/渲染均不变。

两轮共64条训练尝试、20次Adam、20480个chunk呈现；每轮8×4固定评估，共64条；每轮保存native/full/replay/learner，共两代。第二轮保留以覆盖首次评估后再采样的容量边界。不是100轮资源稳定性证明。

## 精确命令、输出与停止条件

执行环境与export完整见[wrapper](../../../local_scripts/bc_eval8_wrapper_20260905.sh)。等效训练入口：

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  runner.logger.experiment_name=pi0-bc-u10-eval8x4-smoke-gpu6
```

输出`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8`，拒绝覆盖已有目录。使用共享Ray6389上的独立namespace/code working dir，不重启Ray。

19:17 GPU6无compute仅11MiB，主机RAM available约1.2TiB，memory PSI0；/data约422GiB可用，足够本次两代checkpoint约34.5GiB及另一项smoke/当前Sidney近期保存，但不据此承诺所有新正式长期保存。预计约40—50分钟（8×4串行评估可能更慢），90分钟硬上限；每5秒只读记录本卡显存、主机RAM/PSI、本人Env FD/RSS。

首个OOM/fatal/NaN或接口错误结束验收，不自动改参重试；只处理本run，不动别卡任务。成功需2轮/20更新/两次完整32固定评估/两代checkpoint实体、有限loss/grad。通过后依用户最新决定停在smoke，不启动formal、不切/home。
