# OpenDW 四卡 smoke：并行与每轮采样

2026-10-03。用户要求smoke并行接近正式、WM优先于RLT。此文件记录本轮规格和执行入口；实际进度以服务器唯一owner回执为准。

| 项目 | 旧LIBERO Wan | 已完成OpenDW小测 | 本次短测 | 本次完整回合信号测 |
|---|---:|---:|---:|---:|
| 物理GPU | 4–7 | 4 | 4–7 | 4–7 |
| 逻辑环境N / 组G / 采样轮R | 64 / 8 / 8 | 8或16 / 8 / 1 | 64 / 8 / 8 | 64 / 8 / 8 |
| 每runner轮轨迹 | 512 | 8或16 | 512 | 512 |
| 动作上限 / 每块动作C | 320 / 8 | 32 / 32 | 32 / 32 | 384 / 32 |
| 每runner轮动作块槽位 | 20,480 | 8或16 | 512 | 6,144 |
| global batch / microbatch | 2048 / 128，修复拟64 | N / 1 | 512 / 8 | 2048 / 8 |
| 更新遍数 / 优化器调度次数 | 1 / 10 | 2 / 2 | 2 / 2 | 2 / 6 |

G8已经包含在N64内。`N×R=512`表示每轮收集512条轨迹，不表示512条同时生成。新布局：actor两个rank、rollout两个rank均在4/5；env两个rank在6/7，每rank管理32环境。6、7卡各一个OpenDW服务，每次只生成一条视频（B1），两服务可以并行；仍需实测WM吞吐，不能将N扩大等同GPU batch扩大。

短测与完整回合测都只跑一个runner轮、save1、不做原生评估，保持H50/C32/M10、noise0.5、GRPO过滤阈值和RM。先短测成功退出、清理对应actor，再在同一借卡scope跑完整回合；中途不恢复RLT。完整回合的6次优化器调度不同于旧LIBERO的10次，不能写成训练预算完全相同。

## 启动与停止合同

服务器根目录S=`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`。

- 独立源码：`S/rlinf-multigpu-v1`，基于`2151a08ee1bd75df1bef0d8190e594bd5c7f7977`；原单卡源码保持。
- 输入计划：`S/multigpu-v1-plan.json`；owner冻结后以`S/runs/multigpu-v1/owner-plan.json`为准。
- 两组原RLT借还：`S/multigpu-gpu4-v1`、`S/multigpu-gpu567-v1`；统一入口`S/multigpu-cycle-v1`，分别保留原repo/完整CP/实配，不混用2f4840与d48e4a源码。
- 输出：`S/runs/multigpu-v1/{n64_short,n64_full}`；WM日志/视频/动作证据在`services/{wm6,wm7}/records`。
- 资源：限定物理4–7；CPU-first加载两个WM，ready后才精确撤RLT。资源监控含全卡C/G、进程GPU/RSS/PSS与服务阶段torch峰值。两服务各保存前640条生成片段，可覆盖短测和完整测第一个R的12块；大原始产物留服务器。
- 上限：短测3600秒，完整测10800秒；服务退出、错误卡位、训练异常或超时立即停止后续阶段，精确释放本owner进程，再按完整CP归还RLT。正常归还最多等待首轮60秒，未完成记pending，不能谎称验收通过，也不阻止后续WM优先接管。

启动命令：

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B \
  /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/multigpu-control-v1/multigpu/opendw_multigpu_owner.py \
  --plan /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/multigpu-v1-plan.json owner
```

启动前已通过服务器CPU检查：路由4项、owner10项、组合借还7项、5–7借还8项，共29项；后者包含原Python缺pidfd接口时的真实CPU子进程syscall兼容验证。首次路由测试因旧PYTHONPATH import失败的回执保留，改为新源码优先后通过。

## 如何判读

exit0与成功保存证明工程接通；资源峰值与每条耗时决定并行是否合适。正式学习还需有效GRPO组、非零有限优势/梯度及完整CP证据；不能把optimizer调用或weight decay造成的参数变化当作奖励学习。旧N8/N16全滤/零梯度不算学习通过。

RM分数是生成环境中的代理判断。正式训练候选保持三相机/C32/384，save10、原生评估10；原生成功率仍需同协议SFT与检查点对照。本轮尚未启动正式训练。
