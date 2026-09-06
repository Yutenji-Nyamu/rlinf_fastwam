# 深圳多任务 LeRobot pi0.5 到 current RLinf

更新时间：2026-09-04 17:10 CST（服务器只读快照；后续状态仍须刷新）

最新只读刷新入口：[2026-09-05 12:37状态](evidence/PI05_STATUS_REFRESH_20260905_1237.md)：完整99/100，step100采样中，预计约13:00收尾，最终评测与checkpoint尚待产出。下文09-04状态保留为历史。

09-05用户已授权100自然结束后原地续到200并交付100步轻量ZIP；唯一执行入口：[RESUME100_TO200_EXECUTION_LEDGER_20260905.md](evidence/RESUME100_TO200_EXECUTION_LEDGER_20260905.md)。原实验目录/GPU4/5与其他参数保持，完整resolved已验证仅max_steps/resume_dir两叶变化；是否已启动以该账本和服务器现场为准。

## 0. 当前目标

第一候选锁定 `SidneyXie/pi05_robotwin`。checkpoint 自带的 LeRobot policy、preprocessor、
postprocessor、H50、M10、三相机与 14D absolute-qpos 已作为 oracle 验证；current RLinf 严格转换、
B=1 环境闭环和两卡 GRPO Step1 smoke 也已完成。用户先锁定 `move_stapler_pad`，随后明确切换到
`move_pillbottle_pad`；保持200-action、noise0.5，授权跳过独立 SFT Control。现役两卡 formal100
已于2026-09-03 20:23:41在GPU4/5从SFT fresh启动；不是此前stapler运行的续训。

## 1. 官方推理协议

- source：Hugging Face LeRobot official repository，锁定实际 commit 后记录；
- checkpoint：`SidneyXie/pi05_robotwin`，锁定实际 revision 后记录；
- RoboTwin：复用深圳已验证的 source/assets/CuRobo 兼容层，不改任务或控制语义；
- task：`adjust_bottle` 与 `move_stapler_pad`；
- 每任务 5 episodes，`eval.batch_size=1`；
- camera rename：`head/left/right_camera` 到 checkpoint 的 `cam_high/cam_left_wrist/cam_right_wrist`；
- 独立环境、模型目录、输出目录；只使用服务器自身网络。

这里的 5 条只用于接通与粗粒度成功率，不替代模型卡的每任务 100 条正式结果。

## 2. 接入 current RLinf 的推荐边界

首选路线是把 LeRobot pi0.5 权重和 processor 合同转换到 current OpenPI backend，而不是把整个
LeRobot 运行时嵌入 RLinf：

1. 权重转换必须显式核对 matched/missing/unexpected keys；
2. checkpoint 自带 mean/std 只 normalize/unnormalize 一次；
3. 三相机、14D absolute qpos、内部 32D padding、H50/M10 保持原样；
4. rollout 继续走 `PolicyOutput -> ChunkStepResult -> Builder -> Trajectory`；
5. Flow-SDE old/new log-prob 重放沿用 current pi0.5 actor，不另建训练旁路。

正式实现前的关键 parity 是：同一 observation、prompt、noise 和 M10 下，比较 LeRobot oracle 与
转换后 current OpenPI 的 normalized model action 和最终 14D qpos chunk。

这条路线的主要依据是 LeRobot 官方已经维护 pi0.5 与原始 OpenPI 的对齐测试：同一 batch、noise 和
time 下检查 loss、M10 action，以及零 missing/unexpected key。Sidney checkpoint 仍需单独复测，
但不需要先假设一个全新的模型架构。相关原始材料：

- [Sidney checkpoint 与原生执行合同](https://huggingface.co/SidneyXie/pi05_robotwin)
- [checkpoint 完整文件树](https://huggingface.co/SidneyXie/pi05_robotwin/tree/main)
- [LeRobot pi0.5/OpenPI parity test](https://github.com/huggingface/lerobot/blob/main/tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py)
- [LeRobot pi0.5 processor](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/processor_pi05.py)

未发现现成的 Sidney/LeRobot policy 到 RLinf 的 adapter。RLinf 当前的 LeRobot 支持主要位于数据侧，
不能据此把 LeRobot policy 当作已经接入。最危险的错误是 current OpenPI loader 的 `strict=False`
让不完整权重加载静默通过，或者继续使用旧 OpenPI norm stats；所以权重 key 与 processor/norm parity
是接入前置条件，而不是额外的防御性测试。

## 3. 轻量实验产物的版本边界

源码分支按方法实现维护，不按每个 seed、强度、resume 或 GPU 布局重复建分支。每条科学 run 另存
不可变 manifest：source commit、resolved config、命令、指标、TensorBoard、资源 CSV、关键日志、
图与 summary。checkpoint、视频、Ray 日志、数据不进 Git。

当前深圳主要方法实现均已推送。按用户选择，轻量实验 evidence 直接回填对应算法分支的
`evidence/<run-id>/`：保留 resolved、命令/source lock、指标、资源、关键日志、图与 manifest，排除
checkpoint、视频、模型、数据和完整 Ray 日志；Sidney smoke 已按此规则推送，历史 run 尚待逐分支回填。

## 4. 接入与运行状态

official-native oracle 已完成并按授权停止：

- `adjust_bottle`：5 个有效 seed，`5/5`；
- `move_stapler_pad`：5 个有效 seed，`2/5`；
- 均为 checkpoint 自带 processor/norm、H50/M10、absolute 14D、三相机、B=1；
- current RLinf adapter 已在独立分支实现、严格转换并推送；B=1 环境闭环与真实两卡 GRPO Step1 smoke 均已通过。
- official LeRobot v0.6的13文件Python3.10 typing兼容diff仍是服务器detached dirty runtime，未commit/push；
  主要RLinf方法分支已推送，但本轮轻量oracle证据仍仅在服务器和本地账本中。

逐操作、source lock、精确命令、逐 seed 结果和停止边界见
[`evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md`](evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md)。

current RLinf 接入的三层结构、逐文件改动、严格合同与最小验收见
[`01_SIDNEY_TO_CURRENT_RLINF_LAYERED_ADAPTER_PLAN_20260903.md`](01_SIDNEY_TO_CURRENT_RLINF_LAYERED_ADAPTER_PLAN_20260903.md)。
下一批 RL 任务的短表、模型卡评测 horizon 歧义与 `UnStableError` reset 调用链见
[`evidence/TASK_SELECTION_AND_UNSTABLE_RESET_AUDIT_20260903.md`](evidence/TASK_SELECTION_AND_UNSTABLE_RESET_AUDIT_20260903.md)。
实现账本见
[`evidence/CURRENT_RLINF_ADAPTER_IMPLEMENTATION_LEDGER_20260903.md`](evidence/CURRENT_RLINF_ADAPTER_IMPLEMENTATION_LEDGER_20260903.md)。
当前分支为 `codex/sz-sidney-pi05-current-rlinf@f50e235c`；实现代码锁在父提交 `bab221af`，随后两次
普通提交把约 80 KiB 的筛选 smoke evidence 放入同一分支；真实转换已经做到
813/813 keys、零 missing/unexpected/shape mismatch，并逐 tensor 完全相等。224x224 core parity 的
双 backend forward 已完成：token/mask 对齐，H50x32 与最终 H50x14 action 均通过官方 action 容差。
B=1 current RLinf 环境闭环已验证请求坏 seed `1001/1002` 后自动重试到 `1003` 并成功；两卡 GRPO
Step1 也完整经过 64-trajectory rollout、两次 optimizer update、fixed8 与 local-shard 保存后自然
`exit 0`。GPU4/5 峰值约 58.0/58.3 GiB。随后按用户明确口径在 GPU4/5 启动
`move_stapler_pad` formal100：200-action、noise0.5、64 env x rollout4、G8、GB1024/update2、
fixed32/eval5/save10。启动证据见
[`evidence/SIDNEY_MOVE_GRPO_FORMAL100_H200_NOISE05_LAUNCH_20260903.md`](evidence/SIDNEY_MOVE_GRPO_FORMAL100_H200_NOISE05_LAUNCH_20260903.md)。

以上stapler启动属于历史。其后按用户授权切换pill任务，其他训练/资源/模型叶不变，见
[`evidence/SIDNEY_MOVE_PILLBOTTLE_GRPO_CUTOVER_20260903.md`](evidence/SIDNEY_MOVE_PILLBOTTLE_GRPO_CUTOVER_20260903.md)。

### 4.1 2026-09-04 学习速度审计与最新快照

17:10:52只读刷新：pill完整Step53，训练145/256=56.64%，最近10步均值58.48%；最新fixed为Step50
16/32，Step10/35最好19/32，尚未持续改善。所查fatal/OOM/OIDN/Traceback/RuntimeError为0；
Step50双rank shard与full_weights本轮确认存在，未做恢复测试。Sidney工作树f50e235c仍clean。

与深圳旧两卡π0 adjust_bottle Control同为前50步、12,800条轨迹、100次optimizer call：
Sidney首10→末10步训练均值41.68→57.23%，旧π0为78.36→89.61%；训练百分点增幅并不更慢。
但Sidney fixed仍波动，旧π0从Step20附近稳定在约30/32。到Step50耗时19.52h vs22.37h，也非墙钟更慢。
不同任务/SFT、内部delta vs absolute、state输入路径、M4 vs10与norm使这不是纯架构比较。

当前唯一学习对比证据入口：
[`evidence/PI05_VS_PI0_LEARNING_REVIEW_20260904.md`](evidence/PI05_VS_PI0_LEARNING_REVIEW_20260904.md)。
含实际resolved52项差异、actual embodied actor审计、完整曲线、源码与剩余adapter边界；未发现明显漏更新。
下一步建议优先同协议SFT/current及失败阶段核查，判断200-action内是否来不及完成放置/松爪；
尚未证明超时或过拟合，未改LR/noise/horizon或训练实现，额外模型评估需另获授权。

## 5. formal 前的参数口径

RLinf upstream 没有 RoboTwin pi0.5 GRPO recipe；深圳既有 pi0.5 GRPO 是从官方 pi0.5 模型与
current GRPO 接出的本地对照。当前 Sidney formal 保持其两卡训练壳：`64 train / 32 eval`、
`rollout4=256 trajectories`、G8、GB1024/MB32/update2、fixed32/eval5/save10、local-shard；模型合同为
Sidney checkpoint/norm、M10、absolute 14D，用户明确将 rollout noise 从 native 0.3 改为0.5。

任务 horizon 已按用户选择统一锁成200：train/eval 的 `max_episode_steps`、
`max_steps_per_rollout_epoch` 与 `task_config.step_lim` 均为200。H=C50 下最多4条 query record/trajectory，
所以256 trajectories上限为1024 records；GB1024/update2对应每步两次optimizer call、每条record呈现两次。
这与深圳成功pi0两卡GRPO的数据行为对齐，但相对RoboTwin为move任务列出的400-action合同会更早截断。

图像方面，native LeRobot 使用float Torch resize，current OpenPI使用uint8 JAX resize并round；两者从
480x640到224x224不逐像素相同。当前没有修改公共processor，而是用224x224输入隔离验证权重、token、
norm与M10模型核心，并把真实raw-frame效果留给RLinf-400 fixed-seed SFT Control。这保持current RLinf
主路径干净，但在完成Control前不能把Sidney模型卡成功率直接当成本地基线。
