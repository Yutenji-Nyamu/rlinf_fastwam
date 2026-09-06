# Sidney 多任务 pi0.5：RL 任务初选与 `UnStableError` 核对

更新时间：2026-09-03 CST

## 1. 结论

- 若现在就接 current RLinf 做第一批 RL，首选 `place_a2b_left` 与 `move_can_pot`：Sidney 模型卡分别为
  `49%/48%`，RoboTwin 官方上限均为 400 actions，RLinf reward 与预筛 train/eval seed bank 都已存在。
- `place_fan=50%` 是最贴近目标成功率的科学候选，但 current RLinf 的 seed bank 为 `null`；reward 已支持，
  需要先生成/固定一份稳定 seed bank，才适合做可重复的正式比较。
- **不能把“Sidney 49%”和“官方 400 actions”理解为同一评测协议。** 模型卡公布每任务 100 回合，
  但没有写 episode horizon；其示例命令也没有传 `episode_length`。LeRobot `v0.6.0` 当前默认是 1200，
  而 RoboTwin 主仓 `_eval_step_limit.yml` 对这些任务写的是 400。模型卡当时究竟用了 400、1200，还是
  其他版本/参数，现有公开材料无法确认。因此下表把二者分别作为“成功率先验”和“任务长度先验”；
  formal 前必须在目标 RLinf 400-action 协议下重测 SFT fixed seeds。

## 2. 推荐表

Sidney checkpoint 固定合同为三相机、14D absolute qpos、`H=50`、每次预测执行 50 actions、`M=10`。
所以 400-action 上限最多需要 8 次 policy query。下面的 `4--5 min` 是由深圳 cache-hit B=1
`move_stapler_pad` 失败回合 `1200 actions / 739.95--853.43 s` 线性缩放出的**粗略最坏墙钟**，不是这些
任务的实测耗时；向量化 RLinf 的墙钟应另做 smoke。

| 优先级 | task | Sidney 模型卡 SFT | RoboTwin 官方上限 | 最多 query（C50） | RLinf reward | current train/eval seed bank | 建议 |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | `place_a2b_left` | 49% / 100 | 400 | 8；B=1 粗估最坏 4--5 min | 支持 | 1000 / 150 | 最干净第一任务；离 50% 仅 1 pp，运行条件齐全 |
| 2 | `move_can_pot` | 48% / 100 | 400 | 8；B=1 粗估最坏 4--5 min | 支持 | 1000 / 150 | 同样 ready；可与第一任务形成不同物体/运动语义 |
| 3 | `place_mouse_pad` | 45% / 100 | 400 | 8；B=1 粗估最坏 4--5 min | 支持 | 1000 / 320 | seed 覆盖最宽；起点稍低，仍处于适合 RL 的中段 |
| 4 | `lift_pot` | 57% / 100 | 400 | 8；B=1 粗估最坏 4--5 min | 支持 | 1000 / 200 | 起点稍高；适合作为 lift 类任务补充 |
| 5 | `place_a2b_right` | 41% / 100 | 400 | 8；B=1 粗估最坏 4--5 min | 支持 | 1000 / 150 | 与 left 成对，适合测左右泛化，但离 50% 更远 |

第二梯队也都是 400-action 且 reward 已支持：`place_fan=50%`、`turn_switch=55%`、
`stamp_seal=45%`、`pick_diverse_bottles=59%`。它们目前 train/eval `success_seeds` 均为 `null`，不是
模型或环境不兼容，而是 formal 前少一份固定的稳定 seed bank。`place_object_scale=46%` 虽然数值理想，
但 RoboTwin `RLinf_support` 明确把它列为“reward 尚未支持”，不应进入首批。

较低成功率但更长的探索档为 `scan_object=18%/500 actions` 与
`handover_mic=23%/600 actions`；它们不符合当前“短、约 50%”的第一优先级。

## 3. qpos、相机与 current RLinf 兼容性

- Sidney 输入是高位、左腕、右腕三相机，状态/动作是 Aloha 14D absolute joint qpos；这与深圳
  current RLinf OpenPI/RoboTwin 的外部控制语义相近，不是 EEF/IK 路线。
- checkpoint 仍要求自己的 camera key、processor 与 norm；接入时要把三相机映射到
  `cam_high/cam_left_wrist/cam_right_wrist`，并保证 norm 只执行一次。
- 任务层还要分别满足 RLinf reward 与稳定 seed。上表前五项已经同时具备 reward 和非空 seed bank；
  因此它们比只看模型卡百分比更接近“可以直接形成 formal packet”的候选。

## 4. `adjust_bottle` 的 `UnStableError` 到底是什么

这次失败发生在策略执行前，不是 pi0.5 输出坏动作：

1. RoboTwin `setup_demo()` 创建桌面、机器人、相机和物体后调用 `check_stable()`；若物体初态不稳定，
   `_base_task.py` 直接抛 `UnStableError`。深圳 seed1001、1002 日志中的具体对象是 `001_bottle`；
   reset-only 又确认 1005--1007 同类失败。
2. LeRobot `v0.6.0` 的 `RoboTwinEnv.reset(seed)` 把 exact `actual_seed` 直接传给
   `self._env.setup_demo(...)`，没有捕获或重试，所以一个无效初始化 seed 会直接中止该 native eval。
3. 深圳 current RLinf 通常先从 `success_seeds.json` 读取预筛 seed；`adjust_bottle` 当前有
   train 1000 条、eval 150 条。其 RoboTwin `RLinf_support` vector adapter 还有第二层处理：捕获
   `UnStableError`，关闭该 env，把 `trial_seed` 加一后重新初始化。

因此 current RLinf 单任务 pi0.5 没显露这个错误，核心差异是**seed 选择与 reset 适配器**，不是模型更稳定：

```text
LeRobot native: requested seed -> setup_demo -> unstable -> exception exits
current RLinf:  pre-screened seed -> setup_demo
                         or unstable -> close -> next seed -> retry
```

一个复现细节：RLinf 若走 `trial_seed += 1`，实际运行 seed 可能不再等于最初请求 seed。严格 fixed-eval
应保存 adapter 最终接受的 actual seed；已有非空 seed bank 的任务则优先直接使用固定预筛列表。

## 5. 证据与口径

- [Sidney 模型卡：模型合同、100-episode 分任务结果](https://huggingface.co/SidneyXie/pi05_robotwin)
- [RoboTwin 主仓官方 action 上限](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/env_cfg/task_config/_eval_step_limit.yml)
- [LeRobot v0.6 RoboTwin wrapper：默认 1200 与 direct `setup_demo`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/envs/robotwin.py)
- [RoboTwin `RLinf_support` reward 支持列表](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support)
- current RLinf seed 选择入口：
  `references/rlinf_fastwam_audit_20260824/worktrees/sz-current-dvac-grpo-w0to5/rlinf/envs/robotwin/robotwin_env.py:417`
- current seed banks：同目录 `seeds/train_seeds.json` 与 `seeds/eval_seeds.json`。
- 深圳逐 seed 运行证据：
  [`IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md`](IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md)。

本轮只读源码锁：native `_base_task.py` SHA256
`a4fa65537d2ba34206b89980b031e0bdc543561e01b0b1a0ae735617d577d2e7`；LeRobot
`robotwin.py` SHA256 `e63e579d1b238090640593b1fbb7cac4668fe27dce2ac28d87f1da75a36bdd84`；
RLinf-support `vector_env.py` SHA256
`7afbdfc744d43a5bae49d7b26d8aba857e461930abd1776698e6ab42358d624c`。

