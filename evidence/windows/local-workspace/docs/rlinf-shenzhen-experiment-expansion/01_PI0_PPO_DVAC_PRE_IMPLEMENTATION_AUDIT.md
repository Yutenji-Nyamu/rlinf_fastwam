# 深圳 current RLinf × π0 两卡 PPO-DVAC 实现前审计

> 状态：2026-08-30 只读收束；未改服务器、未启动 smoke。

## 1. 结论

- 首线做两卡 `π0 PPO Control` 和两卡 `π0 PPO-DVAC Action-Adv Fix [0,2]`。
- 从 `codex/sz-grpo-dvac-action-adv-fix@e434f409...` 建独立
  `codex/sz-ppo-dvac-action-adv-fix` worktree。
- 现有 Action-Adv Fix 已天然支持 `adv_type=gae + loss_type=actor_critic`；
  无需新增一套算法生产逻辑。
- 实施只需：新 PPO YAML、一个 GAE/critic 形状聚焦测试、两份 resolved packet。
- 没有待用户决定的算法语义；只有 eval cadence 和 checkpoint format 两个运行口径。

## 2. 两卡 PPO Control 精确继承

| 字段 | 深圳四卡 PPO | 两卡 PPO Control | 变化依据 |
|---|---:|---:|---|
| actor/env/rollout ranks | 4 | 2 | 物理卡减半 |
| train/eval env | 128/64 | 64/32 | 保持每 rank 32/16 env |
| rollout epochs | 4 | 4 | 不改每轮行为 |
| trajectories/step | 512 | 256 | `env × rollout` 随卡数减半 |
| max query records | 2048 | 1024 | `64 × 4 × 200 / 50` |
| global/micro batch | 2048/32 | 1024/32 | 保持每 rank 唯一 record 和 micro 工作量 |
| update epochs | 2 | 2 | 不改 PPO 更新强度 |

这些与两卡 GRPO 共用的是**资源壳**。PPO 必须从四卡 PPO 继承：

- `group_size=1`、`adv_type=gae`、`loss_type=actor_critic`、`filter_rewards=false`；
- chunk reward；Control 使用 chunk-level log-prob；
- value head、GAE `gamma=.99/lambda=.95`、value clip/huber、actor/value optimizer；
- π0 checkpoint/config、M4、H=C=50、D=14、三相机和 RoboTwin task/seed。

不能把 GRPO 的 G8、group filter、group-relative advantage 或 actor-only 搬进 PPO。

## 3. PPO-DVAC 相对 Control 的唯一科学增量

PPO 先按原路径计算 GAE：

$$
A^{\mathrm{GAE}}_i,\;R_i\in\mathbb{R}^{B\times1}.
$$

仅 actor 使用：

$$
A^{\mathrm{eff}}_{i,h}
=A^{\mathrm{GAE}}_i\,\operatorname{stopgrad}(w^{\mathrm{DVAC}}_{i,h}),
\qquad w\in[0,2].
$$

- `logprob_type: chunk_level -> action_level`；
- `mode=apply`、`application=action_advantage`、`selected_l=3`；
- recent-5 global-z，`weight_min/max=0/2`；显式端点存在时 `strength` 不参与映射；
- actor loss 先求和有效 $H$，再在 query/batch 上平均，避免错误多除一个 $H$;
- return、bootstrap、value target、value loss、GAE、critic optimizer 全部不变。

Control 和 DVAC 应运行在同一 superset source/YAML 上：Control 是
`mode=off + chunk_level`，DVAC 是 `mode=apply + action_level`。

## 4. 逐文件实施边界

### 直接复用，不再改

- `rlinf/models/embodiment/openpi/openpi_action_model.py`：同次 rollout 已返回
  `prev_values[B,1]` 和 `z_endpoint[B,M,H,D]`。
- `rlinf/workers/rollout/hf/huggingface_worker.py`：已把 `dvac_v_l3[B,H]`
  接入 typed `forward_inputs`。
- `rlinf/algorithms/dvac_train_weighting.py`：recent-5、global-z、`[0,2]`、
  rank reduce、detach 和 sidecar 均与 GAE/GRPO 类型无关。
- `rlinf/workers/actor/embodied_fsdp_actor_worker.py`：DVAC 权重已随
  `forward_inputs` 一起 shuffle；GAE 路径仍计算 value。
- `rlinf/algorithms/utils.py`：已支持 `A[B,1] × w[B,H] -> A_eff[B,H]`，
  value/return 保持 `[B,1]`。
- `rlinf/algorithms/losses.py`：actor 使用修正后的 H-sum；critic 仍走标准
  `compute_ppo_critic_loss`。

### 本次新增

1. `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_dvac_action_adv.yaml`；
2. 在既有 `test_dvac_train_weighting.py` 增加一个聚焦测试：
   - GAE advantage 从 `[B,1]` 变为 actor `[B,H]`；
   - critic value/return 仍是 `[B,1]`；
   - 开关 DVAC 只改 actor loss，value loss 与 Control 一致。

明确不改 `advantages.py`、RoboTwin/schema/builder、reward/bootstrap、OpenPI 模型数学、
optimizer 和采样预算。

## 5. 运行口径建议

- fixed eval：建议 `fixed32 / eval every 5 / save every 10`。每卡16 eval env 已验证稳定；
  每10步累计64 eval episodes，与旧四卡 PPO 的 fixed64/eval10 评估量相同。
- checkpoint：建议 Control/DVAC 共同用已验证 `local_shard`，避免并发 Ray job
  的 DCP 协调卡住；这不是方法变量。
- env offload：首版保持四卡 PPO/两卡 GRPO 的 `false`，不静默改内存行为。
- formal 是否两项并发：等 smoke 实测单项显存/主存后再定；不属于算法模糊点。

## 6. 下一轮实施与 smoke 边界

1. 建 branch/worktree，新增 YAML 和唯一聚焦测试，push Git。
2. 生成 Control 和 DVAC 的 resolved diff，意外差异必须为0。
3. Control smoke 需1个真实 outer step；DVAC smoke 需2步，因为step1只建
   recent history，step2才能观察非均匀 `[0,2]` 权重。
4. 可在空闲 GPU2/3 上轮流 smoke；不干扰 GPU4--7 现有实验。
5. 真正启动前按规则向用户展示 resolved config、命令、路径、预算、
   资源和停止条件，等待授权。

## 7. 2026-08-30 21:45 CST 现场（只读）

- GPU4/5 Action-Adv Fix `[0.5,1.5]`：完整 Step26，正在Step27，fatal=0；
  最近 fixed32 为27/32。
- GPU6/7 ST-DVAC `[0.8,1.2]`：完整 Step11，正在Step12，fatal=0；
  Step10 fixed32=29/32，Step10 local-shard 已完整落盘。
- 显存约55/55/68/70 GiB；主机 available 约674 GiB；shared Ray 正常。

