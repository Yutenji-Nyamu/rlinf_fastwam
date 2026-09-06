# 深圳 current RLinf × π0 PPO-DVAC 实施流水账

## 2026-08-30 授权与边界

- 用户授权：实现、push Git、简洁检查，并在空闲卡上 smoke。
- 目标：两卡 π0 PPO Control 与两卡 π0 PPO-DVAC Action-Adv Fix `[0,2]`。
- 约束：不停止或修改 GPU4--7 现有实验；GPU2/3 启动前必须刷新所有权。
- 源码边界：从 `codex/sz-grpo-dvac-action-adv-fix@e434f409...` 建独立 worktree；
  复用已有 Action-Adv Fix，只新增 PPO YAML 和一个 GAE/critic 聚焦测试。
- 配置边界：Control/DVAC 共用同一 source；资源壳为
  `64 train / 32 eval / rollout4 / B1024 / MB32 / update2`；PPO 保留 GAE/value head/group1。

## 操作记录

1. 只读审计收束：现有 `e434f409...` 已支持
   `adv_type=gae + loss_type=actor_critic`；无需新算法生产逻辑。
2. 21:45 CST 只读现场：GPU4/5 和 GPU6/7 两项既有训练均 alive/fatal=0；
   主机 available 约674 GiB；shared Ray 正常。
3. 正在进行：刷新 GPU2/3 与 Git/worktree 现场，实现独立分支。
4. GPU2/3 启动前只读现场：两卡均9 MiB、util=0、无 compute process；可用。
5. 服务器 worktree：
   `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix`；
   branch `codex/sz-ppo-dvac-action-adv-fix`。
6. Git 实现链：`e434f409...` + 已验证 local-shard 透传 + PPO YAML/聚焦测试；
   最终 local=remote HEAD=`74617ced87d64045ab6850d0efd90956a494af66`，tree clean，
   普通 non-force push 成功。
7. 简洁检查：`diff --check`、`py_compile` 通过；聚焦 pytest `10/10`通过；
   Hydra Control/DVAC resolve 分别确认 PPO `GAE+actor_critic+value head`，且只有方法叶子不同。
8. Control smoke1 已提交到 GPU2/3：
   `ppo-control-smoke1-2gpu64x4-b1024-noeval-localshard-phys23-v1`，wrapper PID=`3731946`，
   resolved 断言通过；1 个完整 outer step、4/4 rollout、optimizer update 与
   `global_step_1` 保存均完成，exit code=0；两份 rank-local shard 和 full weights 均已落盘，
   GPU2/3 自然释放。启动与结束时 GPU4--7 既有训练均未被操作。
9. DVAC smoke2 已按相同两卡资源壳提交到 GPU2/3：
   `ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1`；
   resolved 断言确认 `action_level + action_advantage + L3 + recent5 + [0,2]`，
   2 个 outer step 自然完成并 exit code=0。Step1 warm-up 权重为全1；Step2真实权重
   `mean=0.986 / sq_mean=1.177 / ESS=0.826`，非均匀且全部有限；actor policy loss
   `-2.909`、grad norm `43.536`，critic value loss `0.075`、explained variance `-0.071`，
   均为有限值。
10. `global_step_2` 保存完整：两份 rank-local shard、`full_weights.pt` 和两份
    DVAC state sidecar 均存在；总量约25 GiB。wrapper自然退出，GPU2/3回到9 MiB；
    无 fatal、OOM 或 worker crash，GPU4--7 两项既有正式训练仍存活且未受影响。
11. SSH 凭据始终仅进入当前进程，未写入脚本、文档或仓库。
