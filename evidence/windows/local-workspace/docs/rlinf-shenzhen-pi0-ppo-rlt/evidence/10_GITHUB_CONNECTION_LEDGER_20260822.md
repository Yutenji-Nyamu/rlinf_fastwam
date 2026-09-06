# SZ-H100 GitHub 连接与 GRPO worktree 流水账

日期：2026-08-22

边界：用户已明确要求由 Codex 尽量完成 GitHub 连接准备。本账只记录 `chenyiteng` 的 repo-scoped
deploy key、后续个人 remote 和 GRPO worktree；不记录密码、private key、token。当前不执行 push，
也不创建 GRPO worktree，直到公钥在 GitHub 登记并完成读写探针。

## GIT-KEY-001 — 精确 preflight

状态：`PASS`，2026-08-22 16:00:37 CST，exit `0`。

- 本地 command file：
  `local_scripts/remote_commands/shenzhen_github_deploy_key_preflight_20260822.sh`
- 目标 canonical：`/data/chenyiteng/projects/rlinf-shenzhen/RLinf`
- 目标 key：`/home/chenyiteng/.ssh/github_rlinf_fastwam_deploy_ed25519`
- 检查：身份、exact HEAD/status/remotes/worktrees、`ssh-keygen`、目标 key 不存在；只读。
- command SHA256：`8e3019138a9e41c9adc618a284e8069c87bbf3b1a4ed5f7dd15c959d200393b3`。
- 结果：UID 1003 `chenyiteng`；canonical exact `7d07a421...`、clean，仍只有 official
  HTTPS `origin`；worktree 仍只有 canonical 与 clean PPO；`ssh-keygen=/usr/bin/ssh-keygen`；
  `~/.ssh` 与精确目标 key 均不存在。可以无覆盖风险进入生成步骤。

## GIT-KEY-002 — 生成 repo-scoped Ed25519 keypair

状态：`PASS`，2026-08-22 16:01:29 CST，exit `0`。

- 本地 command file：
  `local_scripts/remote_commands/shenzhen_github_deploy_key_generate_20260822.sh`
- 只创建上述专用 keypair 与必要的 `~/.ssh` 目录；不覆盖已有 key。
- private key 保持服务器本地 mode `600`，不进入聊天或账本；只向用户交付 `.pub` 公钥与 fingerprint。
- 本步骤不写 `known_hosts`、SSH config、Git remote，也不尝试尚未登记的认证。
- command SHA256：`f81861d2c6aed5b697e07edcf0e4a0cf865fb6a183d0fec74cf55c6252977a5d`。
- 结果：private/public mode 分别为 `600/644`；Ed25519 fingerprint 为
  `SHA256:rHlM9Y7x89Y+1n4F98ZYB70oHiTiQuPbw2YU8oBGs24`。
- 需登记的公钥（不是 private key）：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFiXGArwNtYn67ZT/5HMQlXAkSNaiaJ0OyzGRioL6p57 rlinf_fastwam deploy key on SZ-H100 2026-08-22
```

## 用户网页步骤与后续动作

生成成功后，用户只需在 `Yutenji-Nyamu/rlinf_fastwam` 的
`Settings -> Deploy keys -> Add deploy key` 粘贴公钥并启用写权限。用户确认后，再单独执行：

1. 依据 GitHub 官方 host-key fingerprint 写入专用 known-hosts；
2. 使用专用 `core.sshCommand`/host alias 做只读 `git ls-remote`；
3. 添加 `personal` remote；
4. 从 exact `7d07a421...` 创建 GRPO branch/worktree；
5. 实现并验证后再 push，绝不 force-push。
