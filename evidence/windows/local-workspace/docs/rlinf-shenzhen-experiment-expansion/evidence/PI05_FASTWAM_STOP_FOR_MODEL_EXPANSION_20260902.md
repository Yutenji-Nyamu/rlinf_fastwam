# π0.5 / Fast-WAM 当前 formal 精确停止记录（2026-09-02）

## 授权与边界

- 用户明确授权中止当前 π0.5 GRPO 与 Fast-WAM GRPO，并将空卡用于后续 Sidney π0.5 official 推理及
  Fast-WAM offload smoke。
- 只处理两条已核验的 chenyiteng-owned wrapper、process tree、Ray job 和 exact namespace；不停止
  shared Ray，不触碰 liwenbo 的 GPU0 StarVLA 服务或其他用户进程，不删除任何 run/checkpoint。

## 停止前事实

现场时间为 2026-09-02 21:31 CST。

| Run | GPU | wrapper / PGID | Ray job / namespace | 最后完整 step | checkpoint | fatal |
|---|---:|---:|---|---:|---|---:|
| π0.5 Control `pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2` | 4,5 | `1477572` | `3f010000 / RLinf`，15 actors | 58 | Step 50 | 0 |
| Fast-WAM256 `fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2` | 6,7 | `826980` | `6e010000 / RLinf_1`，15 actors | 3 | 无 | 0 |

π0.5 Step58 的最新 actor KL / clip / grad 为 `-0.0090 / 0.044 / 13.721`；Fast-WAM Step3 为
`0.000835 / 0.012 / 6.112`，均为有限值。本次停止不是错误处置，而是用户主动切换实验。

## 精确停止

21:34 CST 依次执行：

1. 再次验证 wrapper PID 等于 `owned.pgid`、属主为 `chenyiteng`、命令行包含精确 run path；
2. 验证各目标 GPU 上恰有 6 个 compute PID，且全部属于预期 Ray job；
3. 验证目标 namespace 停止前恰有 15 个 named actors；
4. TERM 该 wrapper 的精确进程树和独立 observer；
5. wrapper 退出后两个 namespace 均已由 RLinf 自动清空，因此清理器观测到 0 actors，没有向其他
   namespace 发出 `ray.kill`；
6. 等待预先核验的 target job GPU PID 全部消失，并写入各 run-scoped stop marker。

停止标记：

- `<pi05-run>/runtime/stopped_by_user_for_model_expansion_20260902.txt`
- `<fastwam-run>/runtime/stopped_by_user_for_model_expansion_20260902.txt`

## 终态与服务器快照

21:35:49 CST，两条 wrapper、两个 namespace 及 π0.5 最后两个 native EnvWorker 均已退出；GPU4--7
均为 `0 MiB` 且无 compute process。shared Ray GCS PID `321933` 保持原启动时间，状态 active、无 pending
demand；没有重启。

21:37 CST 服务器摘要：

- RAM：约 `1.9 TiB available`，memory PSI 为 0；
- 磁盘：`/` 余 225 GiB，`/home` 余 1.4 TiB，`/data` 余 1.6 TiB；
- 网络：Mihomo active、restart 0；经 `127.0.0.1:7890` 访问 GitHub/Hugging Face 均 HTTP 200，约
  `0.57 / 0.42 s`；
- 其他用户：liwenbo 在 GPU0 保留一条 StarVLA `Qwen3-VL-OFT-LIBERO-4in1` policy server，约
  `9.7 GiB`；zhangwei 与 liwenbo 另有 VSCode/Codex 轻量进程。均未干预。

本记录只证明精确停止及资源释放；没有启动 Sidney π0.5 或 Fast-WAM offload 实验。
