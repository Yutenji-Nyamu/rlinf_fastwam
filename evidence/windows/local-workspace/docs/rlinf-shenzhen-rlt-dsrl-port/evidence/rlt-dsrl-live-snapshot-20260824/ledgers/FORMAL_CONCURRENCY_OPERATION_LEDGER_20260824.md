# RLT + DSRL 双 formal 并发操作流水账

日期：2026-08-24  
范围：SZ-H100；普通账号 `chenyiteng`；只控制本轮 owned process/Ray namespace。  
完整结论：[08号并发解决文档](../08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md)。

## 1. 固定输入

| 项 | 值 |
|---|---|
| shared Ray | `172.17.0.1:6389`；8 GPUs visible |
| RLT | `codex/sz-rlt-pi0-robotwin-ar@f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4` |
| DSRL | `codex/sz-current-dsrl-pi0-robotwin@4b609178d10d2534f3f972435ad972e4e015c392` |
| RLT placement | physical GPU 4--5 |
| DSRL placement | physical GPU 6--7 |
| 训练环境 | `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin` |

所有远程执行均通过固定 host-key 的 `local_scripts/remote_exec_autodl.py --command-file ... --pty`；密码只在
当前进程环境中提供，没有写入脚本、日志或文档。

## 2. 操作与结果

| 序号 | 精确入口/操作 | 结果、问题与处理 |
|---|---|---|
| 01 | `shenzhen_shared_ray_head_start_20260823.sh` 的早期地址尝试 | 长 AF_UNIX temp path 失败；`127.0.0.1` 被 Ray 2.57 规范化为公网地址；公网自回连超时。均未启动训练 |
| 02 | 同脚本最终使用 `--node-ip-address=172.17.0.1 --port=6389 --block` | persistent head 成功；`ray status` 识别 1 node / 8 GPUs；runtime 位于 `/data/chenyiteng` |
| 03 | RLT 第一次 formal root `...20260823-v1` | driver 连上 Ray，但 NodeProbe `ModuleNotFoundError: rlinf`；未进入模型训练。保留失败根 |
| 04 | 两个 launcher 显式设置各自 `RLINF_CODE_WORKING_DIR` | Ray runtime package 分别来自 RLT/DSRL worktree，worker import 闭环 |
| 05 | `shenzhen_rlt_current_formal_chain_gpu4_5_20260823.sh` | v2 Stage 1 正式运行；2,000/2,000、exit0；`global_step_2000` 与 manifest 落盘 |
| 06 | `shenzhen_dsrl_current_formal_gpu6_7_20260824.sh` 初版 | DSRL v1 在 GPU6--7 完成首轮：40 global transitions，warm-up 40/500；证明可与 RLT 同时运行 |
| 07 | `shenzhen_dsrl_worker_cwd_outputs_20260824.sh` | 确认 external worker cwd=`/home/chenyiteng`，resolved `save_path=./data` 会与另一项训练共享输出 |
| 08 | `shenzhen_stop_dsrl_formal_v1_output_collision_20260824.sh` | 仅 TERM owned PGID `344172`；RLT和Ray head仍 alive；v1目录原样保留并写 `manual_stop.txt` |
| 09 | `shenzhen_cleanup_ray_namespace_dsrl_v1_20260824.sh` | 发现 `RLinf_1` 仍有15个named actor；逐 actor `ray.kill(no_restart=True)`，未动其他namespace |
| 10 | 修正 `shenzhen_dsrl_current_formal_gpu6_7_20260824.sh` 后重启 | DSRL v2 PID `370980`；train/eval data/video改为run-scoped绝对路径；算法参数不变 |
| 11 | RLT chain 自动进入旧 Stage 2 | 完整到Step4，但resolved仍为`save_path=./data`；尚未过10k warm-up、`update_step=0` |
| 12 | `shenzhen_stop_rlt_stage2_v2_output_collision_20260824.sh` | 在保留Stage1 checkpoint后停止RLT owned PGID `332632`，清空exact `RLinf` namespace；DSRL v2仍alive |
| 13 | `shenzhen_rlt_current_stage2_formal_gpu4_5_20260824.sh` | 从同一Stage1 `global_step_2000`启动隔离后的Stage2 v3，PID `390905`；8 env/250 cycles等预算不变 |
| 14 | `shenzhen_rlt_v3_dsrl_v2_live_status_20260824.sh` | 01:04:53 CST：RLT Step5；DSRL Step12后已完成首次真实optimizer段、进入Step13 eval；旧actor0、无异常 |

## 3. 当前正式实验

### RLT

```text
Stage 1:
/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1

Stage 2:
/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3
```

- Stage 1：正式 2,000 steps 已完成、exit0。
- Stage 2：正式 250 cycles 正在运行；01:04:53 CST 完整到Step5。

### DSRL

```text
/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2
```

- 正式200 cycles正在运行。
- 完整Step12为480 global transitions；随后已跨500 warm-up并完成首个真实optimizer段，正在Step13 eval。

## 4. 当前资源与停止合同

- RLT GPU4--5约18.1/19.1 GiB；DSRL update/eval边界GPU6--7约36.0 GiB/card。
- host约83 GiB used、约1.9 TiB available；并发不是当前内存压力源。
- 两个wrapper alive；无exit marker、traceback、OOM或worker crash；stale actor=0。
- 任一训练失败只处理其owned PGID和exact namespace；两项运行期间不执行全局`ray stop`。
- 不按早期成功率自动早停；按各自formal预算自然运行。

## 5. 本轮没有做的事

- 没有修改RLT/DSRL算法源码、loss、batch、UTD、env数或formal预算。
- 没有升级依赖、修改系统网络、删除旧结果或清理`/home/chenyiteng/data`。
- 没有重启GRPO或控制其他用户进程。
