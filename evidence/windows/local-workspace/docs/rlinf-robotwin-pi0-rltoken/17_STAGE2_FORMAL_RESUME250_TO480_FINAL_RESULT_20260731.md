# Stage 2 formal 续训 250→480 最终结果（2026-07-31）

## 1. 结论

RoboTwin `adjust_bottle` RLT Stage 2 已从完整 `global_step_250` 连续训练到绝对终点
`480/480`，自然结束且 `exit_code=0`。续训没有改变模型、算法、环境、batch、UTD、
BC/Q schedule 或评估/保存协议；只改变了 resume source、绝对总终点和隔离输出目录。

续训后的主要结论是：

- train rollout 从原 P4 的 `375/472=79.45%` 提升到续训区间
  `1690/1840=91.85%`，最后 50 cycles 为 `370/400=92.50%`；
- 续训的 10 个 fixed-20 deterministic eval 合计 `178/200=89%`，最终点
  `17/20=85%`；相对 cycle 250 的 `18/20=90%` 只差同一 20-seed bank 中 1 条，
  属于高位平台波动，不能判定为退化，也没有证据表明继续训练仍持续提高 eval；
- critic/actor、replay、UTD5 和 resume schedule 全部连续，最终
  critic/actor updates 为 `215055/107528`，pending update budget 为 0；
- loss、Q 和 gradient 全部 finite，CUDA OOM、NCCL fatal、Ray actor death、
  NaN/Inf、cgroup OOM/OOM-kill 均为 0；
- GPU 未持续顶满；本轮约束仍在 host/cgroup memory，不能仅凭显存余量继续扩大 env。

完整审计现场为 `2026-07-31T09:24:59+08:00`；文档收尾后的最小 live 复核时间为
`2026-07-31T09:44:57+08:00`，完成/进程/GPU/checkpoint状态未变化。

## 2. 运行与连续性

| 项目 | 结果 |
|---|---|
| 启动 | `2026-07-30T23:51:08+08:00` |
| 完成 | `2026-07-31T09:17:41+08:00` |
| wall-clock | `9h26m33s` |
| source checkpoint | `global_step_250` |
| final checkpoint | `global_step_480` |
| runner | `480/480` |
| exit | `0` |
| driver / monitor | 均已退出 |
| live GPU | 两卡 0 MiB、0% |
| resolved SHA-256 | `cbbfffda43a6ca17ee938da21d7f71ccb70ba394d1247b8e5ae8d3f48dda5787` |
| Git | `codex/rlt-pi0-robotwin@46a2d19bae629eaa57830f5faeac71ac81a1a494`，worktree clean，ahead upstream 2 |

新旧 resolved 的机器 diff 仍只包含 6 个运行控制项：resume source、绝对终点、新
logger root/name，以及由新 root 派生的 train/eval video 目录。算法、环境、UTD、
batch、BC/Q schedule 与原 formal250 完全相同。

TensorBoard 两段分别覆盖 step `0–249` 和 `250–479`；按
`outer cycle = TensorBoard step + 1`，无重叠、无缺口地覆盖 cycle `1–480`。两个文件
都有同一组 89 个 scalar tags，共 35,180 条 scalar records，NaN/Inf 为 0。

## 3. 成功率

### 3.1 四阶段 train

| 阶段 | Cycles | 成功 / episodes | 成功率 |
|---|---:|---:|---:|
| P1 reference collect | 1–135 | 156/1080 | 14.44% |
| P2 reference + SAC | 136–154 | 22/152 | 14.47% |
| P3 student + ramp | 155–191 | 94/296 | 31.76% |
| P4 stable student | 192–480 | 2065/2312 | 89.32% |
| 其中原 P4 | 192–250 | 375/472 | 79.45% |
| 其中续训 | 251–480 | 1690/1840 | 91.85% |

续训内部：

- cycle 251–300：`356/400=89.00%`；
- cycle 301–400：`750/800=93.75%`；
- cycle 401–480：`584/640=91.25%`；
- 最后 20 cycles：`150/160=93.75%`；
- 最后 10 cycles：`73/80=91.25%`；
- final cycle：`8/8=100%`，return `9.125`。

### 3.2 fixed-20 deterministic eval

| Cycle | 成功 / 20 | Cycle | 成功 / 20 |
|---:|---:|---:|---:|
| 25 | 0/20 | 275 | 18/20 |
| 50 | 0/20 | 300 | 20/20 |
| 75 | 0/20 | 325 | 16/20 |
| 100 | 0/20 | 350 | 17/20 |
| 125 | 0/20 | 375 | 19/20 |
| 150 | 1/20 | 400 | 18/20 |
| 175 | 6/20 | 425 | 16/20 |
| 200 | 16/20 | 450 | 19/20 |
| 225 | 11/20 | 475 | 18/20 |
| 250 | 18/20 | 480 | 17/20 |

续训 10 点合计 `178/200=89%`，前 5 点 `90/100=90%`，后 5 点
`88/100=88%`。线性斜率约为每 100 cycles `-1.41` 个百分点，而单点的离散分辨率是
5 个百分点，因此应解释为 `80%–100%` 的高位平台震荡。最终 eval return 为 `8.1`，
`success_once` 与 `success_at_end` 均为 `17/20`。

## 4. 优化与 replay

| 指标 | Cycle 250 | Cycle 480 | Last-20 mean |
|---|---:|---:|---:|
| actor loss | -0.081186 | -0.132776 | -0.131363 |
| BC loss | 0.012802 | 0.013183 | 0.013166 |
| weighted BC | 0.032005 | 0.032956 | 0.032914 |
| weighted Q | 0.113190 | 0.165732 | 0.164277 |
| actor grad norm | 2.820909 | 2.160029 | 2.180301 |
| critic loss | 0.001286 | 0.000813 | 0.000814 |
| critic grad norm | 0.170410 | 0.133467 | 0.127441 |
| Q0(policy) | 0.251534 | 0.368294 | 0.365060 |
| Q1(policy) | 0.238317 | 0.348915 | 0.345439 |
| Q(data) | 0.212281 | 0.330563 | 0.326895 |

解释：

- resume 边界 cycle 250→251 连续，没有 optimizer、target、schedule 或权重重置迹象；
- actor loss 始终满足 `weighted BC - weighted Q`，最大数值残差仅
  `2.24e-8`；变得更负主要来自 Q 项增强，而不是 BC 爆炸；
- actor/critic grad 全程最大 `3.2935/0.7899`，远低于 clip 10；
- critic loss 最终约 `8.1e-4`，无发散；
- final twin-Q gap 为 `0.01938`；保守 policy Q 与 data Q 的差为
  `0.01835`，约为 Q(data) 的 5.6%，属于温和差距，未见 Q 分叉；
- Q 仍缓慢上升而 fixed-seed eval 已平台化，说明数值训练健康，但新增训练的边际收益主要
  表现在 train rollout 更稳定，不能外推成 eval 泛化持续提高。

resume 的状态闭合为：

| 项目 | Cycle 250 | Cycle 480 | 增量 |
|---|---:|---:|---:|
| critic update | 102,260 | 215,055 | 112,795 |
| actor update | 51,130 | 107,528 | 56,398 |
| rank0 replay | 17,170 | 28,460 | 11,290 |
| rank1 replay | 17,681 | 28,950 | 11,269 |
| global replay | 34,851 | 57,410 | 22,559 |

`112795 = 22559 × UTD5` 精确成立，final pending update budget 为 0。两 rank replay 均
低于各自 50,000 cache 上限。

TensorBoard 在 final cycle 写出的 `train/rlt/update_step=214580` 是该 cycle 更新前值；
同 cycle 随后执行 475 次 critic update，所以 completion/checkpoint 的精确终值为
`215055`。

## 5. 资源

| 指标 | 续训 251–480 | 原 1–250 |
|---|---:|---:|
| GPU0 active mean / P95 / peak | 29.0% / 100% / 100% | 30.1% / 100% / 100% |
| GPU1 active mean / P95 / peak | 30.1% / 100% / 100% | 27.7% / 100% / 100% |
| GPU0 / GPU1 显存峰值 | 19.37 / 19.56 GiB | 19.37 / 19.51 GiB |
| matched process RSS 峰值 | 78.75 GiB | 87.42 GiB |
| EnvWorker RSS 峰值 | 52.46 GiB | 62.05 GiB |
| cgroup anon 峰值 | 73.03 GiB | 82.43 GiB |
| cgroup file 峰值 | 165.89 GiB | 191.01 GiB |
| cgroup current 峰值 | 240.00 GiB | 240.00 GiB |
| memory.max 增量 | 11,485 | 249,188 |
| OOM / OOM-kill | 0 / 0 | 0 / 0 |

两卡负载是 rollout/train 交替形成的突发负载，并非持续顶满；显存只用到约 24%。
但是 cgroup current 仍触及 240 GiB，上限压力主要来自 EnvWorker/anon 的运行期增长和
大量 file cache。因此不能仅凭 GPU 余量扩大并行。

续训结束后：

- matched training-process RSS 已归零；
- anon 约 `0.29 GiB`；
- live file cache 约 `165 GiB`；
- host available 约 `982 GiB`；
- memory PSI `avg10/60/300` 均为 0。

现场约 177 GB 十进制的 cgroup file 项是 Linux page cache，不是仍存活的训练进程堆内存；
内核可在压力下回收，不需要为了本次验收主动清 cache。

续训磁盘可用空间从 `819.19 GiB` 降至 `807.55 GiB`，减少 `11.64 GiB`，仍远高于
200 GiB 停止线。续训粗均值约 `147.8 s/cycle`，未出现随资源增长而整体减速。

## 6. Checkpoint 与产物

新 run 保存 10 个 checkpoint：

```text
275, 300, 325, 350, 375, 400, 425, 450, 475, 480
```

全部 completion 都是：

```text
complete=true
actor_world_size=2
schema_version=1
rlt_resume_contract_sha256=
  82cd409bef1549afb3feb41fa5a80ed08d207d112e4dfe8020af14f49cad1fc9
```

10 个 checkpoint 合计 `11,967,404,588 bytes`（约 11.15 GiB）和 482,255 个文件；
final480 为 `1,415,135,834 bytes`（约 1.32 GiB）。final rank state SHA-256：

- rank0：`193dfa8df3ddd34e7b7e5fd3768763f77d42a22a84be5eab72d73c22928e87c1`；
- rank1：`ebd426d9825fff5d3a19b6d684e7ef6d1d0d5fe43c9f3194ac836fbc290af444`。

服务器完整产物：

```text
run root:
  /root/autodl-tmp/experiments/
  rlt_stage2_formal_resume250_to480_20260730_v1

experiment:
  robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1

runtime/evidence:
  /root/autodl-tmp/experiment_exports/
  rlt_stage2_formal_resume250_to480_20260730_v1/runtime
```

本地高信息量只读快照：

```text
exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1
```

其中远端下载核心为 50 个文件、5,874,658 bytes；加下载 manifest 后为 51 个文件、约
5.62 MiB。加入本地分析 JSON、绘图脚本、README 和三张图后，完整文件夹为 57 个文件、
7,654,908 bytes（约 7.30 MiB）。下载 manifest 的 50 个远端条目已逐项通过 byte count
和 SHA-256 复核。它不包含模型权重或 replay `.pt`，不能独立 resume；完整可恢复
checkpoint 仍只保存在服务器。

## 7. Fatal 与已知告警

正式 fatal 扫描中 CUDA OOM、NCCL、Ray actor/rank death、NaN/Inf、SIGTERM/SIGKILL 和
训练 `ERROR` 均为 0。

日志中唯一两段 traceback 是两个 Env rank 启动时导入可选 Curobo planner 失败：
`missing pytorch3d` / `curobo.types.math` 不可用。随后立即加载 norm stats、恢复
checkpoint，并完整运行 cycle 251–480；这是本任务不使用的可选 planner 告警，不是训练
失败。FSDP/Vulkan/Ray CPU 提示也均未阻塞。

服务器 RLT worktree 当前比 upstream ahead 2。HEAD 和全部 provenance 已固定，因此当前
产物可复查；但若将来要在另一服务器重建环境，应先保留或推送这两个提交，不能只依赖旧
远端分支。

## 8. 图

- 成功率、阶段和 update 进度：
  `exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1/visuals/`
  `rlt-success-complete480.png`
- actor/critic/BC/Q/gradient：
  同目录 `rlt-optimization-complete480.png`
- 两段进程的 memory/RSS/GPU：
  同目录 `rlt-resources-complete480.png`

图中黑色虚线只表示 cycle 250 后进程重启；它不是算法阶段切换。四个算法阶段仍是：
P1 `1–135`、P2 `136–154`、P3 `155–191`、P4 `192–480`。

## 9. 当前停点

本轮只做只读服务器验收、本地小型日志/指标下载、分析、制图和文档收尾；没有启动、停止、
删除、覆盖、清 cache 或修改服务器训练产物。

RLT Stage 2 当前可称为：**正式 RoboTwin `adjust_bottle` 单任务运行已完整覆盖
480 cycles / 3,840 train episodes，resume 合同得到真实 load→continue 证据，训练数值
健康且 fixed-20 eval 位于高位平台。**

下一步若要继续扩展，不应直接因 Q 仍上升而盲目延长；优先选择独立 held-out seeds、
第二任务验证，或先定位 EnvWorker retained-memory。三者都超出本轮只读授权。
