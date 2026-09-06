# Idea2 DVAC telemetry：实现结果与首次 smoke 审阅包

更新时间：2026-08-20。本文只回答三件事：实现并验证了什么、真实π0+RoboTwin smoke怎样运行并完成、
首批数据之后还没有分析什么。逐指令过程与问题修复见
[实施与前测流水账](evidence/IMPLEMENTATION_AND_PRETEST_LEDGER.md)；本次真实运行的逐指令过程单列在
[2GPU/16-env smoke执行流水账](evidence/SMOKE_2GPU16ENV_EXECUTION_LEDGER.md)。

## 1. 当前结论

默认关闭的 π0 DVAC endpoint telemetry 已在独立 AutoDL worktree 完成并推送：

```text
worktree: /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch:   codex/idea2-dvac-pi0-robotwin
commit:   61996e15cc7f5a32bd6012b61b20893d94636c82
remote:   personal/codex/idea2-dvac-pi0-robotwin，HEAD已复核一致
base:     6d0db56bf26f972cd27fa29535f5eb939e80e5bf
```

实现只覆盖 standalone eval 的原始 π0 SFT 路线；DSRL、RLT、NFT、expert/decoupled、多 epoch、训练和
`runner.ckpt_path`都显式拒绝。两卡、16-env、1-epoch真实smoke已于19:03:17启动、19:07:11自然完成，
driver exit code为0；64-query/16-episode数据合同、资源和视频后检均通过。

## 2. 实现了什么

### 2.1 模型旁路

在 π0 已有四步 flow 去噪循环中，telemetry 开启时旁路保存：

```text
x_chain       [B, M+1, H, 14]
z_endpoint    [B, M,   H, 14]
timesteps     [M]
final_action  [B, H, 14]，normalized model action
```

其中 `M=4`、`H=50`，每一步使用已经存在的 `x_t_prev/v_t/t_i` 计算
`z_i=x_t_prev-t_i*v_t`；没有增加模型 forward，也没有改变原 noise 调用或 Euler update 顺序。代码入口在
`rlinf/models/embodiment/openpi/openpi_action_model.py` 的
`return_dvac_telemetry`分支和返回块。

### 2.2 worker 与索引

- RolloutWorker 只在显式开关打开且满足 plain-SFT 合同时，向模型注入
  `return_dvac_telemetry=True`；action 完成 CPU 搬运并发给 Env 后才记录数据。
- EnvWorker 直接从当前 wrapped RoboTwin env 读取 reset ID、elapsed action slots、success、env rank、
  stage、slot、eval epoch/query 和视频索引，不用 batch slot 猜 seed。
- telemetry v1 限定一个 eval epoch；若开启 telemetry 后出现非最后 chunk 的提前 reset，会停止并报告，
  不继续写无法可信 join 的 episode。该 fail-fast 已明确受 telemetry 开关门控，默认关闭路径不触发。

### 2.3 落盘合同

每个 rollout/env rank 独立写文件：

```text
dvac_telemetry/
  run_manifest.json
  resolved_config.yaml
  manifest_rollout_rankNN.json
  trace_rollout_rankNN.npz
  query_index_rollout_rankNN.csv
  episode_index_env_rankNN.csv
  query_images/rollout_rankNN/*.png
```

`trace_rollout_rankNN.npz`保留原始 `x_chain/z_endpoint/timesteps/final normalized action/env action/
14D state`；三路 policy 输入图以 lossless PNG 保存。`run_manifest.json`区分 common base 与实际 source
commit，并记录RLinf/RoboTwin commit、checkpoint revision、norm/seed hash、完整启动命令、主机/时间、
运行时读取的`H/D_model/D_active/C/M/dtype`及预期rollout/env shards。相同 rank 的目标文件已存在时直接
拒绝覆盖，因此每次运行必须使用唯一 output path。

方差 `V_L(h)`、坐标级 `U_L(h,d)`、threshold 和 crossing 都没有在线计算；后续从 raw trace 离线同时算
`L=2/3/4`。这保持了“先看信号，再决定如何训练”的两步路线。

### 2.4 配置与测试

新增 official-semantics 专用配置
`evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml`：默认仍为 official 128 env、H/C50、
M4、D14、三相机、fixed IDs、200 action slots，且 telemetry 默认关闭。获批smoke的两卡16-env完整
resolved 配置另存为
[SMOKE_RESOLVED_2GPU_16ENV_V1.yaml](evidence/SMOKE_RESOLVED_2GPU_16ENV_V1.yaml)，
不放进 source worktree，避免为运行参数弄脏已锁定实现代码。

服务器当前通过：

```text
Python builtin compile: 6/6
targeted Ruff: all checks passed
pytest tests/unit_tests/test_dvac_telemetry.py: 2 passed
Hydra dedicated config compose/contract: passed
external smoke config entrypoint --cfg job --resolve: passed
```

两项 pytest 验证：

1. 固定输入/noise/RNG 的 synthetic sampler 在 telemetry off/on 时 action、chain、最终 Torch RNG state
   相同，并逐步核对 `z=x-t*v` 和 shape；
2. writer 实际写回 NPZ/CSV/6 张 PNG/episode join，检查共享 `timesteps.shape==(4,)`、expected shard
   manifest 和旧 shard 拒绝覆盖。

这里要区分证据边界：这些是服务器上的 CPU synthetic sampler 与真实 writer/compose 测试，不等于真实
7.6G π0 checkpoint 的 CUDA forward，也不等于 RoboTwin episode；真实链路由本次获批smoke补齐。

## 3. 实施中发生的问题

完整原始过程都在流水账，关键问题没有隐藏：

1. 首次 sparse-worktree 同步方法把未展开的 314 个 tracked path编码成删除；补丁只应用到新建 Idea2
   worktree，发现后按精确 `diff-filter=D` 从 HEAD 全部恢复。未提交、未push、未碰其他 worktree；之后每次
   检查均为0删除，并废弃该同步方式。
2. 第一轮 sampler test 的 fake fixture 少了生产代码原有的 `action_in_proj.weight.dtype`，在进入新增
   telemetry逻辑前失败；只补fixture后同一测试通过。
3. 独立审阅发现 timesteps落盘shape、旧shard覆盖、提前reset join、manifest source标识和plain-SFT门控
   问题；均以首轮语义范围内的窄修复处理。
4. 提交后复核又发现提前done fail-fast漏了telemetry开关门控；在任何push/真实eval前修正、完整复测并
   amend为核心功能提交`73da63f0…`。
5. GitHub默认smart-HTTP探针30秒超时；没有误判为认证失败。按既有AutoDL流程只在一次性
   `/etc/network_turbo`子shell中完成非force push，退出后父shell无代理遗留。
6. smoke前最后一次合同复核发现run manifest还缺RoboTwin/seed/launch来源及真实`H/D_model/dtype`；以
   3文件窄补丁补齐、复测，并普通fast-forward到当前`61996e15…`，没有amend已推历史或force push。

## 4. 视频结论与本次核对点

服务器上确实同时保留两类历史产物，但不是同一次 RLinf official eval 自动双写：旧
`/root/autodl-tmp/RoboTwin/eval_result`来自 standalone RoboTwin native 录像；当前 RLinf runtime 是
`/root/autodl-tmp/RoboTwin_RLinf`，VectorEnv 默认关闭 native `eval_video_log/render_freq`，本次实测只新增
RLinf `video/eval/seed_*/*.mp4`。

16-env配置产生两个RecordVideo worker，每个worker把8个环境拼成2×4 tile，实测为两支1280×480 MP4，
而不是16支单环境视频。没有显式`video_cfg.fps`，代码fallback为30 FPS；C50路径每次policy query只向
wrapper返回一个chunk后观测，因此每支实测6帧：初始、4个C50 chunk端点、1个auto-reset spill。
30 FPS只决定这6帧多快播放，不是仿真频率，也不增加信息。已用`ffprobe -count_frames`核对真实帧数，
并解帧检查tile、query前后帧和三路输入PNG。

## 5. 已获批的第一次真实 smoke

### 5.1 预算与并行

```text
GPU:                2 × A800-80GB
placement:          env, rollout: 0-1
total_num_envs:     16（每卡/每rank 8个）
rollout_epoch:      1
episode slots:      200
H / C / M / D:      50 / 50 / 4 / 14
expected episodes:  16
expected queries:   64（每episode最多4次policy query）
expected PNG:       192（64 queries × 3路policy输入）
expected MP4:       2（每支8-env tile）
```

选择16而不是原候选2：它保留最小串行度1，同时覆盖接近正常评估的并行与64个自然query-state；同机旧的
两卡16-env纯SFT eval已有约20 GiB/card工程记录。暂不直接上32，因为当前机器没有同等干净的32-env纯评估
资源实测。完整YAML中的`runner.max_epochs:1000`与`task_config.episode_num:100`是继承字段，不决定本次预算。

### 5.2 已上传且验证的完整配置

```text
本地审阅副本:
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/SMOKE_RESOLVED_2GPU_16ENV_V1.yaml

服务器配置:
/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml

SHA256:
d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018

完成后的输出目录:
/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1
```

关键source/assets：

```text
RLinf Idea2: 61996e15cc7f5a32bd6012b61b20893d94636c82, clean
RoboTwin_RLinf: 481380fbd97cbf9ff830aedfb2279851e1e58969
checkpoint: /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
checkpoint revision: 92684e50c8a3dcf13b76a06713e3152625967be1
norm_stats SHA256: 649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
seed bank SHA256: 194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f
```

### 5.3 精确核心命令

```bash
cd /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin && CUDA_VISIBLE_DEVICES=0,1 REPO_PATH=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin EMBODIED_PATH=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/examples/embodiment PYTHONPATH=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf /root/autodl-tmp/RLinf/.venv/bin/python evaluations/eval_embodied_agent.py --config-path /root/autodl-tmp/idea2_dvac_run_configs --config-name idea2_dvac_sft_smoke_2gpu_16env_v1
```

该字符串与YAML中的`launch_command`一致。实际wrapper在启动前验证source/remote HEAD、clean tree、配置
hash及唯一output；然后启动上述driver并等待其自然退出。没有`timeout`，也没有向driver发送任何信号。

## 6. 参数血缘

- **official原样**：`adjust_bottle`、原始task-matched π0 SFT、H/C/M/D=50/50/4/14、三路相机、clean、
  fixed reset IDs、200 slots、1 epoch、HF backend、offload、TensorBoard和RLinf视频。
- **同机经验适配**：8卡/128-env official placement缩为2卡；总并发取旧两卡纯SFT已跑通的16 env，当前
  每rank 8 env。服务器绝对checkpoint/assets/seed路径绑定当前AutoDL现场。
- **本次Idea2新增**：默认关闭的DVAC telemetry在这个外置配置里显式打开；使用唯一run/output；manifest
  增加source、checkpoint、norm/seed和shape来源字段。它们不改变action生成或环境交互。
- **明确没有继承**：旧10 epochs、actor placement、PPO/RLT、第二任务、任何阈值控制或资源驱动停止。

## 7. 资源观察与完成条件

独立observer每2秒只读记录GPU显存/利用率/温度/功率、cgroup current/anon/file/events、host available
RAM、`/dev/shm`、磁盘和GPU compute process数；每10秒记录本次driver/Ray/worker进程RSS。启动前后各保存
一次资源快照。observer以driver的`/proc/<pid>`是否仍存在决定何时自然退出。

它没有显存/RAM阈值、告警动作、timeout、signal或退出码联动；观测值没有改变运行行为。smoke由
`16 env × 1 epoch × 200 slots`自然结束。没有自动进入训练或扩大到32/128 env。

## 8. 真实smoke结果

```text
launcher wall:     19:03:17 -> 19:07:11，约3分54秒
driver exit:       0
source after run:  HEAD/remote仍为61996e15…，clean
residual process:  0；GPU compute process 0
output:            10,129,300 bytes，206 files
queries/episodes:  64 / 16；unique reset IDs 16
rank split:        每rank 32 queries + 8 episodes
query images:      192张RGB 320×240
smoke outcome:     14/16 success_at_end；只作链路事实，不作为official-128成功率
```

两支NPZ均满足：`x_chain[32,5,50,14]`、`z_endpoint[32,4,50,14]`、
`timesteps=[1,.75,.5,.25]`、`final_model_action/env_action[32,50,14]`、`robot_state[32,14]`；全部
float32、finite，final action与`x_chain`终点相等。64×50个`query×h`的L2/L3/L4总体方差均非零且finite，
说明首批内部信号不是全零/常量；这只是sanity，不在本轮选阈值或解释任务阶段。

资源实测：GPU0/1峰值19,132/19,102 MiB（18.684/18.654 GiB）；cgroup current峰值55.226 GiB；
`memory.events`从头到尾全0。最大两个RolloutWorker RSS约13.89/13.14 GiB，两个EnvWorker约13.80/13.43
GiB。完整汇总在runtime的`resource_summary.json`，原始2秒观测保留在`resource_monitor/resources.csv`。

启动日志中的Curobo import traceback来自optional backend的主动`traceback.print_exc()`；本次实际backend是
`mplib`，异常被捕获并设`CuroboPlanner=None`，历史成功评估也有同样提示。本次无需安装、改环境或重跑。

## 9. 视频实测与可调参数

两支RLinf MP4均为H.264、1280×480、30 FPS、实际解码6帧、0.200秒；每支是8环境2×4 tile。画面对应
初态、4个C50 chunk端点和auto-reset spill；三路query PNG是独立的320×240 policy输入。没有同时间窗的
RoboTwin native第二套MP4。contact sheet与原视频已下载到本地evidence目录供直接查看。

本地证据入口：

- [seed0 contact sheet](evidence/SMOKE_VIDEO_SEED0_CONTACT.png) / [seed1 contact sheet](evidence/SMOKE_VIDEO_SEED1_CONTACT.png)
- [seed0 MP4](evidence/SMOKE_VIDEO_SEED0.mp4) / [seed1 MP4](evidence/SMOKE_VIDEO_SEED1.mp4)
- [代表性q0 head输入](evidence/SMOKE_QUERY_RANK00_EP000000_Q000_HEAD.png)
- [资源汇总](evidence/SMOKE_2GPU16ENV_V1_RESOURCE_SUMMARY.json) / [run manifest](evidence/SMOKE_2GPU16ENV_V1_RUN_MANIFEST.json)

- 只想看慢一点：下次加`env.eval.video_cfg.fps:1`或`2`，或离线慢放；它不增加帧、不改推理。
- 想看更多过程：现有RLinf配置没有C50内部逐action录像开关。把C改为10会同时改变闭环策略，不应用来
  “修录像”。若首批信号分析确实需要，再单独设计默认关闭的RoboTwin逐action recorder。
