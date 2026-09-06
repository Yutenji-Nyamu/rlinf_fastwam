# GRPO-DVAC Action-Adv 实施与 smoke 流水账

> 日期：2026-08-27  
> 机器：深圳 `SZ-H100`  
> 认证：固定 host-key Paramiko 密码路线；凭据只进入当前进程，不写入本文、脚本或仓库。

## IMP-000 — 授权与固定边界

用户明确授权：基于深圳当前两卡成功 GRPO，实现 raw `[0,2]` 的显式 action-level advantage；做简洁
必要检查；在两张空闲卡（优先 physical GPU2/3）运行真实两步 smoke；成功后普通 push 到用户 GitHub。

固定边界：

- 从服务器 clean current DVAC source `0e28ac6f...` 建独立 branch/worktree；不修改正在运行的工作树。
- 原始 trajectory-level `adv_type=grpo`、reward、group filter、采样/batch/update预算保持不变。
- 复用现有 `V_L3`、recent-5、global-z、raw `[0,2]` 权重与 resume sidecar。
- 新模式只做 `A_eff[i,h] = A_i * stopgrad(w[i,h])`，使用 current `action_level` ratio/clip。
- 新模式关闭现有 logprob straight-through，避免重复加权。
- 不做 mean-one、Prism/RLOO、all-ones formal、额外 reward 或新的 telemetry writer。
- 不停止或干预其他用户、现有正式训练和 shared Ray；GPU2/3不空闲就不启动。

## IMP-001 — 计划与验收

1. 只读核对服务器身份、GPU2/3、shared Ray、source HEAD/dirty、branch/worktree和remote。
2. 创建 `codex/sz-grpo-dvac-action-adv` 与独立 worktree。
3. 下载精确少量源码快照，在本地用 `apply_patch` 形成最小增量，再经SFTP放回专用worktree。
4. 运行 `diff --check`、聚焦单测、changed-file compile/ruff与两份Hydra compose。
5. commit并普通push专用branch。
6. 生成相对两卡Control的resolved parity与2-step smoke packet。
7. 若GPU2/3仍空闲，fresh运行2 outer steps；inline eval关闭，只验真实rollout/update/终态checkpoint。
8. Step1应为warm-up全1权重；Step2应出现非均匀权重和有限的action-level loss/clip/grad；自然exit0。

从下一个条目开始逐条补充命令、结果、问题与窄修。

## IMP-002 — 16:39 CST实施前只读现场

精确命令：`local_scripts/remote_commands/sz_action_adv_preflight_readonly_20260827.sh`，通过普通账号
`chenyiteng`和固定host-key Paramiko执行，exit 0。

结果：

- hostname=`admin`，UID1003；source HEAD=`0e28ac6f09f821ea12e7d54eba7118ce0000ca86`，
  branch=`codex/sz-current-pi0-dvac-grpo-w0to5`，tracked clean。
- 新branch `codex/sz-grpo-dvac-action-adv`与target worktree均不存在；personal remote正常。
- GPU2/3均8 MiB、0%，无compute PID；GPU4/5为既有Control，GPU6/7为既有Prism formal，均不操作。
- host `MemAvailable=892,221,884 KiB`；`/`、`/home`、`/data`分别余约226 GiB、1.4 TiB、2.1 TiB。
- shared Ray `172.17.0.1:6389`健康，无pending/failure。

## IMP-003 — 独立worktree

精确命令：`local_scripts/remote_commands/sz_action_adv_create_worktree_20260827.sh`。

从clean `0e28ac6f...`创建：

```text
branch: codex/sz-grpo-dvac-action-adv
worktree: /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
```

创建后HEAD与base一致、branch正确、status clean；原DVAC、Control、Prism工作树未修改。

## IMP-004 — 精确源码与首版上传

服务器target的5个相关文件SHA与旧本地审计树不同，因此没有复用旧快照。通过
`local_scripts/sync_shenzhen_action_adv_sources.py get`一次SFTP会话下载target exact base到
`references/action_adv_impl_20260827/base/`并复制工作副本到`work/`；五个SHA均与服务器现场一致。

本地仅用`apply_patch`修改四个文件：

- actor：增加`logprob_st | action_advantage` application；新模式关闭ST，把冻结权重传给loss；
  step artifact与resume sidecar记录application，旧ST sidecar缺字段时按`logprob_st`兼容。
- `algorithms/utils.py`：在current action-level shape整理完成后，构造`A_eff=A*w.detach()`。
- 原GRPO YAML：只加default `application: logprob_st`，保持所有旧run默认行为。
- 既有DVAC focused test：增加精确shape/数值/detach与错误logprob_type测试。

未修改`advantages.py`、`losses.py`、π0/OpenPI、RoboTwin、rollout/schema、reward、group filter、optimizer或
checkpoint主体。上传前再次验证HEAD/branch/status和四个base SHA；随后在一次固定host-key SFTP会话中原子
替换上述四文件。原DVAC weighting模块没有变化、没有上传。

## IMP-005 — 最小静态检查与聚焦单测

精确命令：`local_scripts/remote_commands/sz_action_adv_minimal_checks_20260827.sh`。

首次执行只在 `ruff format --check` 停止：actor 文件需要机械格式化；前置的 `git diff --check` 与
`ruff check` 已通过。随后只对该 actor 文件运行一次 `ruff format`，未改语义，再完整重跑同一检查：

- `git diff --check`、changed-file `ruff check`、`ruff format --check`、Python compile：通过；
- `tests/unit_tests/test_dvac_train_weighting.py`：`9 passed in 3.19s`；
- control Hydra compose：`logprob_type=chunk_level`、`application=logprob_st`；
- method Hydra compose：`logprob_type=action_level`、`mode=apply`、
  `application=action_advantage`、权重范围 `[0,2]`；
- 最终增量限定为四文件：`+112/-23`，其中 actor 的部分行数来自同一机械格式化。

没有运行宽泛测试矩阵或增加额外兜底。

## IMP-006 — commit与push

精确命令：`local_scripts/remote_commands/sz_action_adv_commit_push_20260827.sh`。

- branch：`codex/sz-grpo-dvac-action-adv`；
- commit：`a5b94b6f10a9212502d6930f07543f61e31af52e`；
- message：`feat(embodiment): add DVAC action-level advantages`；
- personal remote：`Yutenji-Nyamu/rlinf_fastwam`同名branch，`ls-remote`与本地commit一致；
- commit后worktree clean。

## IMP-007 — 两步smoke resolved packet与启动

准备命令：`local_scripts/remote_commands/sz_prepare_action_adv_smoke2_20260827.sh`；启动命令：
`local_scripts/remote_commands/sz_launch_action_adv_smoke2_20260827.sh`。

```text
packet: /data/chenyiteng/results/rlinf-shenzhen/grpo/packets/
        dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1
run:    /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
        dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1
GPU:    physical 2,3
budget: 2 steps; 64 env x 4 = 256 trajectories/step; G8; max1024 records;
        GB1024/MB32/update2; inline eval disabled only for smoke
method: trajectory GRPO A_i; action-level ratio/clip;
        A_eff[i,h]=A_i*stopgrad(raw DVAC w[i,h] in [0,2])
stop:   complete global_step_2 checkpoint + exit0; hard timeout 7200s
```

same-code control与method resolved逐叶比较通过；method只改变`logprob_type`以及DVAC的`mode`、
`application`、`weight_min/max`五项。相对既有两卡Control的其它差异仅为smoke步数/关闭inline eval、物理卡和
输出路径，以及新源码增加的default `application`字段。

16:58:05 CST启动前GPU2/3均8 MiB、无compute PID；host约846 GiB available；GPU4--7既有任务未操作。
wrapper PID `769082`，observer PID `769086`，15秒启动探针通过。后续只读等待终态。

## IMP-008 — 两步smoke终态

17:56 CST自然结束，`runtime/exit_code.txt=0`；driver完整到`Global Step 2/2`，未见
Traceback、OOM、WorkerCrashed、nonfinite或fatal。GPU2/3均回到9 MiB、0%，已释放。

step artifact使用0-based命名：outer Step1为`runner_step_0000.pt`，outer Step2为
`runner_step_0001.pt`。两rank只读tensor核对：

- Step1：`application=action_advantage`、`warmup=true`；weights shape=`[4,128,50]`，
  min=max=mean=1、std=0，逐元素exact all-ones；base/effective advantage均finite。
- Step2：`warmup=false`；rank0权重min/max/mean/std=`0/2/0.95710/0.48410`，rank1为
  `0/2/1.01806/0.47248`，均非全1且finite；effective advantage shape=`[4,128,50]`且finite。
- Step2总体日志：weight mean=`0.988`、ESS=`0.809`；grad norm=`0.476`、policy loss=`0.020`、
  total loss=`0.0013`、KL=`-0.0012`、clip fraction=`0.0091`，均finite。

完整checkpoint：

```text
/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1/
robotwin_dvac_action_adv_w0to2_smoke2_2gpu64x4_b1024_noeval_phys23_v1/
checkpoints/global_step_2
```

其中包含DCP `.metadata`、两个`.distcp`、`full_weights.pt`和两份actor DVAC sidecar；sidecar锁定
world size2、`application=action_advantage`、mode apply、L3、raw `[0,2]`与runner step2，并保存recent
stats。实现、两步真实rollout/update、非均匀权重、终态保存与自然退出全部闭环；没有启动formal。

## IMP-009 — formal packet与逐叶对照

2026-08-28用户授权：停止GPU4/5上的两卡clean GRPO Control，以相同两卡启动Action-Adv formal-100；
GPU6/7上的Prism保持不动。准备脚本：
`local_scripts/remote_commands/sz_prepare_action_adv_formal100_phys45_20260828.sh`。

formal从Control resolved逐叶继承：`64 train env×4=256 trajectories/step`、G8/32组、max1024 records、
GB1024/MB32/update2、fixed32/eval5/save10/100步，同模型、seed、环境、优化器与视频合同。方法只改变：

- `logprob_type: chunk_level -> action_level`；
- DVAC `mode: off -> apply`；
- `application: logprob_st -> action_advantage`；
- `weight_min/max: null/null -> 0/2`；
- 方法实验名及其派生的run内DVAC输出目录。

第一次packet生成在严格差异检查处停止：方法实验名会派生不同的DVAC telemetry `output_dir`；该路径是
run-scoped命名而非科学参数。只将这一观察到的派生路径加入允许差异后重新compose，
`unexpected_reference_diff=[]`。没有改采样、batch、评估、保存或算法其它字段。

## IMP-010 — 精确切换与并发隔离

执行脚本：`local_scripts/remote_commands/sz_cutover_control_to_action_adv_formal100_phys45_20260828.sh`。

切换前Control wrapper/PGID=`3882371`、job=`3e000000`、namespace=`RLinf`、GPU4/5；Prism
wrapper/PGID=`2485082`、job=`5a000000`、namespace=`RLinf_1`、GPU6/7；两边各15个named actors。

只对Control执行：TERM其owned PGID，再按exact namespace杀其15个named actors；未执行`ray stop`。
Control完整到Step96，停止标记写入`runtime/stopped_by_user_for_action_adv.txt`。清理后确认GPU4/5无旧job，
Prism wrapper仍alive、15个actor集合未变、GPU6/7仍全属job `5a000000`。

随后在同一persistent Ray上fresh启动Action-Adv：

```text
run=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
    dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
GPU=4,5
source=a5b94b6f10a9212502d6930f07543f61e31af52e
```

旧Control停止到新wrapper启动的空窗为2秒。

## IMP-011 — formal启动终态

11:06:46 CST只读启动复核：新wrapper PID=`105735`、Ray job=`5f000000`、namespace=`RLinf`、15个
actors完整，GPU4/5上的6个compute process均属新job；已进入首个`Generating Rollout Epochs 0/4`，
fatal=0。Prism仍为原PID/job/namespace并继续推进；shared Ray未重启。该状态满足“正常开始后停止主动盯守”。

## IMP-012 — 被替换Control轻量封存

Control完整到Step96：末步train success=`92.97%`，末5/末10均值=`94.69%/94.73%`；19次fixed32累计
`570/608=93.75%`。轻量包包含完整driver/resource日志、resolved/contract/parity/command、TensorBoard、
逐步CSV和三张图；排除checkpoint、视频、Ray全量日志与逐动作tensor：

`exports/shenzhen_grpo_control_2gpu_stopped_step96_light_evidence_20260828.zip`（417,914 bytes）。

## IMP-013 — Action-Adv reduction 根因与窄修

2026-08-28用户授权修复并push。只读审计确认旧Action-Adv的
`A_eff=A*stopgrad(w)`、shape、shuffle、detach和ST互斥均正确；真正错误是
`compute_ppo_actor_loss`继续把 `[B,H]` policy loss在 $H=50$ 上取均值。前21步pre-clip grad
Control/旧Action均值为`31.229/0.685`，约差45.6倍，与理论 $H=50$ 一致。

执行入口：

- patch：`local_scripts/patches/sz_action_adv_fix_20260828.patch`
- 服务器应用/检查/push：
  `local_scripts/remote_commands/sz_create_test_push_action_adv_fix_20260828.sh`

生产修改仅3 files：

1. `losses.py`增加显式`action_level_sum`，对有效action loss先求和再对query平均；
2. 同文件将rank-2 action metric mask扩展到`[B,H]`，修正ratio/clip统计分母，同时保留
   chunk-summed KL口径；
3. actor只在`application=action_advantage`时启用该reduction，并在step artifact记录语义；
4. 增加一个小测试覆盖loss尺度、ratio、clip fraction和KL口径。

结果：ruff与format通过，10项聚焦测试通过，Hydra resolved compose通过；commit
`e434f409b21d281ce883df29487ecae7cb3e4839`已push到
`codex/sz-grpo-dvac-action-adv-fix`。

## IMP-014 — 新双实验packet与逐叶对照

执行：

`local_scripts/remote_commands/sz_prepare_action_adv_fix_and_st_half_dual_formal100_20260828.sh`

生成两套fresh100 packet：

- Action-Fix `[0,2]` physical GPU4/5；
- ST global-z `[0.5,1.5]` physical GPU6/7。

两者共同为`64×4/G8/max1024/GB1024/MB32/update2/fixed32-eval5/default-DCP-save10`；
逐叶比较`unexpected_reference_diff=[]`。Action方法差异仅
`action_level/apply/action_advantage/[0,2]`及run路径；ST方法差异仅
`apply/[0.5,1.5]`及run路径。显式端点使`strength`被有意忽略。

## IMP-015 — 精确停止旧Action/Prism并启动Fix/ST

执行：

`local_scripts/remote_commands/sz_cutover_action_prism_to_action_fix_st_half_dual_formal100_20260828.sh`

切换前动态解析：

- 旧Action：job `5f000000`、namespace `RLinf`、GPU4/5；
- 旧Prism：job `5a000000`、namespace `RLinf_1`、GPU6/7；
- 两边各15 named actors。

依次按exact owned PGID停止wrapper，再按exact namespace清理actors；shared Ray未停止，也未操作
GPU0--3。停止标记显示旧Action完整Step29、旧Prism完整Step55。

随后顺序启动：

- Fix run：`dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1`
- ST run：`dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1`

脚本在manager/actor注册后立即检查GPU job，当时CUDA context尚未出现，故输出
`gpu_processes=0`并在末尾assert返回1；两个wrapper和namespace已成功启动，未被该检查停止。22:54 CST
独立只读复核确认：Fix job `6a000000`仅在GPU4/5，ST job `74000000`仅在GPU6/7，各15 actors，
fatal=0，均进入`recv_rollout_trajectories/generate/interact`；GPU约11--18 GiB/card，host
MemAvailable约1.90 TiB。

## IMP-016 — 旧实验轻量封存

执行：

- `local_scripts/download_sz_action_adv_prism_stopped_20260828.py`
- `local_scripts/package_sz_action_adv_prism_stopped_20260828.py`

结果：

- Action Step29包：`exports/shenzhen_grpo_dvac_action_adv_w0to2_stopped_light_evidence_20260828.zip`，
  324,480 bytes、23 files；
- Prism Step55包：`exports/shenzhen_prism_dvac_rank_rloo_v2_stopped_light_evidence_20260828.zip`，
  354,409 bytes、22 files。

两包均含driver/resource、resolved/contract/parity、TensorBoard、核心CSV与success/optimizer/resource图；
排除checkpoint、视频、Ray全量日志与逐action tensor。
