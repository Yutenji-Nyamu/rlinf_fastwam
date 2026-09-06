# Idea2 训练实现、前测与 smoke 逐操作流水账

日期：2026-08-20 至 2026-08-21  
机器：Windows 本地（文档/轻量代码副本）与 AUTODL-A800（权威源码、测试、smoke）  
授权：用户已依次授权v1训练实现/smoke/100-step formal、在完整step后主动停止v1并整理轻量closeout，
以及v2 R-only两步smoke；当前v2 formal仍未授权。资源监控只记录，不发送信号或改变运行行为。

## 冻结目标

- 以服务器 Idea2 `61996e15...` 为父提交派生独立训练分支/worktree。
- 保持历史成功 GRPO 的任务、模型、`H=C=50`、chunk reward/logprob、joint ratio/clip、优化器、batch 与两卡路径。
- 训练 rollout 当场旁路计算 `V_L2/L3/L4[h]`；不增加模型 forward，不改 action/RNG。
- 第一个 runner step 使用 `w=1` 并建立 train-SDE 的 `log(V_L3+eps)` 统计；随后使用最近 5 个已完成 runner step 的全部已观察 `query×h` 统计：
  `s=clip((log(V+eps)-mu)/max(sigma,1e-6),-2,2)`，`w=1+strength*s`，首版 `strength=0.1`。
- 在逐 action log-prob 汇总成 chunk log-prob 之前做前向恒等、反向按 `w[h]` 缩放的窄挂点；不切换到 action-level PPO。
- 细录像默认关闭；打开后只旁路采样 head frame 与 success，近似 action 粒度，不重新规划、不拆分 C50，并限制训练抽样规模。

## 操作记录

### L001｜刷新工作区规则、当前交接、专题主计划与训练规划

命令：

```powershell
Get-Content -Raw PROJECT_CONTEXT.md
Get-Content -Raw HANDOFF.md
Get-Content -Raw docs\rlinf-robotwin-pi0-dvac-telemetry\00_INDEX_AND_PLAN.md
Get-Content -Raw docs\rlinf-robotwin-pi0-dvac-telemetry\05_TRAINING_MODIFICATION_PLAN.md
```

结果：确认当前权威服务器父提交为 `61996e15...`，旧100-step GRPO工程锚点为 `6d0db56...`；此前授权只到只读规划，本轮用户已新增实现、前测和smoke授权。旧规划中的 artifact/per-`(h,i)`/mean-one 路线已被本轮更直接的 rolling-global 首版设计取代，待更新正文。

### L002｜轻量查阅既有 SSH/GRPO memory 索引

命令：

```powershell
rg -n "GRPO|Paramiko|DVAC|Idea2|training" E:\Codex\home\memories\MEMORY.md
```

结果：定位到既有低层 Paramiko 密码路线与历史 GRPO 运行路径；所有动态状态仍须由本轮服务器现场刷新。

### L003｜核对本地 Git 可见状态与现有远程 helper

命令：

```powershell
git status --short
rg --files tmp local_scripts docs/rlinf-robotwin-pi0-dvac-telemetry/evidence | rg "paramiko|remote_exec|verified|TRAINING"
Get-Content -Raw local_scripts\remote_exec_autodl.py
git -c safe.directory=C:/Users/86136/Documents/rl status --short
```

结果：普通 `git status` 因 Codex sandbox 用户与目录owner不同触发 Git dubious-ownership；没有修改全局配置，后续只对单条Git命令使用 `-c safe.directory=...`。确认 `remote_exec_autodl.py` 使用固定主机指纹、低层 `Transport.start_client()/auth_password()`、认证前有界重试和认证后keepalive，密码仅从当前进程环境读取。工作区文档与各镜像均为既有untracked内容，本轮必须只改当前专题文件。

### S001｜建立单一低层 Paramiko 会话并完成身份探针

连接方式：Codex bundled Python 3.12 交互进程中用 `getpass`读取密码到当前进程环境，再调用既有
`remote_exec_autodl.connect()`；固定host-key、`Transport.start_client()`、密码认证和keepalive均成功，
后续命令复用同一`SSHClient/Transport`。密码没有写入文件、账本或远程命令。

远程命令：

```bash
date -Is
hostname
pwd
id -u
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
```

结果：`2026-08-20T22:08:20+08:00`，container
`autodl-container-nekaqbwt43-6ce5babb`，`/root`，UID 0；两张A800-SXM4-80GB均
`0 MiB / 0%`，主机available RAM约`980 GiB`。

### S002｜首次合并现场审计在cgroup路径引用处提前退出

命令意图：继续读取cgroup、磁盘、进程、Idea2/RoboTwin/baseline Git状态和历史checkpoint。

问题：远程loop中的cgroup文件名被多包了一层字面双引号，`cat`尝试读取名称中包含`"`的路径；
`set -e`令命令在第一项退出，rc=1。

原因与解决：这是命令引用错误，不是服务器或cgroup问题。拆成四条固定路径后只读复测；没有重放任何
写操作。

### S003｜修正后刷新资源、进程、source与历史工程锚点

主要命令：

```bash
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.high
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
ps -eo pid,ppid,stat,rss,etimes,args --sort=-rss | head -n 20
git -C /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin rev-parse HEAD
git -C /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin status --short
git -C /root/autodl-tmp/RoboTwin_RLinf rev-parse HEAD
git -C /root/autodl-tmp/RLinf rev-parse HEAD
sha256sum /root/autodl-tmp/RLinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml
find <historical-run> -maxdepth 5 -type d -name global_step_100
```

结果：

- cgroup current `2,907,787,264` bytes，max `257,698,037,760`，high
  `253,403,070,464`；`high/max/oom/oom_kill=0`。
- `/root/autodl-tmp`可用`824G`；无Ray、训练或GPU计算进程。
- Idea2权威worktree clean，HEAD=`61996e15cc7f5a32bd6012b61b20893d94636c82`，branch
  `codex/idea2-dvac-pi0-robotwin`。
- 历史RLinf为`6d0db56...`，只有既有untracked配置/helper；GRPO YAML SHA256仍为
  `e1dd7b2e...d6d6`，`global_step_100`目录仍存在（正文完整性尚未在本项检查）。

### S004｜澄清RoboTwin的特殊Git布局

初次从`RoboTwin_RLinf`执行无路径限制的`git status`，输出了大量`../RLinf`变化。只读追踪确认这不是
RoboTwin源码突然被改坏，而是其Git top-level实际为`/root/autodl-tmp`：外层`wamppo`仓同时跟踪
`RLinf/`与`RoboTwin_RLinf/`两个子树；各RLinf专题又是嵌套的独立worktree。

核对命令：

```bash
git -C /root/autodl-tmp/RoboTwin_RLinf rev-parse --show-toplevel --show-prefix HEAD
git -C /root/autodl-tmp/RoboTwin_RLinf status --short -- .
git -C /root/autodl-tmp remote -v
git -C /root/autodl-tmp show HEAD:RoboTwin_RLinf/envs/_base_task.py | sha256sum
sha256sum /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py
git -C /root/autodl-tmp show HEAD:RoboTwin_RLinf/robotwin/envs/vector_env.py | sha256sum
sha256sum /root/autodl-tmp/RoboTwin_RLinf/robotwin/envs/vector_env.py
du -sh /root/autodl-tmp/RoboTwin_RLinf/assets
```

结果：外层repo HEAD=`481380fbd97cbf9ff830aedfb2279851e1e58969`、`origin/main`，remote为
`Yutenji-Nyamu/wamppo`；两个目标RoboTwin源码文件与该commit字节一致。RoboTwin子树只有既有untracked
`policy/starvla_policy`符号链接，16G `assets/`不在Git中。后续不能直接改共享子树；将从外层commit派生
独立worktree，并只给新worktree的RoboTwin子树链接既有只读assets。

### S005｜派生两个独立实施worktree

远程命令：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin worktree add -b codex/idea2-dvac-train-weighting /root/autodl-tmp/RLinf_idea2_dvac_train 61996e15cc7f5a32bd6012b61b20893d94636c82
git -C /root/autodl-tmp worktree add -b codex/idea2-dvac-control-trace /root/autodl-tmp/idea2_dvac_train_wamppo 481380fbd97cbf9ff830aedfb2279851e1e58969
ln -s /root/autodl-tmp/RoboTwin_RLinf/assets /root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf/assets
mkdir -p /root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf/eval_result
git -C <new-worktree> rev-parse HEAD
git -C <new-worktree> status --short
```

结果：RLinf实施树为`/root/autodl-tmp/RLinf_idea2_dvac_train` /
`codex/idea2-dvac-train-weighting` / `61996e15...`；RoboTwin外层实施树为
`/root/autodl-tmp/idea2_dvac_train_wamppo` / `codex/idea2-dvac-control-trace` / `481380fb...`。
只新建assets符号链接和空e`eval_result`，没有复制16G assets、修改共享运行树或删除任何文件。

### S006｜一次无界网络探针卡住，只停止本地控制进程

问题：在既有Paramiko交互控制器里执行了没有有界超时的`git ls-remote`，网络等待导致本地
Python controller长时无返回。

处理：只终止了本地controller PID `11672`；没有给服务器训练、Ray、Git或其他进程发信号，
该命令也没有任何远程写入。随后重建同一固定host-key的低层Paramiko会话；后续Git网络操作必须有界。

### L004｜下载权威source的窄文件副本并开始一个连贯实现批次

操作：通过已认证SFTP从两个新worktree只下载将修改的源文件；历史GRPO YAML因原仓中是
untracked文件，从`/root/autodl-tmp/RLinf`读取已核对SHA的副本。本地目标为
`tmp/idea2_train_impl_source/`。

首批新建/修改：

- 新建`rlinf/algorithms/dvac_train_weighting.py`：endpoint variance、recent-step统计、连续权重、
  straight-through梯度缩放和rank-local writer；
- 新建`tests/unit_tests/test_dvac_train_weighting.py`：公式、warmup/window、mask和前向恒等/反向缩放测试；
- 修改`huggingface_worker.py`：只在train `mode=apply`时复用已有endpoint旁路，当场降维为
  `V_L2/L3/L4[B,H]`并携带query metadata；`off`不请求endpoint。

本地检查：

```powershell
<Codex bundled python> -m py_compile \
  rlinf/algorithms/dvac_train_weighting.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
```

结果：三个文件语法检查通过。Windows launcher `py -3`未找到系统Python；改用Codex bundled
Python后通过，没有在本机安装任何依赖。项目import/pytest/compose仍留到服务器既有环境。

### L005｜完成actor、环境元数据和细帧录像实现

继续在`tmp/idea2_train_impl_source/`用`apply_patch`完成一个连贯批次：

- 修改`rlinf/workers/actor/fsdp_actor_worker.py`：在shuffle前用全rank pooled float64统计生成并冻结
  `[query,H]`权重；在原`[B,H,D]`log-prob求和前应用straight-through梯度倍率；每个runner step
  成功训练后才推进recent-5状态；写rank-local NPZ/summary/state。
- 修改`rlinf/workers/env/env_worker.py`和`rlinf/envs/robotwin/robotwin_env.py`：给每个训练query附上
  rollout/env/reset/query/action-slot身份；默认关闭路径不附加DVAC请求。
- 新建`RoboTwin_RLinf/envs/control_trace.py`并窄改`_base_task.py`、`vector_env.py`：在既有control loop
  已完成render后读取head camera，以左右臂TOPP进度的较小值近似映射到`h`；只在新h-bin、success或
  query终点编码，不重新调用planner/TOPP/policy或随机函数。
- 新建专用smoke YAML：历史GRPO主体参数不变，只用`max_steps=2`、`rollout_epoch=8`，开启DVAC apply
  和单个`worker0/slot0` control trace样本。

本地只做Python语法检查。一次把YAML误传给`py_compile`得到预期`SyntaxError`；修正输入只包含`.py`
后通过。Codex bundled Python没有项目`torch/pytest`，未安装依赖，项目测试继续留在服务器。

### S007｜上传暂存实现并完成第一次服务器前测

通过同一已认证SFTP只上传上述窄文件到两个新worktree。主要命令：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile <changed Python files>
PYTHONPATH=/root/autodl-tmp/RLinf_idea2_dvac_train:/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf \
  /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_dvac_train_weighting.py \
  tests/unit_tests/test_dvac_telemetry.py
PYTHONPATH=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf \
  /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  RoboTwin_RLinf/tests/test_control_trace.py
git -C <worktree> diff --check
```

问题与窄修复：

1. import探针最初引用了不存在的`MultiStepEnvWorker`类名；源码真实类为`EnvWorker`。只改探针后
   `IMPORT_OK`，实现本身无需改。
2. 训练每个runner step后会offload/recreate SubEnv，首版episode counter会回到0并在step2撞同名录像
   目录。把“本次driver已认领的样本”状态提升到长寿命VectorEnv并用exclusive claim文件后修复；不使用
   覆盖旧目录的`exist_ok=True`。
3. ffmpeg在`terminate()`后若仍不退出，第二次`wait(timeout=5)`可能把可选录像错误抛回训练。补上
   第二层timeout→kill并只记录encoder error，side channel不再中断环境清理。
4. actor writer起初把trajectory的`T+1 dones`与`T`个query信号并列。改成显式
   `done_before=dones[:-1]`、`done_after=dones[1:]`并增加shape断言。
5. 对配置增加`strength*z_clip <= 1`校验，避免错误参数产生负权重、反转advantage方向。
6. recent统计从“loss-mask有效query”改成“本step全部已观察query×h”；这既符合用户提出的全池统计，
   也保证2-step smoke的第1步即使GRPO filter屏蔽同质组，仍能给第2步提供DVAC历史。实际梯度参与者的
   weight/advantage汇总仍使用loss mask。

复测结果：RLinf telemetry+train共`8 passed, 3 warnings`；RoboTwin control trace共`4 passed`；
warnings来自既有TensorFlow/SWIG依赖，没有test failure。新增合成trajectory测试覆盖`T/T+1`边界、
stack/reorder/shuffle和straight-through前向/反向；`git diff --check`与`py_compile`均通过。

### S008｜解析新/旧配置并做编码探针

命令/产物：

```bash
# 新2-step入口完整resolve
<python> train_embodied_agent.py --config-path <new-config-dir> \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_train_2step_smoke \
  --cfg job --resolve
sha256sum /root/autodl-tmp/idea2_dvac_train_pretest/resolved_2step_smoke_v2.yaml

# 历史baseline经新source解析，确认没有dvac key
<python> train_embodied_agent.py --config-path <new-config-dir> \
  --config-name robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline \
  --cfg job --resolve
```

结果：新resolved SHA256=`5633cc32b787528b8b6a12ec5d4eee27be732f7312df1f0135b114187f2c324e`；
确认2GPU、16 env、rollout epoch8、H/C50、active D14、M4、flow_sde、G8、B512/mb32、update2、
clip_grad1、max_steps2。历史default-off resolved SHA256=
`657c82015034e90a826af764641a967d3fba00520bc47784bd53874a19238edb`，输出
`DEFAULT_OFF_CONFIG_ABSENT_OK`。

用10张合成RGB frame走真实`ControlTraceRecorder`/ffmpeg：输出160×120、10 FPS、10帧，MP4 1,917B、
CSV 1,217B、metadata 769B；编码和自然finish通过。该探针不启动RoboTwin或训练。

### S009｜提交两个独立实现分支

命令：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_train add <7 target files>
git -C /root/autodl-tmp/RLinf_idea2_dvac_train commit -m 'feat: add DVAC-weighted pi0 GRPO training'
git -C /root/autodl-tmp/idea2_dvac_train_wamppo add <4 RoboTwin target files>
git -C /root/autodl-tmp/idea2_dvac_train_wamppo commit -m 'feat: add sampled RoboTwin control trace'
```

结果：RLinf=`052a2ee8902c51595caa997736f1ec76699ad8df`，worktree clean；RoboTwin/wamppo=
`43696bbab85fef3dd98074c5ba0ccb90786d0e94`，仅保留预期未跟踪`RoboTwin_RLinf/assets`符号链接。
没有把16G assets、运行产物或凭据加入Git。

### L006/S010｜准备纯观察资源采样与2-step启动器

本地用`apply_patch`新建：

- `tmp/idea2_dvac_train_observe_resources.sh`；
- `tmp/idea2_dvac_train_run_smoke.sh`。

上传后服务器执行`bash -n`均通过。observer dummy probe让`sleep 5`充当driver，自然得到3个时间样本/
6个GPU rows，`resources.csv`、`process_rss.tsv`、`observer_exit.txt`均非空。observer只读取
`nvidia-smi`、`/proc`、cgroup、`df`和`ps`，每2秒采样；没有阈值、告警动作、timeout、signal、kill或
exit-code联动。

启动前核对：两卡`0 MiB/0%`、无GPU compute process，cgroup current=`3,130,310,656` bytes、
`high/max/oom/oom_kill=0`，目标run目录不存在。YAML/observer/launcher的本地与服务器SHA256分别一致：

- YAML `b913b2ba4a400f732e6fde3e99a73934b04dc03321b260db5d4bd4a0ca46312e`；
- observer `f048f044d2d16a2aae106abd73b6617ac83676892eb1263bddae374c31156cdc`；
- launcher `c483cefb89ecf71fb7660f292af329ec68ac79ad54cd3bfc09218c84609678e8`。

精确训练命令由launcher写入`launch_command.txt`：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf_idea2_dvac_train/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_idea2_dvac_train/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_train_2step_smoke
```

环境固定`CUDA_VISIBLE_DEVICES=0,1`、`ROBOT_PLATFORM=ALOHA`、`MUJOCO_GL/PYOPENGL_PLATFORM=egl`，
`PYTHONPATH`只指向两个新worktree。输出：
`/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_smoke_2step_2gpu16env_20260820`；runtime/log/observer：
`/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_smoke_2step_2gpu16env_20260820`。

23:09左右以detached wrapper启动，wrapper PID=`45632`、driver PID=`45635`。首次轮询Ray已正常在2卡节点
启动，GPU仍仅4 MiB初始化占用；smoke继续自然运行。

### S011｜第1个runner step完成并进入第2步

运行中只读轮询`driver.log`、`nvidia-smi`、observer CSV和产物目录。两组Env/Rollout/Actor按两卡placement
初始化；日志出现`curobo.types.math`可选导入traceback，但当前resolved配置使用`planner_backend=mplib`，
driver继续运行且历史成功RLT/GRPO环境也有同类可选后端打印，因此没有改环境或重启。

第1步8个rollout wave用时约12分52秒，随后两次update epoch完成并自然进入第2步。rank CSV共同记录：

- `runner_step=0, mode=apply, warmup=1`；
- history count=`0`，current pooled count=`25,600=512 queries×50 h`；
- current pooled `log V_L3` mean=`-4.4432252688`、std=`0.6690434629`；
- 所有weight严格为`1.0`；
- loss-mask有效query为rank0 `181`、rank1 `105`，但recent统计不按该mask筛；
- actor grad norm=`46.7489853`、PPO clip fraction=`0.1936209`、approx KL=`0.0878863`、
  ratio=`1.0013819`；
- 两个rank各写一个约189 KB `rollout_step0000.npz`、CSV、manifest与rolling state。

因此“第1步普通GRPO照常更新并建立全部trajectory action-query统计”的合同已实测成立。

同一阶段control trace自然完成：只生成一套`worker_000/env_slot_000/recording_0000...reset_57`，
`finish_reason=success`、111帧，CSV 111行，MP4 H.264/yuv420p 160×120/10 FPS/11.1秒/15,011 bytes，
`encoder_error=null`。第2步环境offload/recreate后MP4计数仍为1，persistent claim生效。

### S012｜第2步apply完成，driver和observer自然退出

运行中与结束后只读命令：

```bash
tail -n 160 <runtime>/driver.log
cat <runtime>/driver.exitcode <runtime>/observer.exitcode
cat <runtime>/launch_started_at.txt <runtime>/launch_finished_at.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
find <run>/dvac_train -maxdepth 2 -type f -printf '%P %s\n' | sort
find <run> -maxdepth 5 -type d -name global_step_2
du -sh <run>
```

结果：23:09:32启动，23:38:00自然结束，wall time 28分28秒；driver、wrapper与observer rc均为0。
第二步两个rank共同记录：

- `warmup=0`，history=`1 step / 25,600 values`，mean/std=`-4.443225/0.669043`；
- 当前step 512 queries、25,600个`query×h`，mean/std=`-4.411428/0.639711`；
- weight min/max=`0.8/1.2`；rank0有效query p05/mean/p50/p95=`0.8736/1.0197/1.0189/1.1766`，
  rank1=`0.8705/1.0173/1.0151/1.1779`；
- pre-clip grad norm=`40.9796`，PPO clip fraction=`0.10404`，approx KL=`0.04059`；
- 当步on-policy `success_once=0.7734375`，只作smoke事实，不当fixed-ID评估；
- `global_step_2`存在，run约9.7 GiB；两卡回到0 MiB，无残留GPU compute process。

### S013｜机器后检与纯观察资源汇总

命令：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python <runtime>/postcheck.py <run>
/root/autodl-tmp/RLinf/.venv/bin/python <runtime>/summarize_resources.py \
  <runtime>/resource_monitor <runtime>/resource_summary.json
```

结果分别打印`IDEA2_TRAIN_SMOKE_VALIDATION_PASS=1`与`RESOURCE_SUMMARY_PASS=1`：

- 每step两rank共512 query；`weights/V/meta/done` shape与finite检查通过；step1全1、step2非均匀且有界；
- 4个NPZ共931,890 bytes，checkpoint、manifest、rolling state与唯一录像齐全；
- GPU0/1峰值29,774/29,490 MiB；cgroup峰值82.034 GiB，anon/file峰值68.591/12.427 GiB；
- host最低MemAvailable约909.65 GiB，`/dev/shm`最低约119.99 GiB；
- EnvWorker RSS峰值27.398/20.558 GiB；actor约15.54/14.38 GiB；rollout约15.61/14.38 GiB；
- 773个时间样本、1,546个GPU rows；cgroup `high/max/oom/oom_kill`全部0→0。

observer只采样，没有阈值、告警动作、timeout、signal、kill或exit-code联动。

### S014｜离线分析与contact sheet；三次命令错误均未改变run

第一次分析错误使用不存在的`/root/autodl-tmp/conda/envs/rlinf/bin/python`，rc=127；改为已验证golden
venv后，第二次又把脚本位置误写成worktree下`runtime/analyze_smoke.py`，rc=2。找到实际runtime脚本后，
第三次误按`--run-dir/--output-dir`命名参数调用，而脚本合同是两个位置参数，因此它把既有run目录当成
output并在`mkdir(exist_ok=False)`处安全失败，rc=1；没有覆盖或写入run。最终命令：

```bash
/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717/bin/python \
  <runtime>/analyze_smoke.py <run> \
  /root/autodl-tmp/idea2_dvac_train_analysis/smoke_2step_20260820
```

打印`IDEA2_TRAIN_SMOKE_ANALYSIS_PASS=1`。512个第二步query的完整权重统计为min/median/mean/max
`0.8/0.99614/1.00281/1.2`，大于1/小于1比例`48.33%/51.67%`；前/后25个h均值
`0.99531/1.01031`。

contact sheet首次在交互式Python中因引号转义得到本地`SyntaxError`，没有发出远程ffmpeg命令；改用
三引号命令字符串后成功：

```bash
ffmpeg -y -v error -i <control-mp4> \
  -vf "select='not(mod(n\,10))',scale=320:240,tile=4x3:padding=4:margin=4" \
  -frames:v 1 <analysis>/CONTROL_TRACE_CONTACT_SHEET.png
```

输出1300×736 PNG。后检把reset57 q2执行前的`V_L3(h)`与首次success近似`h≈10`对齐；该点
`V_L3=0.010621`。这是单条control trace的机制示意，不作为阶段或性能统计结论。

### L007/S015｜SFTP下载轻量证据并核对hash

通过同一已认证Paramiko Transport的SFTP下载26个文件，共3,286,715 bytes，到：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/training_smoke_2step_20260820/
```

包括resolved config、launch/log、postcheck、4个rank-step NPZ、rank CSV/manifest/state、resource CSV/TSV/
summary、MP4/frames/metadata、两张图和逐h CSV；不下载9.7 GiB checkpoint或完整Ray目录。PowerShell
`Get-FileHash -Algorithm SHA256`逐文件完成；关键hash与完整清单写入该目录README/SHA256SUMS。

### S016｜有界push两个已验证实现分支

启动前核对两个worktree：RLinf tracked clean；outer仅有预期的untracked `RoboTwin_RLinf/assets`链接。
命令：

```bash
timeout 60s git -C /root/autodl-tmp/RLinf_idea2_dvac_train \
  push -u personal codex/idea2-dvac-train-weighting
timeout 60s git -C /root/autodl-tmp/idea2_dvac_train_wamppo \
  push -u origin codex/idea2-dvac-control-trace
git -C <worktree> rev-parse HEAD @{upstream}
git -C <worktree> status --short --branch
```

结果：两次push rc=0；本地HEAD与upstream分别同为
`052a2ee8902c51595caa997736f1ec76699ad8df`和`43696bbab85fef3dd98074c5ba0ccb90786d0e94`。
没有force push、持久化proxy、删除文件或提交assets。

### L008｜结果文档与当前停点

用`apply_patch`新增`06_TRAINING_IMPLEMENTATION_AND_SMOKE_RESULT.md`，并同步`00_INDEX_AND_PLAN.md`、
`05_TRAINING_MODIFICATION_PLAN.md`、`HANDOFF.md`和轻量证据README。当前明确停止在正式训练前：
30/50/100-step均未启动；下一步只讨论正式预算、是否需要recent-stats checkpoint恢复，以及完整packet。

## 100-step正式修改版训练

### L009/F001｜用户授权与正式配置最小增量

2026-08-21用户明确选择直接运行100 steps，并授权立即启动；要求除DVAC相关新增项和必要的独立输出外，
其余参数对齐历史成功GRPO。启动后只确认正常开始即可退出观察，不长期盯守。

本地用`apply_patch`新建：

- `robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal.yaml`；
- `tmp/idea2_dvac_train_run_formal_100step.sh`。

相对已通过smoke只改变：`max_steps 2→100`、`rollout_epoch 8→16`、独立formal run/name；保留
2×A800、16 env、G8、B512/mb32、update2、H=C50/D14/M4、flow_sde、chunk reward/logprob、
lr5.6e-6、clip_grad1、save10、fresh SFT、DVAC公式与单条抽样control trace。正式run目标：

```text
/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
```

当前录像选择仍为整个不间断driver只录`worker0/slot0`第一条episode，最多200帧；smoke实测111帧三件套
约28.7 KB，因此100步也不会乘25,600条trajectory，预计仍只是几十KB。DVAC NPZ按smoke线性外推约
91 MB；主要空间仍是每10步checkpoint。

### F002｜正式100-step训练启动并完成最小健康确认

上传文件SHA256：formal YAML `bbed31cf...d8125`、launcher `f01d083e...0864`、复用只读observer
`f048f044...56cdc`。formal YAML已提交并推送为RLinf commit
`145fa810`（`personal/codex/idea2-dvac-train-weighting`）。

启动命令：

```bash
nohup bash <formal-runtime>/run_formal_100step.sh \
  > <formal-runtime>/wrapper.log 2>&1 < /dev/null &
```

启动后事实：wrapper PID `114146`、driver `114149`、observer `114150`；两组actor、rollout和env worker
均已创建，模型/norm stats开始加载，两卡各约22,889 MiB，cgroup约38.9 GB，memory events全0。
日志中的可选Curobo导入traceback与smoke相同；当前实际planner为mplib，进程继续运行。

权威路径：

```text
run:     /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime: /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
log:     <runtime>/driver.log
```

按用户要求，到此退出持续观察；没有等待第1步完成、没有设置外置停止条件或控制训练行为。

### F003｜10:03只读现场检查与轻量快照下载

用户随后要求查看当前训练、方法信号、资源和全部主要产物。用原有Paramiko低层密码链路执行只读脚本
`tmp/idea2_formal_live_audit_20260821.sh`，主要命令为：

```bash
date --iso-8601=seconds
ps -p 114146,114149,114150 -o pid,stat,etimes,cmd
tail -n 180 <run>/metrics.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
find <run> -maxdepth 5 -type d -name 'global_step_*'
du -sh <run>
```

2026-08-21 10:03现场wrapper/driver/observer均在；检查开始时最新UI表为Step22，下载过程中Step23完成。
服务器已有step10/20 checkpoint各约9.7 GiB，run约20 GiB；两卡显存约25–29 GiB，cgroup约173 GiB，
memory events全0。日志错误扫描只命中已知optional Curobo import traceback；实际planner为mplib，进程继续。

用`tmp/download_idea2_formal_snapshot_20260821.py`经同一Transport的SFTP下载metrics、resolved config、
两rank Step0–22 DVAC artifacts、唯一control trace与截至10:05的资源记录，到：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step22_20260821/
```

首次本地调用因脚本没有把workspace root加入`sys.path`而无法import `local_scripts`；只用`apply_patch`
补入root后重跑成功，共下载63个原始文件、58,186,739 bytes。该问题发生在本地下载器，未影响服务器run。

### F004｜录像contact sheet与三类离线分析

服务器只读/派生命令：

```bash
ffmpeg -y -v error -i <control-trace>/head_camera.mp4 \
  -vf "select='not(mod(n\,10))',scale=320:240,tile=5x4:padding=4:margin=4" \
  -frames:v 1 <analysis>/CONTROL_TRACE_CONTACT_SHEET.png
```

输出20格contact sheet；原录像为reset57、200帧、step limit、未成功。随后仅在Windows快照上运行三类
CPU分析脚本：`build_training_analysis.py`、`analyze_dvac_snapshot.py`、`analyze_resources.py`，产出：

- `TRAIN_METRICS.csv/png`与`BASELINE_COMPARISON.csv/png`；
- `DVAC_STEP_SUMMARY.csv`、`DVAC_BY_H.csv`与method overview PNG；
- `RESOURCE_SUMMARY.json/png`及与历史GRPO同elapsed的resource CSV/PNG。

主要结果：完整快照为UI Global Step23=`runner_step22`；最近10步rollout success与历史GRPO相差
+0.156个百分点，近10步反而快约6.4秒/step。最新有效query weight p05/median/mean/p95为
0.890/1.019/1.026/1.200；future-h与median V_L3 Spearman为0.895。GPU峰值30.37/30.22 GiB；
cgroup峰182.69/240 GiB、events全0。与旧GRPO各自起点后9.86小时比较，cgroup增长只差-1.68 GiB，
EnvWorker合计峰值只差-0.80 GiB，未见DVAC带来明显主机内存增量。

### F005｜10:23最终只读刷新

先调用系统`python.exe`失败，因为WindowsApps占位程序在sandbox用户下不可执行；改用Codex bundled Python
运行同一Paramiko helper，没有安装依赖。远程只读脚本`tmp/idea2_formal_latest_status_20260821.sh`执行：

```bash
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' <run>/metrics.log | tail -n 3
tail -n 55 <run>/metrics.log
ps -p 114146,114149,114150 -o pid=,stat=,etimes=,cmd=
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
```

结果：Step24/100已完成，三个进程仍在，`driver.exitcode`和`launch_finished_at`仍不存在。g24
success_once=78.5156%、KL=0.045、clip=0.129、grad norm=33.519、ratio=1.002、step=1512.2s；日志ETA
31:35:00。即时GPU为27.1/27.0 GiB，cgroup约166 GiB，events仍全0。所有动作均为只读观察与本地文档/
派生图写入；没有改变、暂停或控制训练。

### F006｜16:31 Step39只读刷新、增量下载与强度反事实

用户要求再次查看训练曲线、与历史GRPO同轴比较，并解释梯度缩放幅度。继续复用Codex bundled Python、
临时Paramiko runtime与同一低层Transport；远程只读命令与F003/F005同类：

```bash
date --iso-8601=seconds
ps -p 114146,114149,114150 -o pid,stat,etimes,cmd
tail -n 180 <run>/metrics.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
find <run>/dvac_train -maxdepth 2 -type f | sort
```

结果：2026-08-21 16:31:49三个进程仍在，Global Step39/100完整结束；g39
success_once=89.8438%、KL=0.054、clip fraction=0.241、grad norm=27.365、ratio=1.007，实际step
1469.0秒。即时两卡显存约25.98/25.87 GiB，cgroup约187.6 GiB，`low/high/max/oom/oom_kill`仍全0。

本地先复制Step0–22已有轻量快照，再用`tmp/download_idea2_formal_increment_step39.py`经SFTP只补下载
metrics、resource记录和两rank runner step23–38的DVAC NPZ/CSV，形成：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step39_20260821/
```

随后只在Windows CPU运行：

```text
analysis/analyze_dvac_snapshot.py
analysis/analyze_step39_training_curves.py
analysis/counterfactual_gradient_weighting.py
```

生成历史GRPO同轴图、DVAC method overview、逐step/逐h CSV、梯度语义说明和`a=.1/.2/.3`/hard-top20
反事实。g1–39 rollout success为85.53% vs历史87.46%；g1尚未apply已先相差-6.25个百分点，因此当前
结论是“效果尚不明显”，不能把约-1.93个百分点直接归因于DVAC。当前权重ESS proxy为0.992；`a=.2/.3`
为0.970/0.938，hard top-20%为0.2。所有操作均为远程只读、本地派生分析和文档更新；没有修改运行中
config、代码、进程或产物。

### F007｜17:15 Step41只读刷新、训练数据流与相关工作复核

用户要求逐图解释g39反事实/方法诊断图，并从reward、GRPO advantage、PPO ratio clip到global gradient
clip重新教学，同时复核近期强降权/增权方法的实际幅度。先用`apply_patch`新建只读命令文件
`tmp/idea2_formal_status_refresh_20260821.sh`，再经Codex bundled Python、临时Paramiko runtime与原固定
host-key/password Transport执行：

```bash
hostname
pwd
id -u
date --iso-8601=seconds
ps -p 114146,114149,114150 -o pid=,stat=,etimes=,cmd=
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' <run>/metrics.log | tail -n 3
tail -n 65 <run>/metrics.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
```

身份探针为`autodl-container-nekaqbwt43-6ce5babb`、`/root`、uid0。2026-08-21 17:15:43现场三个进程
仍在，最近完整表为Global Step41/100。g40/g41训练rollout success为91.4063%/93.3594%；g41
KL=0.069、PPO clip fraction=0.139、pre-clip grad norm=30.726、ratio=0.965、step=1476.2秒，
DVAC current/history mean为-4.102/-4.104、rank-local weight mean=1.026。即时GPU0/1为
26,255/25,892 MiB，cgroup为211,764,916,224 bytes（约197.2 GiB），`low/high/max/oom/oom_kill`全0；
driver仍无exitcode/finished marker。

随后只读核对历史/当前resolved config与源码：两份run均为`only_eval:false`、
`val_check_interval:-1`，故success曲线均为on-policy训练rollout而非held-out eval；OpenPI路径实际使用
joint chunk PPO ratio clip `[0.8,1.2]`与最后`clip_grad=1`，配置中的`clip_ratio_c=3`不进入π0 loss。
DVAC straight-through位于`[B,H,D]`log-prob求和前，固定forward时不改变ratio/clip gate；被PPO gate
裁平的query之50个h一起为0，未裁query再按per-h weight重排，最后global clip只消除公共尺度、保留方向。

一手来源复核覆盖NeurIPS 2025 Beyond 80/20、ACL 2026 STEER/A3PO/HTMR、Findings ACL 2026 LESS与
ICLR 2026 ResT。新增
`09_PPO_DATAFLOW_CLIPPING_AND_REWEIGHTING_DISCUSSION_20260821.md`，逐项记录图1–5、完整数据流、两层
clip、相关工作实际幅度和`a=.2`解释；同步更新专题索引与根`HANDOFF.md`。远程动作仍只有只读检查，
没有修改formal config、代码、进程或服务器产物。

### F008｜20:21 Step48只读刷新、最新方法shard与位置残差讨论

用户要求继续查看当前训练、解释`V→z→w→gradient`的具体数据，并把chunk位置效应、状态效应、
recent-5、straight-through及PAR/HTMR/SSVPO/BPGO等方法依据整理清楚。本次服务器操作仍只读。

先以`apply_patch`新建/复用两个本地只读命令文件：

```text
tmp/idea2_formal_status_refresh_20260821.sh
tmp/idea2_formal_quick_status_20260821.sh
```

再由Codex bundled Python、临时Paramiko runtime、固定host key与process-only密码执行：

```powershell
<CODEX_BUNDLED_PYTHON> local_scripts/verified_password_ssh.py run `
  --command-file tmp/idea2_formal_status_refresh_20260821.sh

<CODEX_BUNDLED_PYTHON> local_scripts/verified_password_ssh.py run `
  --command-file tmp/idea2_formal_quick_status_20260821.sh
```

远程脚本中的核心只读命令为：

```bash
hostname
pwd
id -u
date --iso-8601=seconds
ps -p 114146,114149,114150 -o pid=,stat=,etimes=,cmd=
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' <run>/metrics.log | tail -n 3
tail -n 65 <run>/metrics.log
find <run>/checkpoints -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 6
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
test -f <runtime>/driver.exitcode && cat <runtime>/driver.exitcode
test -f <runtime>/launch_finished_at.txt && cat <runtime>/launch_finished_at.txt
```

第一次长输出会话在结果基本返回后仍未自然交还本地终端；只中断该本地SSH读取会话，没有向远程训练
process group发送任何signal。随后用窄版脚本复核现场，正常返回。`memory.peak`在该cgroup不存在，资源峰值
继续使用既有旁路`resources.csv`，不因此增加新的监控行为。

身份探针仍为`autodl-container-nekaqbwt43-6ce5babb`、`/root`、uid0。2026-08-21 20:21:40现场：

- 最新完整日志表为`Global Step 48/100`；driver进程仍在，`driver.exitcode`和finished marker均不存在；
- 即时GPU0/1为`26967/28384 MiB`，总显存各`81920 MiB`；
- cgroup current为`214537658368 bytes`，约`199.8 GiB`；
- `low/high/max/oom/oom_kill`全0。

随后经同一SFTP通路只下载小文件：

```text
<run>/metrics.log
<run>/dvac_train/actor_rank00/rollout_step0047.npz
<run>/dvac_train/actor_rank01/rollout_step0047.npz
```

落到：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step48_20260821/run/
```

复用此前已核验的本地CPU训练曲线脚本，生成：

```text
analysis/TRAIN_METRICS.csv
analysis/TRAIN_METRICS.png
analysis/BASELINE_COMPARISON.csv
analysis/BASELINE_COMPARISON.png
analysis/TRAIN_METRICS_SUMMARY.json
analysis/METHOD_STEP48_SUMMARY.json
```

两张PNG均已目视后检。Step48结果：

- g48训练rollout success为`85.5469%`，与历史g48相同；g1–48均值为`86.1898% vs 87.9964%`，
  差`-1.8066pp`，仍未形成稳定领先；两者均无训练中held-out eval；
- g1–48平均KL为`0.04583 vs 0.04773`，PPO clip fraction为`0.14685 vs 0.14979`，
  pre-clip grad norm为`34.014 vs 30.757`；
- 最新293个loss-mask有效query、14,650个`h`权重点中，p05/median/p95为
  `0.88665/1.01570/1.20000`；命中0.8/1.2为`0.109%/8.130%`；
- negative/positive advantage位置的平均权重为`1.04877/1.01231`；后25个h比前25个h平均高`0.03705`；
- Step48 current/recent-5 `log V_L3` mean/std为`-4.009/0.530`与`-4.035/0.521`。

完整资源CSV截至20:03覆盖19.895小时：cgroup start/end/peak为`12.79/197.52/207.35 GiB`，GPU0/1
峰值`30.37/30.22 GiB`，OOM事件全0。与历史GRPO相同elapsed比较，各自起点后的cgroup peak增长
`194.56 vs 194.15 GiB`，主机内存轨迹近似；当前GPU峰值约多1 GiB。

本地新增/更新：

```text
10_ACTION_WEIGHT_POSITION_RESIDUAL_AND_RECENT_CREDIT_LITERATURE_20260821.md
11_FORMAL_TRAINING_LIVE_REFRESH_STEP48_AND_METHOD_DISCUSSION_20260821.md
00_INDEX_AND_PLAN.md
HANDOFF.md
```

其中10号文档把分析先收敛成用户原本的两部分：50点位置曲线`b_h`与同位置残差`R(q,h)`；
`S_q/I_qh`仅作为以后进一步拆分第二部分的可选层。11号文档记录Step48训练、方法和资源事实。
本轮未改变服务器代码、配置、训练进程或产物。

## R-only 位置残差 × 偏重降权 `[0.5,1.2]` 开发分支

### L010｜用户冻结新分支目标与本轮授权

2026-08-21用户选择偏重降权范围`[0.5,1.2]`，并明确授权开始开发新的训练修改分支。本轮实施目标收敛为：

- 使用`y(q,h)=log(V_L3(q,h)+eps)`；
- 最近5个已完成runner step按每个future位置`h`分别建立`median/MAD`基线；
- 训练权重只使用去位置后的`R(q,h)`，不把固定位置曲线`b_h`混入online weight；
- 对`R<0`与`R>0`使用不同斜率，使`R clip [-2,2]`最终映射到`[0.5,1.2]`；
- 保持现有straight-through挂点、chunk reward/logprob、joint PPO ratio/clip、advantage、global grad clip、
  rollout/batch/optimizer与环境语义不变；
- 从当前已验证训练实现派生独立child branch/worktree，完成少量服务器前测后停在真实smoke之前。

本轮尚未授权停止当前formal、运行新smoke或启动新正式训练。先用只读现场检查确定当前run与source authority，
再开始分支写入。

### L011｜刷新现场并建立唯一新 child worktree

2026-08-21 21:50以固定host key、process-only密码的Paramiko通路执行身份与现场探针。结果：

- host=`autodl-container-nekaqbwt43-6ce5babb`，cwd=`/root`，uid=`0`；
- 旧v1 formal的wrapper/driver/observer仍在运行，现场已完成`52/100`；
- 它实际读取的source为`/root/autodl-tmp/RLinf_idea2_dvac_train`，branch=
  `codex/idea2-dvac-train-weighting`，HEAD=`145fa810f1d8baee23012922b81e496661d61cf5`；
- 两卡即时显存约25.5/25.7 GiB，cgroup current约211.7 GiB，memory events中的
  `high/max/oom/oom_kill`均为0。

旧run仍占用旧source，因此只建立一个新的RLinf child；RoboTwin/control-trace代码没有算法变化，继续复用
既有outer worktree，不再新建第二个outer分支。命令：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_train worktree add \
  -b codex/idea2-dvac-residual-downweight \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight \
  145fa810f1d8baee23012922b81e496661d61cf5
```

新worktree从与旧formal完全相同的已验证实现提交派生；未修改旧worktree、旧run、checkpoint或进程。

### L012｜实现per-h位置残差与偏重降权映射

本机窄同步目录：

```text
tmp/idea2_residual_downweight_impl_source/rlinf/
```

服务器目标：

```text
/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
```

只修改/新增四个项目文件：

1. `rlinf/algorithms/dvac_train_weighting.py`
   - 新增recent-5、per-h的`median/MAD`历史统计；
   - current step先使用此前history，训练成功后才push；
   - 冻结映射为`u=clip(R,-2,2)`，负侧斜率0.25、正侧斜率0.10，因此
     `u=-2,-1,0,1,2 -> w=0.5,0.75,1,1.1,1.2`；
   - writer schema v2保存每步raw V、position center/MAD/scale、residual与weight；旧global-zscore writer仍保留。
2. `rlinf/workers/actor/fsdp_actor_worker.py`
   - 新增`signal_mode=per_h_robust_residual`参数分支；
   - 两个actor rank用一次`all_gather`把各自`[query,h]`的log-V拼成同一global step矩阵，再各自维护相同history；
   - 只把本rank的`[T,B,H]`weight送入现有shuffle/replay；原straight-through挂点、chunk求和、joint PPO
     ratio/clip与global grad clip不变；
   - 显式校验`weight_mapping=asymmetric_linear`，避免配置名被静默忽略。
3. `tests/unit_tests/test_dvac_train_weighting.py`
   - 增加位置基线消除、非对称映射、robust outlier、recent窗口淘汰与保留h维度测试；
   - 保留旧global-zscore、trajectory shuffle和straight-through前向恒等/反向倍率测试。
4. 新建
   `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke.yaml`。
   它完整沿用已跑通2-step配置的任务、SFT、2卡16 env、G8、B512/mb32、update2、flow-SDE、
   chunk reward/logprob、优化器、FSDP、offload和control trace；只换独立run路径及上述DVAC参数块。

Windows仅做`py_compile`；随后通过同一SFTP一次上传6个窄文件，共124,370 bytes。旧2-step与100-step YAML
也同步进child作为不变对照，没有覆盖旧worktree文件。

### L013｜服务器简洁前测、两个脚本问题与窄修复

服务器核心命令：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile <3 changed Python files>
PYTHONPATH=<new RLinf child>:<existing RoboTwin child> \
  /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_dvac_train_weighting.py \
  tests/unit_tests/test_dvac_telemetry.py

PYTHONPATH=<new RLinf child> \
  /root/autodl-tmp/RLinf/.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /root/autodl-tmp/idea2_residual_downweight_two_rank_probe_20260821.py

<python> train_embodied_agent.py --config-path <new config dir> \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke \
  --cfg job --resolve
git -C <new child> diff --check
```

结果：

- `11 passed, 3 warnings`；warnings为既有SWIG/opentelemetry依赖提示；
- 双进程probe输出
  `TWO_RANK_PER_H_OK queries=4 center=[2.0,12.0] scale=[2.9652,2.9652]`，证明两rank得到相同统计；
- 新R-only config与仓库内一个不含DVAC块的official RoboTwin config均成功完整resolve；default-off resolved中
  不含`dvac_gradient_weighting/dvac_train/return_dvac`；
- 新resolved与已跑通旧DVAC 2-step resolved逐行比较，差异仅为独立run/output路径，以及
  `global zscore + [0.8,1.2]`替换成`per-h median/MAD residual + [0.5,1.2]`；其余参数零差异；
- `git diff --check`通过。

两次前测脚本问题均未触及实现：第一次compose漏设历史必需的`REPO_PATH/EMBODIED_PATH`，补回后新配置
成功resolve；第二次把历史baseline副本名当成child内现成配置，改用仓库真实存在的
`robotwin_adjust_bottle_ppo_openpi`做default-off compose。没有安装依赖、启动Ray、加载checkpoint、运行
RoboTwin、启动smoke或新增训练进程。

### L014｜提交完成；远端推送窗口超时

只stage L012列出的四个项目文件并提交：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight add -- <4 target files>
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight commit \
  -m 'feat: add residual DVAC action weighting'
```

结果：commit=`49b83791c114a90552b7ad7059e675d35fb8d055`，`4 files changed, 650 insertions(+),
25 deletions(-)`；worktree clean。随后两次通过HTTPS remote `personal`做非交互、有界60秒push，均以
`rc=124`超时且没有远端确认；因此当前权威状态是**服务器child已commit、尚未确认push成功、branch无upstream**。
没有继续增加网络重试，也不影响本机服务器上的后续smoke候选。

### L015｜用户批准R-only 2-step smoke

2026-08-21用户明确要求启动smoke，并要求并发结构与此前成功smoke接近。本次批准packet为：

- source=`/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight@49b83791…`；
- RoboTwin/control trace复用`/root/autodl-tmp/idea2_dvac_train_wamppo`；
- 2×A800，placement `env,rollout:0-1`，`total_num_envs=16`，每卡8 env；
- train `rollout_epoch=8`，`max_steps=2`，G8，B512/mb32，update epoch2；
- 共256 trajectories、最多1,024个进入actor/loss的action-query、4次optimizer update；
- step1 `w=1`并建立per-h history，step2应用`[0.5,1.2]`；
- 独立output/runtime目录；资源observer每2秒只读记录GPU/RAM/process，不设阈值或发送signal。

启动前先只读刷新旧v1 formal与GPU/RAM。若旧formal仍占用两卡和大部分cgroup内存，不与新smoke叠跑；
停止旧formal不在本条授权中。

### L016｜22:32启动前现场：旧formal仍占用资源，未叠跑smoke

只读身份/资源探针结果：

- 旧v1 wrapper/driver/observer PID仍为`114146/114149/114150`，已完整记录`Global Step 53/100`；
- GPU0/1即时显存=`28,985/30,944 MiB`，GPU0 util 100%；
- cgroup current=`234,308,689,920 bytes`，约218.2 GiB/240 GiB；
- `memory.events`的`high/max/oom/oom_kill`仍全0，数据盘可用766G；
- 新child仍为`49b83791…`且worktree clean。

此前2-step smoke单独运行的cgroup峰值约82.03 GiB；与当前218.2 GiB叠加会超过容器240 GiB额度。
因此未启动新Ray/driver/observer，也没有停止、暂停或改变旧formal。等待用户明确选择停止旧formal后启动smoke，
或让旧formal继续完成后再运行。

### L017｜用户授权停止旧formal、整理轻量终态包并启动R-only smoke

用户明确选择：中断旧v1 formal，按此前讨论整理主要日志、指标、资源、方法与少量raw shard的轻量包，
然后启动已批准的R-only 2-step smoke。执行顺序冻结为：

1. 核对旧driver身份与最近完整save10 checkpoint；只向旧driver发送一次`SIGINT`；
2. 等wrapper/observer自然收尾，确认旧run不再占GPU/RAM；
3. 保留服务器checkpoint，轻量包不复制约9.7 GiB DCP，也不打包全量NPZ或原始高频资源表；
4. 用新child、独立run/runtime目录启动2卡16-env smoke；确认进入初始化/rollout后回报。

### L018｜旧v1 formal在完整Global Step 54后停止并完成轻量closeout

执行前再次确认旧run的wrapper/driver/observer为`114146/114149/114150`。最初停止脚本把checkpoint目录
少写了一层experiment子目录，因`set -e`在发送signal之前退出；该次没有改变训练进程。随后用只读
`find`定位实际恢复点，并在修正脚本中同时核对driver cmdline和checkpoint：

```bash
find <v1-run> -maxdepth 4 -type d -name 'global_step_*' -printf '%p\n' | sort -V
test "$(tr '\0' ' ' </proc/114149/cmdline)" = <expected-driver-command>
find <.../checkpoints/global_step_50> -type f | wc -l
du -sb <.../checkpoints/global_step_50>
kill -INT 114149
```

结果：最新完整DCP为`global_step_50`，3个文件、`10,393,939,465` bytes、没有`.partial`。`SIGINT`
已精确发送给旧driver，但现场`SigIgn`表明该进程忽略INT，训练继续完成了Global Step 54。重新核对同一
PID/cmdline后，于`2026-08-21T22:42:41+08:00`执行：

```bash
kill -TERM 114149
```

wrapper/driver/observer随后只退出本run；`launch_finished_at=2026-08-21T22:42:45+08:00`，driver
`rc=134`。driver log显示终止发生在未完成的Step 55收集过程中，所以分析只使用完整的Step 1–54；这次
非零退出来自用户授权的主动停止，不是自然训练故障。停止后两卡均回到0 MiB compute占用，cgroup
current约51.7 GiB，`memory.events`的`high/max/oom/oom_kill`仍全0；g50 checkpoint完整保留在服务器。

轻量包先在服务器选取主要材料后一次下载，Windows再补最终曲线、方法摘要、README与文件清单：

```text
exports/idea2_dvac_v1_formal_stop_g54_20260821.zip
4,290,075 bytes；43 entries
SHA256 96df1b432beaafca0f22821d6a6031559b232b5ba4de1db54b4c207ba722e1ae
```

包内只保留配置、最终日志/曲线、历史GRPO同轴对照、rank摘要、Step 1 warmup/Step 2 first-apply/
Step 54 final三组双rank NPZ和一条control trace；排除checkpoint正文、其余51组NPZ、raw资源/RSS、
Ray/TensorBoard/cache/core。ZIP CRC、41行manifest与42项SHA清单均通过。终态训练rollout success为：
latest`88.67%`、recent5`90.31%`、g1–54均值`86.66%`；历史GRPO同窗口均值`88.67%`，差`-2.01pp`。
两者都没有训练中held-out fixed-ID eval。

### L019｜R-only两卡16-env两步smoke启动

先把由旧已验证launcher窄替换得到的两份脚本SFTP到服务器；相对旧smoke只改变R-only source/config、
独立run/runtime路径以及observer的process tag。服务器执行`bash -n`通过，并确认目标run/runtime原先
不存在。启动前现场：新child HEAD=`49b83791c114a90552b7ad7059e675d35fb8d055`且clean；GPU0/1均为
0 MiB/0%；cgroup current=`55,509,528,576` bytes；memory events全0。

精确启动入口：

```bash
nohup setsid bash \
  /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821/launch_smoke.sh \
  > /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821/wrapper.log \
  2>&1 < /dev/null &
```

启动时间=`2026-08-21T23:06:31+08:00`；wrapper/driver/observer PID分别为
`130841/130846/130847`。独立输出：

```text
/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
```

首次刷新时三进程均alive，Ray已建立两组Env/Actor/Rollout workers，TensorBoard与只读resources CSV已
落盘。GPU当时仅约503/504 MiB，cgroup约91.77 GiB，memory events全0。日志中的可选Curobo import
trace与旧成功运行相同；本配置使用`mplib`，driver仍继续初始化。此处只记录“已健康启动”，不提前写
step1/step2结果；自然完成后的权重、资源、checkpoint和录像后检另记L020。

### L020｜Step 1完整完成并进入R-only apply step

通过既有Paramiko路径执行两次只读状态脚本，读取PID、日志、rank-local CSV/NPZ与rolling state；没有向
smoke发送signal。第一次确认Global Step 1完整结束，随后第二个runner step重新进入rollout；稍后刷新时
第二步已到`5/8`。Step 1结果：

- `128 trajectories`，训练rollout `success_once=0.734375`；
- `approx_kl=0.07247`、`clip_fraction=0.22820`、pre-clip `grad_norm=51.9053`，均为finite；
- warmup按合同为1，两个rank的全部weight均为1；
- 两rank均落盘`rollout_step0000.npz`与一行CSV；
- 两rankrolling state完全一致：`history_steps=1`、`history_queries=512`、`history_count=25,600`、H=50，
  并保存逐h center/MAD/scale；因此第二步具备生成R-only非均匀weight的历史；
- 第二步重新进入8轮rollout说明step-to-step的actor更新、history推进和env重新初始化已衔接。它不是外部
  DCP checkpoint reload；真正非均匀weight的actor反传仍需第二步rollout全部收齐后完成。

第二步`5/8`快照时GPU0/1显存约`26,620/26,723 MiB`，cgroup current约121.9 GiB；
`memory.events`的`high/max/oom/oom_kill`仍全0。没有在中途停止；smoke随后于
`2026-08-21T23:35:04+08:00`自然完成，wrapper/driver/observer均退出，driver/observer rc0。

最终只读后检结果：

- Global Step 2/2完整；两rank均落盘`rollout_step0001.npz`，`global_step_2` checkpoint存在；
- step2 `warmup=0`、`history_steps=1`、`history_queries=512`；两rank使用相同per-h center/MAD/scale；
- rank00/01的weight p05约`0.659/0.668`、median约`1.019/1.021`、p95约`1.200/1.193`、
  mean约`0.975/0.981`，min/max均触及`0.5/1.2`且两侧都有样本；
- step2训练rollout success=`0.84375`；`approx_kl=0.01626`、`clip_fraction=0.08516`、
  pre-clip `grad_norm=24.7137`、ratio=`1.00563`，均为finite；
- control trace三件套存在；结束后GPU0/1 compute占用均为0，
  `memory.events`的`high/max/oom/oom_kill`仍全0。

因此R-only已完整覆盖“step1建history→step2逐h residual与非对称weight→straight-through反传→保存”链路。

### L021｜当前R-only分支推送个人远端

先只读执行`git worktree list --porcelain`及目标worktree的`status/HEAD/remote/branch -vv`。Idea2在
RLinf仓内当前保留三个有明确阶段含义的worktree：推理telemetry、v1 global-zscore训练和v2 R-only训练；
正式训练复用v2，不再新增worktree。v2 worktree clean，HEAD为`49b83791…`，但尚无upstream；v1与
RoboTwin control-trace已经各自跟踪个人远端。随后执行：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight \
  push -u personal codex/idea2-dvac-residual-downweight
```

结果：push成功，新远端分支为
`Yutenji-Nyamu/rlinf_fastwam:codex/idea2-dvac-residual-downweight`；本地分支已设置跟踪，HEAD保持
`49b83791c114a90552b7ad7059e675d35fb8d055`且worktree clean。没有推official `origin`，没有新增或删除
worktree，也没有stage外层RoboTwin worktree里既有的untracked runtime assets。

### L022｜用户批准并冻结R-only fresh-SFT 100-step formal执行包

2026-08-22用户明确授权启动修改后的R-only 100-step训练；任务、SFT与历史成功GRPO主体参数继续保持，
只使用已经通过两步smoke的per-h residual和`[0.5,1.2]`权重分支。Windows本机从已跑通R-only 2-step
smoke YAML复制出唯一formal YAML：

```text
tmp/idea2_residual_downweight_impl_source/rlinf/examples/embodiment/config/
  robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal.yaml
```

相对2-step smoke做严格逐行比较，只有四个值变化：

1. `runner.logger.log_path`切到独立20260822 formal run；
2. `runner.logger.experiment_name`切到同名formal experiment；
3. `runner.max_steps: 2 -> 100`；
4. `env.train.rollout_epoch: 8 -> 16`。

其余任务/SFT、2卡16 env、G8、global batch512、micro batch32、update epoch2、flow-SDE、chunk
reward/logprob、优化器、FSDP、PPO ratio/clip、global grad clip、R-only公式和抽样control trace均未改变。
三份本地待上传文件及SHA256为：

```text
40b27febb2d96249a3e0ed4f17e488b6b142be1d76b1b94a95f40217f2a34353  formal YAML
36496ec87f63a398578beccaaae67335b90c8dabafba6392264fd0a77a008b47  launch_formal.sh
c1c2ab69f62be73a1aac6b0dad8ae9bc09d19ffcc5c88203ac450411734b8bc7  observe_resources.sh
```

`2026-08-22T00:09:18+08:00`启动前prescan通过固定host-key Paramiko路线完成。现场事实：source仍为
`49b83791c114a90552b7ad7059e675d35fb8d055`、branch clean且跟踪个人远端；GPU0/1均为
`0/81920 MiB、0%`；cgroup current=`66,030,874,624` bytes，`memory.events`全0；
`/root/autodl-tmp`可用756G；下列run/runtime目标均不存在：

```text
/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
```

先只创建上述独立runtime目录，再以SFTP上传：formal YAML到新RLinf child的config目录，launcher和observer
到该runtime目录。服务器`sha256sum`与上述三个本地hash完全相同，两个shell脚本均通过`bash -n`。
随后用正式source、既有RoboTwin child和同一Python执行完整compose：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal \
  --cfg job --resolve \
  > /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/resolved_config.yaml
```

resolved结果明确包含并通过核对：`max_steps=100`、`rollout_epoch=16`、`total_num_envs=16`、
`group_size=8`、`global_batch_size=512`、`micro_batch_size=32`、`update_epoch=2`、
`signal_mode=per_h_robust_residual`、`weight_min=0.5`、`weight_max=1.2`。resolved文件已保存在上述
runtime路径；本轮没有单独计算它的hash，因此不补写一个推测值。

只stage formal YAML并提交、推送当前既有个人分支：

```bash
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight add -- \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal.yaml
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight commit \
  -m 'config: add R-only DVAC 100-step formal run'
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight \
  push personal codex/idea2-dvac-residual-downweight
```

结果：正式source HEAD=`3061872e30cfb496eb296354d30274d35b66576e`；branch继续跟踪
`personal/codex/idea2-dvac-residual-downweight`，push后worktree clean。没有改旧v1/smoke run、checkpoint、
outer RoboTwin worktree或无关进程。资源observer仍每2秒只读写CSV，不设阈值、timeout或signal。

### L023｜R-only 100-step formal精确启动与首次rollout刷新

正式入口只启动L022冻结的独立launcher：

```bash
test ! -e /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
nohup setsid bash \
  /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/launch_formal.sh \
  > /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/wrapper.log \
  2>&1 < /dev/null &
```

启动时间=`2026-08-22T00:13:16+08:00`；独立run/runtime为：

```text
/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
```

wrapper/driver/observer PID分别为`198255/198259/198260`。`00:16:42`首次只读刷新确认三者均alive，
actor ranks=`201880/201882`、rollout ranks=`201886/201897`、env ranks=`201911/201913`均已生成；driver
日志进入`Generating Rollout Epochs 0/16`，rollout/env进程分别在generate/interact。该时点GPU0/1为
`22,285/22,254 MiB`，cgroup current=`103,735,480,320` bytes，memory events全0；schema2 manifest、
source与control-trace claims已出现，fatal扫描无命中。

`00:18:35`再次只读刷新时，rollout已到`1/16`，该epoch耗时141.78秒；wrapper/driver/observer仍alive，
actor ranks `201880/201882`处于`recv_rollout`，rollout ranks `201886/201897`处于`generate`，env ranks
`201911/201913`处于`interact`。GPU0/1为`21,863/25,720 MiB`，GPU1利用率95%；cgroup current=
`116,019,113,984` bytes，memory events全0；资源CSV已有291行。manifest记录`schema_version=2`、
`source_commit=3061872e30cfb496eb296354d30274d35b66576e`、`source_dirty=false`和R-only formal config；
control-trace claim存在。fatal扫描仍无命中；扫描输出中的`limit_all_gathers: false`只是resolved config文本，
不是错误。

截至该刷新尚无完整Global Step、formal checkpoint或step级权重/效果。这里仅确认获批进程已进入真实
rollout调用链；不能写成step1完成、driver rc0或formal成功。后续完整step与自然终态另行逐时点追加。

### L024｜R-only formal Global Step 23现场快照、产物清点与离线图

2026-08-22 09:54起，继续复用固定host-key、process-only密码的Paramiko helper，只读执行：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  local_scripts/verified_password_ssh.py run `
  --command-file tmp/idea2_r_only_formal_current_audit_20260822.sh
```

结果：wrapper/driver/observer=`198255/198259/198260`仍alive；最新完整Global Step为23/100，
`global_step_10`和`global_step_20`存在；run约20 GiB、`dvac_train`约24 MiB、control trace约40 KiB、
runtime约35 MiB。两rank各有23份`rollout_step0000–0022.npz`。fatal扫描只命中启动阶段已知的可选
Curobo import traceback，driver继续运行，没有CUDA OOM或worker fatal。

第一次直接对仍由observer追加的`resources.csv`打tar时，tar报告`file changed as we read it`。这是读取
增长中文件的快照一致性问题，训练未受影响。窄修复是不触碰原文件，先把四个增长中文件复制成固定快照，
再打包：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  local_scripts/verified_password_ssh.py run `
  --command-file tmp/idea2_r_only_formal_make_snapshot_g23_20260822.sh
```

服务器生成
`current_snapshot_g23_20260822.tar.gz`，随后通过同一helper的SFTP get下载并在Windows解压到
`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/r_only_formal_live_g23_20260822/`。压缩包
2,721,062 bytes、25 entries；checkpoint正文未下载。快照包含固定版metrics/driver/resources/process-RSS、
resolved config、launch信息、双rank manifest/CSV/rolling state/latest NPZ和一条control trace。

09:59:55另执行：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  local_scripts/verified_password_ssh.py run `
  --command-file tmp/idea2_r_only_formal_resource_limits_20260822.sh
```

现场`memory.current=233,413,758,976` bytes、`memory.max=257,698,037,760` bytes（240 GiB）、swap=0；
`low/high/max/oom/oom_kill`全0；`/root/autodl-tmp`约737 GiB可用。资源CSV快照峰值为226.62 GiB，
末值217.56 GiB；两卡峰值显存30.368/30.218 GiB。两个EnvWorker合计RSS峰121.73 GiB，是主要进程来源。

Windows只做CPU日志/NPZ/CSV离线计算，生成：

```text
analysis/TRAINING_VS_GRPO_G23.png/.csv
analysis/DVAC_R_ONLY_DIAGNOSTICS_G23.png
analysis/RESOURCE_OVERVIEW_G23.png
analysis/ANALYSIS_SUMMARY_G23.json
analysis/RESOURCE_SUMMARY.json
analysis/TRAIN_METRICS_G23.csv
analysis/DVAC_STEP_METRICS_G23.csv
analysis/LATEST_HORIZON_G23.csv
analysis/analyze_r_only_g23.py
analysis/analyze_resources_g23.py
```

后检：三张主图已目视核验；CSV/JSON数值与原metrics、双rank NPZ和资源表交叉核对。g1–23训练rollout
success均值82.52%，历史成功GRPO同step轴84.97%，当前没有稳定领先；R-only g2–23 mean weight为
0.9995，p05/median/p95为0.715/1.031/1.198。g23原始`log V`后半减前半为+0.191，per-h位置center为
+0.141，residual只剩+0.023，最终weight只剩+0.0034，符合R-only去固定位置趋势的设计。资源observer
继续只读记录，本轮未发送signal、未改变参数、未停止或影响训练。

### L025｜三run同轴图、credit机制分析与checkpoint评估只读规划

本轮先完整重读根规则、`HANDOFF.md`与专题`00_INDEX_AND_PLAN.md`，随后只读检查g23本地轻量包、历史GRPO
分析、v1 closeout ZIP与v2逐step CSV。根Git状态首次直接执行：

```powershell
git status --short
git branch --show-current
git rev-parse --show-toplevel
```

因sandbox账号与目录owner不同被Git`dubious ownership`拒绝。未改global Git设置；改用单命令
`git -c safe.directory='C:/Users/86136/Documents/rl' status --short`后成功，确认根目录大量既有未跟踪
worktree/专题材料均未清理或stage。

先新建三run绘图脚本
`evidence/three_run_training_comparison_20260822/build_three_run_comparison.py`。第一次使用默认本地Python运行时
报`ModuleNotFoundError: No module named 'matplotlib'`；数据读取尚未开始。没有安装新依赖，改为使用Codex
bundled Python已有的Pillow与NumPy后复跑成功，生成：

```text
THREE_RUN_SUCCESS_COMPARISON_G23.png
THREE_RUN_OPTIMIZATION_COMPARISON_G23.png
THREE_RUN_METRICS_G23.csv
THREE_RUN_SUMMARY_G23.json
```

两张PNG均已目视后检。数据源是历史GRPO `analysis.json`、v1 g54 closeout ZIP内完整训练CSV、v2 g23
训练CSV；没有从不同长度的全程均值推导最终方法排名。

随后用bundled Python只读打开g23双rank `rollout_step0022.npz`，逐项核对key、shape与dtype，并新建：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/r_only_formal_live_g23_20260822/analysis/analyze_credit_chain_g23.py
```

脚本自然完成，生成`DVAC_CREDIT_CHAIN_G23.png`、`CREDIT_CHAIN_G23.csv/.json`并通过目视后检。g23的
loss-valid集合为70条成功trajectory的210个query与34条失败trajectory的136个query；相对uniform
credit，正adv质量为0.990倍、负adv绝对质量为1.029倍，top-20%绝对credit占比从40.24%变为42.33%。
这些明确标为query级PPO gate与global clip前的系数空间诊断。

checkpoint清单与评估代码语义通过固定host-key Paramiko helper只读执行以下command-file；密码只注入当次
进程，未写入脚本/账本：

```text
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_inventory_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_code_audit_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_run_path_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_entry_audit_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_convert_audit_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_live_refresh_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_seed_audit_20260822.sh
verified_password_ssh.py run --command-file tmp/idea2_checkpoint_eval_source_hashes_20260822.sh
```

`live_refresh`第一次把checkpoint路径少写一层experiment name，因此该小段为空；没有修改服务器。清单已由
前一个inventory和最后的source-hashes脚本按正确nested路径复核：原GRPO只剩g100，v1有g10–g50，v2
当时有g10/g20，SFT锚点存在。DCP仅含两份distcp shard，没有`full_weights.pt`；后续需使用仓内
`convert_dcp_to_pt.py`逐份转换后评估。

2026-08-22 10:51:28最后只读刷新：v2最新完整Global Step 25/100；wrapper/driver/observer
`198255/198259/198260`均alive；GPU0/1为26.62/26.91 GiB，cgroup为223.55/240 GiB，memory events与
fatal扫描均无命中，磁盘余737 GiB。本轮没有转换checkpoint、创建服务器评估目录、启动评估、发送signal、
停止训练或清理临时权重。

设计与结果集中写入
`13_METHOD_CAUSAL_CHAIN_CHECKPOINT_EVAL_AND_THREE_RUN_COMPARISON_20260822.md`；专题索引和根handoff只增加
当前路由与10:51刷新点，没有把专题细节复制到根文档。

### L026｜R-only formal g35只读刷新、信号演化、control-trace短文与机制教学

本轮先按根规则完整读取`PROJECT_CONTEXT.md`、`HANDOFF.md`和当前专题`00_INDEX_AND_PLAN.md`，授权边界为
只读刷新、离线分析和本地文档整理；没有停止/修改训练或启动checkpoint转换/评估。

复用固定host-key、process-only密码的Paramiko helper。第一次尝试：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_r_only_formal_live_refresh_20260822.sh
```

Windows默认`python.exe`是不可执行的商店占位符，命令没有连到服务器。改用Codex bundled Python后身份探针、
host-key和密码认证通过，并只读返回进程、step、metrics、GPU/RAM/disk、产物和fatal扫描：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_r_only_formal_live_refresh_20260822.sh
```

首次checkpoint glob少了一层experiment-name目录，因此checkpoint小段为空，其他现场结果有效。窄修复为另一个
只读command-file使用`find ... -name global_step_*`精确定位，同时读取traceback上下文与cgroup extrema：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_r_only_formal_live_refresh_detail_20260822.sh
```

2026-08-22 15:01:25 +08:00现场事实：最新完整g35/100，探针结束时Step36 rollout `8/16`；
wrapper/driver/observer `198255/198259/198260`均alive；g10/g20/g30 DCP各约9.7 GiB。GPU即时约
27.1/27.5 GiB、CSV峰值30.37/30.22 GiB；cgroup即时约223.3/240 GiB、CSV峰值237.41 GiB，
`low/high/max/oom/oom_kill`全0；主机可用RAM约834 GiB，数据盘余约727 GiB。run/runtime约30 GiB/
93 MiB，rank-local NPZ 70份。fatal扫描只命中启动期可选Curobo import traceback，后续连续完成35步。

通过同一helper的SFTP `get`子命令分别拉取10个小文件：metrics、driver、resolved config、resource CSV、
双rank runner CSV/rolling state/g35 NPZ；未复制checkpoint。bundled Python运行：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/r_only_formal_live_g35_20260822/analyze_live_g35.py
```

第一次脚本把workspace parent写成`parents[4]`，找不到复用绘图脚本；用`apply_patch`改为`parents[3]`后成功。
生成并目视后检：

```text
evidence/r_only_formal_live_g35_20260822/analysis/TRAIN_AND_METHOD_G35.png
evidence/r_only_formal_live_g35_20260822/analysis/RESOURCES_G35.png
evidence/r_only_formal_live_g35_20260822/analysis/SUMMARY_G35.json
evidence/r_only_formal_live_g35_20260822/analysis/{TRAIN,DVAC_RANK,DVAC_STEP,LATEST_HORIZON}_G35.csv
```

g1–35 training-rollout success为85.27%，历史GRPO同轴86.75%；最近10步91.02% vs 90.70%。g2–35
weight p05/median/p95均值为0.708/1.030/1.199，正/负advantage mean weight为0.988/1.019；当前仍无
held-out eval。

随后只读使用v1 g54 runner CSV/closeout材料、v2 g35 runner CSV与latest双rank NPZ，bundled Python运行：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/signal_evolution_v1_v2_20260822/build_signal_evolution.py
```

生成并目视核对`DVAC_SIGNAL_EVOLUTION_V1_G54_V2_G35.png`、逐step CSV与JSON。v1 raw mean
`ln V_L3`从-4.4128到-3.9470，v2从-4.4236到-4.02095；这是on-policy模型与访问状态共同变化，未作单因
归因。g35 clipped residual方差约39.8%来自query-wide `S_q`、60.2%来自query内local `I_qh`。

最后全部用`apply_patch`新建/更新：

- `14_ROBOTWIN_CONTROL_TRACE_IMPLEMENTATION_NOTE_20260822.md`：双仓调用、逐control-frame取帧、配置、
  产物、TOPP progress近似h与direct evaluator差异；
- `15_R_ONLY_G35_SIGNAL_CREDIT_AND_MECHANISM_CHAIN_20260822.md`：g35现场、credit/ESS/80/20教学、五层
  机制链的论文直接先例与项目适配、v1/v2信号演化与强度判断；
- `00_INDEX_AND_PLAN.md`：只更新当前动态状态、阶段N与14/15号路由；
- 根`HANDOFF.md`：只更新Idea2路由行和15:01快照，不改变当前深圳主线。

远端操作始终只读；资源observer仍只记录、不控制训练。本轮未发送signal、改参数、创建评估目录、转换
checkpoint、删除或覆盖服务器产物。
