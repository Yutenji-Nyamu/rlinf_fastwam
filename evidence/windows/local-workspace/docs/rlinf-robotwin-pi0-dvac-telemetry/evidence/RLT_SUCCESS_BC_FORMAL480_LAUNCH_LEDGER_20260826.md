# RLT success-episode DVAC-BC 双单卡正式480步启动流水账

日期：2026-08-25--26  
范围：仅记录正式配对训练的前置核对、启动、一次失败定位与健康启动；不记录后续训练效果。

## 1. 只读前置

远端身份探针：`hostname; pwd; id -u`。结果：AutoDL目标容器、`/root`、UID 0。

现场命令：Git HEAD/dirty、`nvidia-smi`、`memory.current/events`、`MemAvailable`、目标路径存在性和相关
Ray/训练进程查询。结果：HEAD=`64f2779f...`且clean；两卡0 MiB；OOM事件0；v1启动前目标均不存在。

resolved审计：历史成功RLT、单卡control、方法版逐叶比较。冻结正式合同为480 cycles、eval/save每25、
8 train env、20 fixed eval、B512/MB128、历史warmup/replay/update schedule；方法版只增加批准的
success-episode DVAC-BC字段与独立路径/placement。

## 2. 正式v1

上传并执行：

```text
autodl_launch_rlt_success_bc_dual_single_gpu_formal480_20260825.sh
```

结果：shared Ray正常，control/method placement分别为GPU0/GPU1，但两条EnvWorker都在训练前退出：

```text
ModuleNotFoundError: No module named 'robotwin'
```

原因：新launcher只设`PYTHONPATH=$repo`，漏了v9成功smoke使用的
`/root/autodl-tmp/RoboTwin_RLinf`。这不是密码、算法、模型、显存或正式YAML问题。v1目录原样保留，
未复用、未删除。

## 3. 正式v2窄修与启动

本地用`apply_patch`建立三份v2运行脚本；远端操作依次为：

```text
remote_exec_autodl.py put <driver> /tmp/<driver>
remote_exec_autodl.py put <ray-head> /tmp/<ray-head>
remote_exec_autodl.py put <launcher> /tmp/<launcher>
chmod +x /tmp/<three-scripts>
bash -n /tmp/<three-scripts>
remote_exec_autodl.py run --command-file <v2-preflight>
remote_exec_autodl.py run --command-file <v2-launcher>
```

窄修只恢复v9已经成功验证的运行环境：`PYTHONPATH=$repo:$assets`、RoboTwin/Stage1/norm环境变量、短
Ray/TMP目录、shared head看见GPU0/1、自动namespace、编译线程1和120秒错峰。没有改Python算法、正式
YAML或训练预算。

启动返回：shared Ray=`172.17.0.9:52001`；control wrapper PID=`586602`，GPU0；方法wrapper
PID=`586603`，GPU1。

## 4. 启动后只读确认

执行两次窄健康查询：wrapper存活、placement、日志进度、GPU、cgroup events和resolved关键字段。

00:06 CST结果：

- 两条wrapper alive，无exit code；
- placement=`[[0]]/[[1]]`；
- control已进入`Generating Rollout`，方法模型也已装载并进入rollout初始化；
- GPU0/1显存约14,659/21,194 MiB；
- `memory.high/max/oom/oom_kill=0`；
- 方法resolved=`max_steps 480`、eval/save 25、train/eval=`8/(4×5)`、正式warmup/replay schedule、
  `application=success_episode_bc`、`selected_l=3`、`z_clip=2`、`strength=.25`、`success_scale=1`。

到此结束主动盯守，后台训练继续。
