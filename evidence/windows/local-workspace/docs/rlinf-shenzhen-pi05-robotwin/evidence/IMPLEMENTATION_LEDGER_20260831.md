# 深圳 π0.5 RoboTwin 实施与 smoke 流水

## 0. 授权与边界

- 日期：2026-08-31。
- 用户授权：完成前置工作；最小 smoke 官方 π0.5 PPO；派生并 smoke π0.5 GRPO；实现并 smoke π0.5 GRPO-DVAC Action-Adv Fix `[0.5,1.5]`；记录资源并与 π0 经验比较。
- 比较主线：优先保证同一 π0.5 模型内部的 clean/method 配置关系；GRPO-DVAC 必须逐叶继承 clean π0.5 GRPO，除方法字段、命名、placement和run-scoped路径外不得改变采样、batch或update预算。
- 执行风格：每层只跑覆盖真实路径的最小 smoke；成功即继续，真实故障只做一个有依据的窄修复。
- 共享服务器边界：先只读刷新GPU、RAM、磁盘、现有RLinf任务和Git状态；不停止或修改其他用户与现有正式训练。

## 1. 已锁定的候选预算

- π0.5 PPO：两卡，64 train / 32 eval，rollout4，GB512/MB32/update5，M5；formal-size one-step smoke。
- π0.5 clean GRPO：两卡，64 train / 32 eval，rollout4，G8，GB512/MB32/update5，M5；逐叶继承同模型PPO资源/优化外壳，只改变GRPO所必需的group-relative、actor-only、group filter等算法字段。
- π0.5 GRPO-DVAC：从clean GRPO复制，Action-Adv Fix `[0.5,1.5]`、L3/recent5；formal-size two-step smoke。

## 2. 流水

### 2.1 本地只读准备

- 完整读取根目录`PROJECT_CONTEXT.md`、`HANDOFF.md`与π0.5专题计划。
- 复核官方π0/π0.5 RoboTwin PPO与model preset：π0为M4/update2，π0.5为M5/update5；两者共用OpenPI/RoboTwin环境合同。
- 下一操作：通过固定host-key、进程内密码Paramiko路线只读刷新深圳现场。

### 2.2 深圳现场只读预检

- 时间：服务器`2026-08-31T05:05:25+00:00`（北京时间约13:05）。
- 身份：`chenyiteng`，固定host-key密码路线正常；shared Ray位于`172.17.0.1:6389`且健康。
- GPU 4/5正在运行π0 PPO Control，GPU 6/7正在运行π0 PPO-DVAC；均不改动。GPU 0--3空闲，π0.5 smoke候选使用物理GPU 2/3。
- 主机：约2.0 TiB RAM、约1.3 TiB available；`/data`与`/home`各约1.4 TiB可用，满足约8.5 GB模型与最小smoke。
- 可复用环境：`/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`、RoboTwin RLinf-support、shared Ray均存在；尚无π0.5模型副本。
- current Action-Adv工作树HEAD为`74617ced87d64045ab6850d0efd90956a494af66`且clean；新工作在独立branch/worktree进行，不修改现有正式训练工作树。

### 2.3 同模型比较口径修正

- 用户明确：首要比较是同一π0.5模型内部PPO、clean GRPO与GRPO-DVAC，而不是先追求与π0的优化预算相同。
- 因此撤回clean GRPO的`GB1024/update2`候选，改为与π0.5 PPO共同使用`GB512/MB32/update5/M5`。两者均为1024 records上限、每rank batch 256、梯度累积8、每轮10次optimizer step。
- clean GRPO相对PPO只保留算法必要差异：G1→G8、GAE→group-relative、actor-critic→actor-only、关闭value head、启用group filter；无效叶不为“看起来像旧GRPO”而额外修改。
- DVAC `[0.5,1.5]`再逐叶继承clean GRPO；只增加action-level advantage、DVAC端点、L3/recent5、sidecar及run命名。

### 2.4 独立工作树与实现

- 本地从服务器等价source superset `74617ced...`建立`codex/sz-pi05-robotwin-rl`；服务器建立独立工作树`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl`，未修改GPU 4--7正式训练的工作树。
- 源码审计确认π0/π0.5共用OpenPI sampler、typed rollout、DVAC `[B,H]`和Action-Adv H-sum路径；M5下L3合法，不需要修改production Python。
- 初版尝试用一份52行Hydra薄配置继承官方π0.5 PPO；真实compose报错：被继承配置含`hydra.searchpath`，Hydra只允许primary config设置该字段。
- 窄修复：遵循RLinf现有主配置风格，改为一份完整`robotwin_adjust_bottle_grpo_openpi_pi05.yaml`；内含inactive DVAC block。方法smoke在同一配置上只覆盖action-level、mode/application和`[0.5,1.5]`，不再增加第二份重复YAML。
- 新增总量：1份配置、183行；production Python零改动。服务器最终commit并推送：`codex/sz-pi05-robotwin-rl@256eeeb4459b4bd5db85bfc6a0eb315771e8c38c`。

### 2.5 模型与resolved packet

- 官方模型锁定：`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98`，官方页面总量约8.53 GB、三份safetensors并含RoboTwin norm stats。
- 下载于服务器`2026-08-31T05:21:17+00:00`开始，目标`/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed`；使用既有Mihomo/HF CLI、单worker和HF自身完整性处理，不增加手工SHA检查。
- 三份v2 packet已真实Hydra compose并通过共享合同断言：GPU2/3、64 train/32 eval、rollout4、256 trajectories、最多1024 records、GB512/MB32/update5、每轮10次optimizer step、M5、offload、local-shard。
- PPO：GAE/actor-critic/value head，1 outer step；clean GRPO：G8/group-relative/actor-only/filter，1 outer step；DVAC：逐叶继承clean GRPO，仅action-level与`[0.5,1.5]`方法叶不同，2 outer steps。
- 下一操作：模型完整落盘后按上述顺序占用GPU2/3 smoke；每段记录30秒GPU/RAM采样、墙钟、finite指标和实际local-shard，GPU4--7任务不动。

### 2.6 下载路由窄修复

- 官方endpoint首轮在4.23 GB shard约2.75 GB处发生`ChunkedEncodingError/IncompleteRead`；HF local-dir保留了约2.6 GiB可续传partial。
- 相同命令第一次续传在metadata请求处再次出现`SSL EOF`。四路短探针结果：官方endpoint代理与直连均失败；`hf-mirror.com`代理和直连均HTTP 200，直连约0.54秒。
- 因此只把`HF_ENDPOINT`改为`https://hf-mirror.com`并取消代理；repo、精确revision、local-dir和partial均不变，不重下已有内容、不增加校验流程。

### 2.7 π0.5 PPO 一步 smoke

- 模型于服务器`2026-08-31T05:59:11+00:00`完整落盘，`du -sh`约8.0 GiB；三份shard、index和checkpoint自带RoboTwin norm stats均存在。
- PPO于`06:07:15--06:37:27 UTC`完成真实Step 1并自然`exit 0`，墙钟30分12秒。训练成功率`0.83203125`，actor KL `0.066`、clip fraction `0.237`、grad norm `23.619`；critic value loss `0.323`、explained variance `-0.113`，均为有限值。
- GPU2/3采样峰值分别为`60479/60513 MiB`（约59.1 GiB/卡）；主机最低`MemAvailable=904785712 KiB`（约862.9 GiB），GPU4--7原正式训练始终保留。
- checkpoint实际完整存在于`$run/pi05_ppo_smoke1/checkpoints/global_step_1/actor/`：rank0/rank1 local shard各约10.16 GB，另有约8.53 GB full weights。

### 2.8 段间验收路径窄修复

- 首版串行总控把checkpoint误查为`$run/checkpoints/...`，漏掉logger自动增加的`$experiment/`目录和`actor/`层；因此PPO本体成功后，总控在验收语句退出，未进入clean GRPO。
- 这是launcher验收路径错误，不是训练、数值或checkpoint保存故障；不重跑PPO、不改训练参数。
- 修复后验收精确指向`$run/$experiment/checkpoints/global_step_N/actor/local_shard_checkpoint/`，并于`06:43:55 UTC`只续跑clean GRPO → DVAC两段；续链为`smoke-chain-grpo-continuation-20260831-v2`。

### 2.9 π0.5 clean GRPO 一步 smoke

- clean GRPO于`06:43:55--07:15:08 UTC`完成Step 1并自然`exit 0`，墙钟31分13秒；成功率`0.8008`、grad norm `13.738`、KL `0.366`、clip fraction `0.383`，均为有限值。
- GPU2/3峰值`60159/60705 MiB`（约58.8/59.3 GiB/卡）；主机最低`MemAvailable=793201912 KiB`（约756.5 GiB）。
- local-shard checkpoint实际落盘并通过修正后的两rank断言；续链随后自动进入DVAC `[0.5,1.5]`两步smoke。

### 2.10 π0.5 GRPO-DVAC 两步 smoke 与终态

- DVAC于`07:15:23--08:07:13 UTC`完成两个真实outer step并自然`exit 0`，总墙钟51分50秒；Step1按设计为history warm-up，权重全1。
- Step2成功率`0.8398`、KL `0.096`、clip fraction `0.027`、grad norm `15.390`，均为有限值。
- Step2 `warmup=0`、weight mean `1.042`、sq mean `1.134`、ESS `0.957`、high/low clip fraction `0.062/0.0000195`；证明M5 endpoint→L3→recent history→`[0.5,1.5]` Action-Adv完整路径实际生效。
- GPU2/3峰值`63349/62873 MiB`（约61.9/61.4 GiB/卡）；主机最低`MemAvailable=634790008 KiB`（约605.4 GiB）。
- `global_step_2`含rank0/rank1 local shard、full weights及两份688-byte DVAC sidecar；续链打印`PI05_GRPO_SMOKES_OK`。三段均无fatal/OOM，结束后GPU2/3释放，GPU4--7原训练未被停止。

### 2.11 与π0资源基线及当前停点

- 历史两卡π0 clean GRPO首步约21.75分钟、早期GPU峰约49.2 GiB/卡；本次π0.5 clean为31.22分钟、约59.3 GiB/卡。
- 差异同时包含M5对M4、GB512/update5对GB1024/update2以及Step1保存开销，不能写成纯模型速度差。
- smoke已经使用约74%--78%的80G显存；当前不扩大每卡env或micro batch。formal候选共同保留`64/32/rollout4/GB512/MB32/update5/M5/fixed32/eval5/save10`，等待用户讨论，不自动启动。
