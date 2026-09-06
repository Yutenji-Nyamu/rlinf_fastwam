# RLT 单卡双实验 fresh-480 实施与启动账（2026-08-25）

## 1. 用户授权与实验合同

- 用户授权在两张 A800 上各启动一个独立的单卡实验：原始 RLT control 与 teacher-DVAC `[0.5,1.5]`。
- 两组都从 Stage 2 fresh 开始，终点为 cycle 480；不从历史两卡 Stage 2 checkpoint 恢复。
- 两组共同继承历史成功 RLT 的 8 train env、fixed-20 eval、batch、更新、replay、Q/BC、seed、teacher、保存与评估字段；新的单卡 A/B 彼此比较。
- 资源 monitor 只记录，不依据阈值改变训练行为。

## 2. 单卡语义

- `FSDP=no_shard`：两卡时每个 rank 已各持完整 student/Q；单卡不是把两份模型合并到一张卡。
- `global_batch=512`、`micro_batch=128`保持不变；world size 从2变1后，gradient accumulation由2自动变4，optimizer step仍使用512条样本。
- 一个 EnvWorker/rollout worker 分别处理8个train env/4个eval env；两卡时每rank分别处理4/2。
- `warmup_min_size=10,000`与cache/window=`50,000`保持字面配置不变，因此形成新的单卡协议：单卡global replay到10,000即ready，而历史两卡需两个rank分别达到10,000。
- 两卡 Stage 2 checkpoint包含`actor_world_size=2`，不能恢复到单卡；fresh单卡与后续1→1恢复受支持。

## 3. 预定配置差异

共同单卡增量：

```yaml
cluster.component_placement: {actor, env, rollout: 0}
runner.max_steps: 480
runner.resume_dir: null
runner.val_check_interval: 25
runner.save_interval: 25
```

control不启用`algorithm.rlt_dvac`。DVAC臂继承已验证telemetry，并唯一覆盖：

```yaml
algorithm.rlt_dvac.z_clip: 2.0
algorithm.rlt_dvac.strength: 0.25
# w in [0.5, 1.5]
```

物理卡通过每个独立进程的`CUDA_VISIBLE_DEVICES=0`或`1`选择；两个独立Ray runtime都只看见逻辑GPU 0。

## 4. 逐指令记录

后续每条远端compose、上传、commit/push、准备、启动、首轮复核、旧`[0,2]`收尾与归档操作按时间追加在这里。

### 4.1 单卡配置上传

- 指令：通过进程内密码 Paramiko/SFTP `put` 两份新增 YAML 到 `/root/autodl-tmp/RLinf_rlt_teacher_dvac/examples/embodiment/config/`。
- 结果：control 与 DVAC `[0.5,1.5]` 配置均上传成功；未修改运行中的旧实验。

### 4.2 第一次 resolved-config 合同检查

- 指令：执行 `tmp/rlt_single_gpu_dual_config_precheck_commit_20260825.sh`，分别 Hydra compose 两份配置，再以 Python 逐字段比较。
- 结果：两次 compose 均成功；检查脚本在读取 control 的 `rollout.rlt_feature_model.openpi.rlt_dvac_mode` 时出现 `KeyError`，因此未 commit、未 push。
- 原因：control 使用旧路径，关闭 DVAC 的语义是字段不存在；检查器错误地要求该字段必须显式存在且为 `off`。
- 修复：检查改为 `get("rlt_dvac_mode", "off")`；训练代码和 YAML 不需要修改。

### 4.3 格式复测与服务器提交

- 第一次格式复测：resolved-config 合同再次通过，但 `git diff --cached --check`发现两份新 YAML 文件末尾各多一空行；未提交。
- 修复：只删除两个文件末尾的多余空行，重新 SFTP 上传。
- 第二次复测结果：`SINGLE_GPU_AB_CONTRACT_OK`；共同合同为 `H50/M4/C10/D14`、train `8×1`、eval `4×5`、global/micro batch `512/128`、world-size 1 推导 accumulation 4、warmup 10,000、replay 50,000、update_epoch 5、critic:actor `2:1`、max cycle 480。
- DVAC唯一方法字段：`mode=apply, selected_l=3, applied_horizon=10, z_clip=2, strength=0.25`，解析权重范围为`[0.5,1.5]`；control为旧路径/off。
- Git：服务器成功创建 commit `74c71551`（`config(rlt): add paired single GPU 480 runs`）。
- Push问题：`origin`当前指向上游`RLinf/RLinf.git`，GitHub返回403；这是remote写权限问题，不影响服务器提交与运行，随后检查是否存在已有可写remote。

### 4.4 远端同步修正

- 只读检查：`origin=https://github.com/RLinf/RLinf.git`；已有可写`personal=https://github.com/Yutenji-Nyamu/rlinf_fastwam.git`，本分支相对`personal` ahead 1。
- 指令：`git push personal codex/rlt-teacher-dvac-weighting`。
- 结果：成功把`a85b101b..74c71551`推送到个人远端；服务器worktree保持clean。

### 4.5 双单卡运行包与Ray隔离检查

- 新建并上传四个窄脚本：统一`run_one.sh`、双实验只读`paired_resource_monitor.sh`、旧实验精简归档`archive_old.sh`、自然收尾后启动的`queue_after_old.sh`。
- 服务器：四个脚本均通过`bash -n`；记录逐文件SHA256。
- 第一次Ray检查误用了当前版本不存在的`utils.get_ray_temp_dir`，运行包语法检查已成功，只有该只读探针失败。
- 随后读取已安装Ray 2.55.1源码：`address="auto"`会扫描本机全部GCS，独立`RAY_TMPDIR`仍可能让第二条driver连入第一条runtime；`address="local"`明确表示即使已有本地Ray也新建实例。
- 修复：两条训练均设置`RAY_ADDRESS=local`，同时各自设置独立`RAY_TMPDIR/TMPDIR`和`CUDA_VISIBLE_DEVICES=0/1`；复测四脚本`bash -n`通过。

### 4.6 自然收尾队列启动

- 启动前服务器现场：旧`[0,2]`已完成Global Step 446/480，GPU显存约18.6/19.0 GiB，cgroup current约121.6 GiB，`memory.events`的high/max/oom/oom_kill全为0，磁盘可用720 GiB。
- 指令：`setsid bash .../queue_after_old.sh >.../queue.log 2>&1 < /dev/null &`。
- 结果：队列PID/PGID `196122`，启动时间`2026-08-25T14:28:32+08:00`。队列仅等待旧wrapper PID自然退出；成功退出后先生成精简归档，再并行启动两条新训练。

### 4.7 双Ray并发合同修正

- 深读Ray 2.55.1源码及现场raylet命令后确认：仅`RAY_ADDRESS=local + 独立temp`仍会让两个隐式runtime竞争固定dashboard-agent端口52365；默认还会让每套Ray分别按整机声明CPU和约71.6 GiB Plasma。
- 现场只读`ray memory --stats-only`：旧训练采样边界的Plasma实际为`0 MiB / 0 objects`；`/dev/shm=120 GiB`，主机`nproc=144`，旧双卡Ray声明CPU 36。
- 修正后的窄合同：先为control/DVAC各启动独立external Ray head，再让driver连接具体GCS地址。control使用46001--46009与worker 46100--46599；DVAC使用47001--47009与worker 47100--47599。每套显式`GPU=1, CPU=18, object_store=24 GiB`，独立temp/spill目录。
- 启动时由只读Python连接探针分别断言每套cluster为`GPU=1, CPU=18`；driver仍各自设置物理`CUDA_VISIBLE_DEVICES=0/1`，配置内均使用逻辑GPU 0。
- 正常训练退出时只向自身Ray head PGID发INT，不调用会全机扫描的`ray stop`。
- 因原等待器已打开旧脚本，先尝试INT未能退出其`sleep`；随后仅对等待器PGID使用TERM，保留`queue_v1.log/pid`，未触碰旧训练PID/PGID。
- 新等待队列PID/PGID：`257181`，启动`2026-08-25T14:51:29+08:00`；旧训练wrapper `106844`继续正常运行。

### 4.8 旧实验自然完成、取消新实验与归档

- 旧teacher-DVAC `[0,2]`于`2026-08-25T15:48:56+08:00`自然完成`480/480`，退出码0；GPU0/1随后均归零，
  cgroup `high/max/oom/oom_kill`仍全为0。
- 用户随后指示新实验先不放。现场确认等待队列已退出，未出现pair-launch标记、两条新run进程或Ray head；
  因而没有启动control或`[0.5,1.5]`实验。
- 第一次精简归档因错误假设`checkpoints`直接位于run root而失败；实际目录位于compose实验名子目录。只修正
  checkpoint目录发现逻辑后重跑，服务器raw归档约11 MiB并通过SHA256校验。
- 本地最终包为`exports/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_20260825_v2.zip`，12,982,235 bytes，
  SHA256 `9c49aed8049c08e8cff093df3782ac5220143ecbb87f78720803aaf90872cf61`；包含日志、resolved配置、
  TensorBoard、DVAC telemetry、资源CSV、checkpoint清单、分析CSV/图，不含checkpoint正文、视频和完整Ray日志。
