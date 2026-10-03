# GPU4 本 job 图形限制（2026-10-03，v4相机已验，真实归还待验）

## v4 当前合同（替代下文 v1–v3 历史方案）

服务器实际 Ray 没有独立或 bundled setproctitle 包；真实入口是 `ray._raylet.setproctitle`。v4 bootstrap `import ray`（不init、不连接服务），只包装这一实际入口，保留原 argv 标题后恢复私有主线程 comm。诊断/相机 probe 均调用同一真实入口，分别检查主线程和后台线程。旧版本缺包失败都发生于 GPU 初始化前。

fragment 显式注入 `pwd.getpwuid(uid)` 的 HOME/USER/LOGNAME；manifest schema3固定该 home。**之前不能排除 HOME 问题：**原 RLT 环境被完整替换后 HOME/USER 缺失，v1 搜索用户 profile 是否受影响仍未证实。

RLinf 基点的 accelerator getter 只接受整数，故 CVD 保持数字 `4`，绝不改成UUID；无mask的driver/CPU manager仍无mask。renderer/probe实际查询唯一可见 CUDA 设备的 UUID 和 PCI，与GPU4 manifest比较。回执新增 start_ticks，拒绝复用PID旧证据。

使用新 `scope_tools_v4` 和 `gpu4-scope-v4`，保留旧scope全部hash和失败回执。目录使用canonical路径，不能使用root-owned的 `/data` symlink。根任务已核旧marker无活进程后精确归档旧profile，helper本身不删除旧文件。

```bash
P=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
S=/home/nvme/team-data/chenyiteng/projects/opendw-robotwin-smoke-20261003
$P -B $S/scope_tools_v4/gpu_scope_prepare.py prepare --output $S/gpu4-scope-v4 --token opendw20261003v4
$P -B $S/scope_tools_v4/gpu_scope_diagnose.py --manifest $S/gpu4-scope-v4/scope.json
$P -B $S/scope_tools_v4/gpu_scope_camera_probe.py --manifest $S/gpu4-scope-v4/scope.json --output $S/gpu4-camera-probe-v4 --allow-existing-owned-pids $S/fresh-gpu4-baseline-v4.json
```

先CPU诊断：fresh child应exit0，CVD=4、HOME正确、Ray主/后台线程改名后comm固定且argv保留、ray未初始化、GPU模块为空；然后才三帧相机。probe只操作自己的新子进程，baseline须实时精确刷新，GPU4≥16GiB空闲，全部卡监控该新PID。相机通过仍需真实EnvWorker验收。

本地新cycle-v3引用 `scope_tools_v4/schema3`，只改新归还runtime，冻结服务器cycle-v2保持原样。status核本job namespace/job/PID-start、marker、comm、真实Ray setter/HOME bootstrap回执、数字4及renderer UUID/PCI；证据不足不确认first_round。CPU mock测试在 `tools/test_opendw_cycle_graphics_scope.py`，在服务器执行。

## v1–v3 历史记录（已被上述现场证据纠正，不可作为当前启动说明）

**深圳3 v1 已实测失败，不能用于正式归还。** 根任务部署后，相机探针新 PID1222658 在GPU0产生7MiB图形上下文，同时GPU4产生60MiB C+G；监督立即结束仅该探针，`remaining_contexts=[]`，三位RLT baseline PID/start均未变。尚未产出 `camera.json`。证据：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/opendw-scope-camera-v1.out`。

v1的只读诊断记录了driver595.71.05、profile路径/掩码16、marker映射和SONAME，renderer_bound回执早于GPU0上下文。**当时“HOME正确、已排除相关配置错误”的结论撤回：**后续实查原RLT启动环境缺少HOME/USER，不能用诊断进程的环境替代目标子进程证据。v1失败也不能单独证明驱动不支持DSO。原始诊断证据同目录 `opendw-scope-diagnose-v1.out`。

**v2、v3均在GPU初始化前因错误的setproctitle依赖假设退出，没有新增GPU上下文。** v3曾尝试从 `ray/thirdparty_files` 定位bundled模块；后续服务器实查证明，这台机的Ray没有该独立/bundled包，实际调用的是 `ray._raylet.setproctitle`。因此“v3装载已解决”属于被纠正的中间判断，不能作为有效方案。旧v2/v3 manifest、marker、profile和runtime保留原hash；当前只使用v4实际入口。

v3的设计保留了独有commname规则、主线程 `/proc/self/task/PID/comm` 固定，以及torch/sapien导入前和Scene构造前再次固定；但其拟包装的两个Python setproctitle函数不是服务器实际入口。v4保留这些有效部分，改为包装实际Ray扩展入口。真实Ray EnvWorker仍须验收；相机探针通过不替代该步骤。

目标：供本轮 GPU4 的 RLT 正常归还及独立 native eval 使用。只新增本job唯一commname profile，无空pattern、无全局兜底，不改共享Ray、驱动或冻结owner/cycle。保留预加载DSO作为精确进程标记/装载回执，**v3 NVIDIA规则仅按commname匹配**。根任务确认失败版本marker无活进程并核原hash后，可精确归档该失败profile，再以新目录/新token准备v3；helper本身不删除或覆盖旧profile。

## 文件与启动

把此处三个 Python 文件及 `gpu_scope_bootstrap/sitecustomize.py` 放入服务器同一 tools 目录。以下只是命令模板，未执行：

```bash
python tools/gpu_scope_prepare.py prepare \
  --output /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/gpu4-scope-UNIQUE \
  --token opendw-g4-UNIQUE
python tools/gpu_scope_camera_probe.py \
  --manifest /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/gpu4-scope-UNIQUE/scope.json \
  --output /data/chenyiteng/results/opendw-gpu4-camera-UNIQUE
```

准备只查询NVML/GPU身份并编译无依赖小DSO，不创建CUDA/渲染上下文。可按下面显式baseline例外与本人RLT同卡做三帧探针；没有该选项仍要求GPU4空闲。工具限定GPU4、普通用户、真实主机/UID/UUID/minor，独占创建新文件；碰到已有EGL设备规则会停下让维护者只读审核优先级，不覆盖旧规则。

探针要求 GPU4 全 C/G 表无人，子进程模拟 `ray::EnvWorker` 改名、校验逻辑 CUDA0 的 PCI、用 SAPIEN3.0.1 实际渲染3帧64×64图，检查非空像素；父进程只观察/结束自己创建的子进程，检查全部卡上的该 PID 上下文与退出回收。输出 `rgb.png`、`camera.json`、`report.json`。默认180秒/4GiB小探针上限。它没有 policy、真实任务动作或 Ray 修改，**也不代表真实 Ray EnvWorker 已验证**。

本轮用户明确允许与本人 RLT 共用 GPU4 做小探针：仅额外传 `--allow-existing-owned-pids /absolute/fresh-identities.json` 才启用，文件为 `[{"pid":123,"uid":20001,"start":456}]`；`start` 是 `/proc/PID/stat` 第22字段。必须刚刷新、全部UID20001且GPU4现存PID集合精确相等、剩余显存至少16GiB。探针仍只有一个子进程/三帧，绝不向baseline PID发信号；新PID越出GPU4或有未知新owner进入GPU4就停止自己的探针。报告保留baseline全卡上下文及前后差，旧RLT的GPU0图形占用不计入probe失败。默认没有身份清单时仍要求空卡。

## 唯一接入点

`gpu_scope_prepare.environment_fragment(manifest, inherited_env)` 返回要注入的 LD_PRELOAD、私有 PYTHONPATH、manifest 和 profile-enable。必须同时用于：

1. 新 RLT/native-eval **driver 的 Popen 环境**，在解释器启动前注入。现有 driver 不可运行中补 LD_PRELOAD。
2. 该 driver 的 `ray.init(runtime_env={"env_vars": fragment})`；并核 RLinf 派生 worker 的 runtime env 合并后仍含相同项。若分配逻辑覆盖 PYTHONPATH，必须将私有 bootstrap 目录仍放在最前。不能改共享 Ray 服务环境。

```python
fragment = environment_fragment(scope_json, driver_env)
driver_env.update(fragment)  # before Popen
# In that new driver, preserve existing runtime_env entries:
runtime_env.setdefault("env_vars", {}).update(environment_fragment(scope_json, current_env))
```

**不要向 discovery driver/CPU manager 强塞 CUDA mask。** RLinf 必须仍能发现物理卡编号；其 actor/env/rollout placement 继续明确为物理4。真实worker的 `CUDA_VISIBLE_DEVICES=4` 必须保持数字：旧“bootstrap改成UUID”的方案会与RLinf整数解析器冲突，已撤回。bootstrap只接受数字4或manifest同一UUID，不改写；RLT归还验收要求数字4，并在renderer中实际核CUDA UUID/PCI。CPU driver/manager允许无mask，但若它试图构造原生环境，绑定器会拒绝。

私有bootstrap在GPU库前固定本job唯一comm并拦截本进程Ray改名；import `robotwin.envs.vector_env` 前应用进程内SAPIEN Scene默认 `RenderSystem("cuda:0")`，保持显式传入的systems。此逻辑继承已验证EXPO的SAPIEN3.0.1 API/源码断言，不导入EXPO方法，也不在actor/rollout进程提前导入SAPIEN。如果原生包导入路径或版本不同，先修正断言/接入点，不能跳过。Ray argv/命令行标题保持原样，进程comm改成manifest中的15字节唯一标记；归属应依job/namespace/PID-start和argv判断，不能要求comm仍以ray开头。

启动后必须核：目标 EnvWorker 的 `receipts/*-bootstrap.json` 与 `*-renderer_bound.json`、`/proc/PID/maps` 的 marker、CVD对应UUID、所有新PID的全C/G范围。已有RLT历史GPU0占用单列，不能据此宣称新job越界或全机空闲。相机 probe 通过但上述 receipt 未出现时，实际接入尚未完成。

## 当前边界

- 冻结的 owner/cycle 未改；维护者仍需在原唯一 owner 的正常归还/新 eval 启动边界接入两处 fragment，不得另起并行归还者。
- `offload()` 只关场景，原生 eval 应退出整个独立 job；回收按已有精确 namespace/job/PID-start 规则完成。
- 原有 NVIDIA profile 只读，新增文件唯一；工具不会自动删除 profile/marker，避免仍存活 worker 失去校验依据。
- v4已通过服务器10项借还CPU检查、真实Ray主/后台线程改名检查与3帧SAPIEN相机：新增上下文仅GPU4，退出后无残留、原RLT身份不变。真实EnvWorker归还仍需receipt、首轮进展和全C/G验收；这些相机结果不是OpenDW训练通过。

依据：[NVIDIA DSO匹配/首次加载/按minor掩码](https://download.nvidia.com/XFree86/Linux-x86_64/575.57.08/README/profiles.html)、[NVIDIA Toolkit EGL/Vulkan profile用途](https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/cmd/nvidia-cdi-hook/update-application-profile/update-application-profile.go)、工作区 `docs/methods/expo-ft/GPU4567_BINDING_20261003.md`。驱动与SAPIEN组合差异由实际相机探针裁定，仍保留显式RenderSystem绑定。

## 新 cycle-v3 的归还接线（本地已写，服务器旧cycle-v2不改）

工作区根目录 `tools/opendw_smoke_gpu4_cycle.py` 以已冻结v2的 `3b79f4a7d1fceb14915aa68731009a51084b06706d5e995dc7a858972992c534` 为基点，新增可选 `prepare --graphics-scope-manifest /.../gpu4-scope-v4/scope.json`。当前新cycle-v3的plan固定schema3 manifest、`scope_tools_v4`下两个helper及marker/profile/bootstrap/runtime的hash。prepare只把scope fragment加入**新归还runtime**环境；原运行环境、源码、RLT N8/C50/200动作等训练配置和GPU5–7不改。cycle版本与scope版本是两套编号。

新冻结cycle的 `install_helper` 包装原 `H.driver`：driver已经由归还Popen加载私有bootstrap，再向其 `ray.init` 的job runtime env注入同一fragment；CPU/discovery保持无CUDA mask，GPU workers仍由既有placement选择物理4。status按新namespace/job/PID-start树核全卡C/G、当前maps里的marker、主线程comm、数字CVD4/真实Ray setter/HOME的bootstrap回执及具有真实CUDA UUID/PCI的renderer_bound回执。没有实际继承证据或新PID越出4，`all_first_rounds_verified`保持false；不修改/终止历史GPU0进程。

scope未配置时兼容旧行为；旧服务器cycle-v2与其plan不覆盖。新helper引用scope_tools_v4，已准备的scope依赖应保持冻结。实配RLinf `2f484040` 在提交actor前通过runtime_env传CVD4；服务器Ray2.57.0的 `RuntimeEnvContext.exec_worker` 先更新环境，再exec目标Python，所以最终sitecustomize前已有CVD4。同PID前置setup若有空值回执，最终bootstrap的数字4回执仍可由现有any判定正确匹配。该时序已核源码，实际归还验收由根任务执行。
