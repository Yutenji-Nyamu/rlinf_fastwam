# Fast-WAM scene fence 窄修、短测、推送与256轨迹重启账本

最终状态（16:47）：修正版`62526cc9`已推送，Env-only RTLD_LOCAL，权重读取+384图回归通过；v3已正常进入采样，首步进度1/8，无完整更新。`b60144fd`的v2全局预加载导致权重读取失败退出，已取消该接入方式，完整过程在下方按顺序保留。原生fence代码/so hash未改；当前合同/边界见§4。

## 1. 授权与不变量

2026-09-04用户明确授权：在Fast-WAM分支干净修复此前定位的scene fence缺口，简要测试、推送，重启当前每轮256轨迹训练，刷新/可视化训练与整机状态。

只修svulkan2 timeline render遗漏的scene-access fence；在Fast分支保存可复现patch/构建接入/回归，不改共享venv原文件。不升级SAPIEN/svulkan2/OIDN，不加GIL/GC/线程模型猜测性补丁，不恢复旧Python生命周期补丁。

保留32env×rollout8=256、G8、GB1024/MB2/update_epoch2（4optimizer calls/step）、LR5e-6、noise0.3、H32/C24/M10、horizon192、fixed32/eval5/save10、原SFT fresh100；旧卡住run保留原目录，不混接指标。实际重启点须先检查checkpoint。

只允许停止核实属于本Fast停滞run的进程组/精确Ray actors；Sidney、shared Ray、其他用户不动。新库在隔离路径构建/加载；测试失败先定位，不自动调参或循环重启。

## 2. 逐操作记录

| CST | 操作/目标 | 结果/边界 |
|---|---|---|
| 15:47–15:49 | 完整读取根规则、window handoff、Fast SSOT、scene fence因果报告；检查现有launch/transfer/health工具 | 完成；尚无服务器写入 |
| 15:50 | `sz_scene_fence_preflight_20260904.sh` + 既有health脚本经SSH stdin | Fast完整1步、无checkpoint、日志仍14:04；Sidney49；三worktree clean，磁盘余1.3/1.4TiB；[现场](SCENE_FENCE_PREFLIGHT_20260904.txt) |
| 15:52–15:57 | wheel headers/RPATH、构建工具、RLinf worker环境传播检查 | wheel带svulkan2/GLM/Vulkan头文件；原生库DT_RPATH优先级要求显式加载，不能只设LD_LIBRARY_PATH；RLinf会传播新增job环境变量。两个读取脚本因pipefail+head提前结束，缺失段已定向补读，无生产影响 |
| 15:57 | 在Fast当前分支新建`tools/fastwam_scene_fence`，上传4个新文件 | 源码diff只为scene fence；构建/运行采用job-local LD_PRELOAD替换同一原生函数，避免整套渲染/降噪依赖重建。共享wheel不变；[传播/阶段证据](SCENE_FENCE_STAGE_20260904.txt) |
| 15:58–16:02 | `bash tools/fastwam_scene_fence/build.sh VENV /home/chenyiteng/builds/fastwam-scene-fence-20260904/release` | 编译成功；补充export map收紧为唯一目标函数，锁原库SHA256、C++11 ABI与CUDA-interoperability布局。发布so SHA256=`45e6cac35ea2edea0724fa4cd92ec82cab401b6abd6f8001805d22306f375df0`；[构建+完整短测配置](SCENE_FENCE_RELEASE_PREPARE_20260904.txt) |
| 16:07 | `sz_scene_fence_stop_owned_fast_20260904.sh` | 精确核实旧wrapper、6个GPU工作进程、RLinf_1的15个named actors后清退成功；GPU6/7空闲，Sidney/shared Ray启动标识不变。[停止记录](SCENE_FENCE_STOP_OWNED_20260904.txt) |
| 16:08–16:12 | 原生加载预检发现`libOpenImageDenoise.so.2`不可解析；读取wheel的`_oidn_tricks.py`与SONAME后窄修构建加载 | 未进入GPU测试。wheel原本由Python按绝对路径预载2.0.1库，没有SONAME软链；在独立build目录建立2个指向原库的链接并仅本job追加LD_LIBRARY_PATH。正式产物改为`release-final`，so hash不变，`/bin/true`预加载检查通过。无共享文件修改。[诊断](SCENE_FENCE_OIDN_LOADER_20260904.txt)、[复建](SCENE_FENCE_LOADER_BUILD_20260904.txt) |
| 16:13–16:15 | 一次真实GPU短测`sz_scene_fence_smoke_20260904.sh` | exit0；2场景×64帧×3相机=384张240×320 RGB，环境/渲染段50.28s，OIDN调用均none。LD_DEBUG确认原libsvulkan2对目标虚函数绑定到补丁库；结束GPU6/7释放。[结果](scene-fence-fix-20260904/smoke-result.json)、[实际绑定](scene-fence-fix-20260904/native-binding.txt)。这是有界回归通过，非全负载长程证明。 |
| 16:16–16:18 | `sz_scene_fence_commit_prepare_20260904.sh` | git whitespace检查将patch文档必须的空行context空格误判；保留有效diff格式，仅对该artifact排除空白检查，其余代码通过。6文件提交`b60144fd60270f303fb5ea229ca92125f6b0a710`并push personal同分支成功，ls-remote SHA一致、worktree clean。compose只6个路径/名称叶子差异，训练参数不变。[提交/逐叶证据](SCENE_FENCE_COMMIT_PREPARE_PASS_20260904.txt) |
| 16:19–16:25 | v2 launch + 两次启动检查 | wrapper1526843成功启动；16:21初检看到双Env加载补丁。终检发现16:21:09已exit255，0step；原SFT的torch zip reader报`Couldn't parse the version`，RLinf自行清退workers，GPU6/7已释放。不能将前一秒的maps核验当作训练健康。[终检](SCENE_FENCE_FINAL_HEALTH_20260904.txt)。先做无GPU的plain/preload/late-load对照定位加载冲突，不重复提交训练。 |
| 16:26–16:27 | CPU-only archive reader对照 | plain与torch先加载的late-load均读出`version=3`、5records；只有global preload失败，Python zipfile三组均正常。确认加载方式导致失败，不是权重损坏。原svulkan2导出mz_zip/CRC等符号，具体冲突符号未逐个拦截证明。[对照](SCENE_FENCE_TORCH_LOADER_PROBE_20260904.txt)；stderr错误已由v2完整终态记录保存。 |
| 16:28–16:32 | 收窄为RoboTwin `_init_env` opt-in RTLD_LOCAL加载 | 原生函数/so hash不变；移除global LD_PRELOAD和LD_LIBRARY_PATH设计，按原wheel顺序局部加载OIDN依赖且不创建设备。新helper只Env调用，Actor/Rollout/driver不加载。手写多文件diff一次check未通过、未应用；改为对明确的新文件生成标准diff，完整check通过后应用。源码与测试变化详见Fast分支后续提交。 |
| 16:34–16:35 | Env-local真实GPU回归 + 渲染前后PyTorch archive读取 | exit0，2场景/128帧/384图，渲染段52.16s；前后原SFT均version3/5records。LD_DEBUG确认原库目标虚函数绑定到同一shim；GPU6/7释放。[结果](scene-fence-fix-20260904/env-local-smoke-result.json)、[局部绑定](scene-fence-fix-20260904/env-local-native-binding.txt)、[完整日志尾](SCENE_FENCE_ENV_LOCAL_SMOKE_20260904.txt)。 |
| 16:37 | 局部加载修正版commit/push/compose | `62526cc95047c8a4a6e948a76be8eeec8a3926de`已push personal原Fast分支，ls-remote一致，worktree clean。v3对旧256/noOIDN v1仍只有6个路径/名称叶子差异。[推送/逐叶证据](SCENE_FENCE_V3_COMMIT_PREPARE_20260904.txt)。 |
| 16:38–16:41 | v3正式启动/首次核验 | wrapper1568962，observer1568964；两Rollout已完成原SFT加载、Actor正在初始化。双Env1569541/1569544映射shim，Actor/Rollout/driver/Sidney均未映射，无LD_PRELOAD；无RuntimeError/fatal/OOM。[启动](SCENE_FENCE_V3_LAUNCH_20260904.txt)、[隔离现场](SCENE_FENCE_V3_STARTUP_HEALTH_20260904.txt)。Sidney完整52步。 |
| 16:43–16:47 | 权重完成、实际采样、整机与图终检 | 四模型实例原SFT加载完成；16:47 rank0采样1/8（315.85s），双Env/Rollout交互/生成，完整0步，所查异常0。峰值分钟显存57.26GiB/卡，首轮结束后回落；Sidney52步、Step50评估16/32、checkpoint文件齐。原库SHA不变、三worktree clean。生成成功率/优化/资源PNG及交互HTML并目视检查、核验链接。[最终现场](SCENE_FENCE_V3_SAMPLING_HEALTH_20260904.txt)、[图与简报](SCENE_FENCE_RESTART_AND_HEALTH_20260904.md)。 |

## 3. 测试与重启合同

短测：[完整resolved](scene-fence-fix-20260904/smoke-resolved.yaml)，GPU6、1env、2次reset×64frames×3相机=384图，CPU torch线程4；真实RT/SPP32/depth8/noOIDN，零策略查询/零更新；timeout300s，异常即停止，不自动改参重试。输出`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/scene-fence-smoke-20260904`。

精确执行入口：[sz_scene_fence_smoke_20260904.sh](../../../local_scripts/remote_commands/sz_scene_fence_smoke_20260904.sh)，在旧environment基础上仅加载发布enable.sh并设CUDA_VISIBLE_DEVICES=6；`timeout --signal=TERM --kill-after=15s 300s env LD_DEBUG=bindings LD_DEBUG_OUTPUT="$OUT/bindings" "$VIRTUAL_ENV/bin/python" tools/fastwam_scene_fence/smoke.py --source-config "$OLD/runtime/resolved.yaml" --output "$OUT"`。仅短测使用LD_DEBUG，正式训练不启用。

重启：[完整resolved](scene-fence-fix-20260904/train-resolved.yaml)、[精确command](scene-fence-fix-20260904/train-command.txt)、[逐叶diff/预算合同](scene-fence-fix-20260904/train-contract.json)。GPU6/7，32并行×8串行=256；100步/25,600轨迹/400次optimizer calls；fixed32每5步、DCP每10步。原120h timeout；driver fatal/OOM退出；启动观察中确认挂住则只停该run，不擅自改参/重试。没有有效新ETA。完整配置/命令/预算已在执行前展示，用户本轮已授权。

输出：`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v2`。旧run原样保留，resume=null从原SFT重启，正式训练不启用LD_DEBUG。生产库路径为`/home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so`，不是初次release目录。

实现选择说明：`scene_fence.cpp`保留原timeline函数及Denoiser抽象接口，链接原有libsvulkan2，使用wheel自带C++头；只新增reset/submit既有fence。ELF符号替换不是改写原二进制或新增降噪实现；最终只在RoboTwin初始化前RTLD_LOCAL加载，不使用全局LD_PRELOAD。短测须核对动态binding，正式EnvWorker须核对maps/env且Actor/Rollout/driver应未加载；绝不因为so存在就算生效。

## 4. 最终v3合同与交付

[完整resolved](scene-fence-fix-20260904/v3-train-resolved.yaml)、[精确命令](scene-fence-fix-20260904/v3-train-command.txt)、[预算/路径/停止条件](scene-fence-fix-20260904/v3-train-contract.json)。执行前已向用户展示；继承32×8、fresh100、固定评估/保存/优化器全部参数。v1停滞和v2启动失败证据均保留、不拼接曲线。

代码：`codex/sz-fastwam-current-rlinf-grpo@62526cc95047c8a4a6e948a76be8eeec8a3926de`，已推`Yutenji-Nyamu/rlinf_fastwam`。
RoboTwin仍`f3e30a83365c`，没有旧Python生命周期补丁；原shared SAPIEN/svulkan2/OIDN库不变。
v3环境只设置`RLINF_SCENE_FENCE_LIBRARY`，无LD_PRELOAD/新增LD_LIBRARY_PATH；加载发生在RoboTwin初始化内，RTLD_LOCAL，torch之后/SAPIEN之前。
原生.so仍为`release-final/librlinf_scene_fence.so`，SHA256=`45e6cac35ea2edea0724fa4cd92ec82cab401b6abd6f8001805d22306f375df0`；此前build目录内的enable.sh已修正为只导出该变量。

结果目录：`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3`。
16:47终检：两侧Actor/Rollout完成原SFT加载，Step1采样已1/8；完整0步、无新eval/checkpoint，所查异常0。未跨过旧Step2挂点，不宣称长期修复验证成功。
仅Env1569541/1569544映射shim，其余Fast Actor/Rollout/driver/Sidney未加载；driver1568973，wrapper1568962，observer1568964。
Sidney完整52步、153/256，Step50 fixed16/32；Step50双rank checkpoint/full_weights文件在，未做恢复测试。
共享库SHA256与构建前相同，三个worktree均clean；shared Ray/Sidney原进程保持不变。
[简报与详细整机/图](SCENE_FENCE_RESTART_AND_HEALTH_20260904.md)、[完整终检原始数据](SCENE_FENCE_V3_SAMPLING_HEALTH_20260904.txt)。
