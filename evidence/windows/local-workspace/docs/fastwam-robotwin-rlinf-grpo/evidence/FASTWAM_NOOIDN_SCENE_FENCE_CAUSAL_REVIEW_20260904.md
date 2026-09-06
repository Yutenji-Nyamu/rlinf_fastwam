# Fast-WAM noOIDN Step2：从等待症状追到场景同步缺口

日期：2026-09-04；服务器只读状态15:17，源码/非阻塞栈15:19，实际加载库的磁盘二进制核对15:25 CST。
本轮范围：研究“是否由我们的改动触发、什么才是干净修复”。未改生产代码、未编译/安装、未启动GPU测试、未停止训练或重启Ray。

## 1. 直接结论与置信边界

**不能再笼统说“与关OIDN无关”：本轮找到了一条具体、可解释的关联。**

- 锁定svulkan2的相机实际调用分支，遗漏了既有的`mSceneAccessFence`重置与提交；另一重载却完整接入。这一缺口也在故障进程实际加载的`libsvulkan2.so`磁盘文件中得到反汇编核对，不是只读错了GitHub版本。
- 同一场景的三台相机先连续`take_picture`，之后才统一取图。每台相机准备渲染时都会进入共享场景的更新路径，可能重置仍在GPU执行的场景命令缓冲、重写仍被访问的资源。上层Python串行调用，不代表底层GPU已经完成。
- OIDN的同步`execute()`会等待输入渲染及降噪完成；关闭后移除了这层隐式等待，**可能暴露上述原本就存在的同步缺陷**。这是目前比“多训练冲突/线程归属不明/资源可能泄漏”更具体、优先级更高的主因候选。
- **已确认的是源码/二进制里的同步接线缺口，以及可达的不安全时序；未确认的是本次卡死那一帧确实走中了该时序。** 没有该帧的提交/完成记录或validation报错，不能宣称已唯一锁定驱动卡死根因，更不能宣布修复成功。

| 本次变化 | 与当前故障的判断 |
|---|---|
| OIDN `oidn → none` | 有明确机制关联：少了同步降噪带来的等待；不是“把渲染关掉”，也不是我们漏写最终相机signal |
| rollout `4 → 8` | 16env/rank并发未变；串行采样和每步optimizer calls翻倍。增加暴露次数/改变策略轨迹，但不是场景fence漏接的来源 |
| 撤下旧Python生命周期补丁 | 当前回到0008ae6的reset/close/cache行为；旧补丁从未改这条C++同步分支，也没把step移出线程池，不能认为恢复它就能补上此缺口。此前生命周期的间接影响仍不能完全排除 |
| 多RLinf任务共用Ray | 前轮已实查namespace、代码包、GPU、输出隔离；本轮Fast停滞时Sidney继续推进。没有支持旧路径/namespace冲突复发的证据；并非排除了所有系统级驱动影响 |

15:17只读现场：Fast仍完整Step1，GPU6/7=17699/58727MiB、0%；无exit_code，所查OIDN/key/fatal/OOM/Traceback为0。Sidney已完整Step48、仍运行；RAM available=1,049,698,332,672B（约0.955TiB）。15:19非阻塞Python栈仍停在左腕相机取图。以上为带时间戳快照；本轮未重新读TB/checkpoint，相关上一轮结论不冒充本轮刷新。

## 2. 问题具体在哪：同一场景的相机之间，不仅是线程之间

现场RoboTwin `f3e30a83365c`：

1. `Base_Task.get_obs:463–476`先`_update_render()`，再`cameras.update_picture()`，之后才`get_rgb()`。
2. `Camera.update_picture:273`依次左腕、右腕、静态相机调用`take_picture()`，没有在相邻相机之间等待当前帧完成。
3. SAPIEN [`takePicture:124–130`](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/sapien_renderer/camera_component.cpp#L124)先等**该相机自己的上一帧**，再调用timeline重载提交当前帧；不等同场景另一台相机的当前帧。
4. 每台RTRenderer都有自己的`mSceneRenderVersion`；[`prepareRender:159–165`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/rt_renderer.cpp#L159)分别进入共享`Scene::updateRTResources()`。该函数没有按`mRTResourcesRenderVersion`跳过第二台相机的重复更新。
5. [`Scene::updateRTResources:1207–1224`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/scene/scene.cpp#L1207)本来会等所有注册的scene-access fences，再更新TLAS和场景buffer；但timeline重载没有把fence置为未完成并接到提交上，初始“已完成”状态因而不能保护真实在途工作。

```text
左相机：更新共享场景 → 提交左相机GPU渲染 → 返回Python
                                           │ GPU可能尚未完成
右相机：检查scene fences（仍是“完成”）→ 重用共享场景更新资源
                                           ↓
                    若命中在途重用：渲染/驱动状态可能异常
                                           ↓
统一取图：左相机等最终完成信号 → 无限原生等待且持GIL → 整个EnvWorker停滞
```

因此，**仅把渲染移到同一个Python线程，仍可能保留以上顺序**；也不需要两个RLinf训练相互干扰才能形成这条危险路径。当前卡点是“最终完成信号未到”，但更早漏接的是另一个**保护共享场景资源的fence**，两者不能混为一谈。

## 3. 排除了哪些“看起来像，但不成立”的解释

### 3.1 不是最终signal被OIDN分支吃掉

[`RTRenderer::render:702–726`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/rt_renderer.cpp#L702)中后处理和最终timeline signal仍在`if (mDenoiser)`外。前轮此判断保留，但不足以证明所有同步正确。

### 3.2 不是队列根本没有锁，也不是缺少RT→后处理屏障

- [`Queue::submit:27–40`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/core/queue.cpp#L27)有`mMutex`；它保护CPU端调用`vkQueueSubmit`，不是GPU完成证明。
- `recordPostprocess:893–904`已有RayTracing/Transfer→Compute屏障；不能看到noOIDN跳过一段就断言所有图像屏障缺失。
- Vulkan明确规定：同一队列的命令也需要正确同步，**不能把仍为pending的命令缓冲reset**。[命令缓冲规则](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html)、[vkResetCommandBuffer约束00045](https://docs.vulkan.org/refpages/latest/refpages/source/vkResetCommandBuffer.html)。
- 这里真正具体的危险操作是`Scene::updateTLAS:842–881`每次重置、重录并异步提交同一个`mASUpdateCommandBuffer`；`TLAS::recordUpdate:275–312`还写同一CPU_TO_GPU实例buffer、原地更新同一TLAS。[`as.cpp`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/core/as.cpp#L275)与[`Buffer::upload:152–182`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/core/buffer.cpp#L152)核对：host-visible上传直接memcpy，不能当成自动等GPU完成。

### 3.3 OIDN为什么可能把问题遮住

[`DenoiserOidn::denoise:156–165`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/denoiser_oidn.cpp#L156)：Vulkan拷贝输入并发信号，CUDA stream等待该信号，然后调用同步`mFilter.execute()`，不是`executeAsync()`。锁定OIDN2.0.1 [API说明](https://github.com/RenderKit/oidn/blob/v2.0.1/doc/api.md#L967)确认execute阻塞到降噪完成。这会让前面的输入渲染已经完成，再走到下一台相机。

这是**具体时序机制的推断**，不是OIDN被设计为scene fence，也不是证明所有OIDN开模式都安全。旧OIDN key/fatal是另一条已观察故障链，不能用本次scene fence解释其所有现象。

### 3.4 实际库与上游历史的双重核对

15:25从故障PID1053120的`/proc/maps`定位实际库，仅读取磁盘文件做SHA256、nm/objdump，未attach、未暂停进程：

- `sapien.libs/libsvulkan2.so` SHA256=`972eff8fc5fedfd59d4cd8794039d9bc850847a6c74dfbf7d549eff9d041b1b7`。
- timeline重载地址`0x574a90`：`prepareRender`后直接取queue，在`0x574af3`把fence参数寄存器`r9`清零，再于`0x574b50`submit；没有对应reset fence步骤。
- binary重载地址`0x574850`：`prepareRender`后执行device-dispatch调用，并于`0x574901`从对象偏移`0x4d0`取fence传给submit，与源码里的reset/提交既有fence一致。
- 上游早在[commit 28852c0](https://github.com/haosulab/sapien-vulkan-2/commit/28852c021815b9b62f4442240e07376fb7f93936)就加入scene fence管理与binary重载的reset/submit；本轮未找到timeline分支已合入的对应修复。不要把这条旧提交包装成新发现的、已验证适用本次故障的上游修复PR。

## 4. 我建议的干净改动：补齐现有scene fence协议

**首选候选是锁定svulkan2版本，修timeline重载漏接的scene fence；不是先改线程模型或整库升级。** 逻辑上对齐同文件已存在的binary重载：

```cpp
prepareRender(camera);
mContext->getDevice().resetFences(mSceneAccessFence.get());
mContext->getQueue().submit(mRenderCommandBuffer.get(), {}, {}, {},
                           mSceneAccessFence.get());
// OIDN选择、后处理、最终timeline signal保持原样
```

这是讨论中的代码形状，**未应用、未编译、未测试**。`resetFences`必须在`prepareRender`之后；提前reset会让prepare等待自己还没提交的fence。提交失败应向上失败，不能吞异常并重用对象。

它补上的约束是：**上一台相机的共享场景访问完成后，下一台才允许重用场景资源。** 不改变物理状态、shader/SPP32/depth8、降噪选项、相机配置、模型和GRPO公式；保留32env×8、GB1024、update2。可能改变吞吐和消除未定义行为产生的画面错误，不承诺像素逐位相同。

实施时只为Fast建立隔离的同版本原生构建/加载路径，并核对真实`/proc/maps`与hash；不覆盖Sidney共用venv或原文件。不能在未核对ABI/构建配置前随意替换一个so。

**对上一轮建议的优先级修正：** 有限等待/GIL处理只能解决挂死后的传播，现在不应把它当首个“修好训练”的补丁。先补这个明确同步协议；GIL释放与原生有限超时如后续要做，应单独审查Python对象/相机生命期和异常退出。贸然释放GIL还会扩大C++并发面。整包恢复旧cache/reset补丁、换TOPPRA、改并发/采样、加sleep/GC都不是此处最直接修法。

## 5. 一次验证应该回答什么，而不是盲跑一晚

不需要先反复跑GRPO。若后续授权实施，第一项验收应是**同一场景、同样三台相机、OIDN关闭的连续取图**：在独立进程中给旧/候选库做相同的有限回归，用Vulkan validation或轻量提交状态记录检查场景命令缓冲/fence的重用边界。不要只看“跑完了没崩”。

- 旧版若捕获pending-reset/资源hazard，候选补丁消除同一错误，才能把本轮静态推断进一步闭环。
- 若没捕获旧版错误，结论只能是补齐了具体协议、短回归通过，不能据此宣布Step2唯一根因已证实；若补后同样卡，则保留提交/完成证据转查该帧，不继续叠经验补丁。
- 不把GPU基础设施失败算作策略失败，不返回旧图、补零或静默重采轨迹。

本轮没有启动这项验证。替换/停止当前停滞run、构建或新GPU运行仍需后续执行授权；不能借“深入研究”干扰Sidney或shared Ray。

## 6. 网络检索与操作证据

相关issue仍是旁证，不替代本次代码链：[SAPIEN #171](https://github.com/haosulab/SAPIEN/issues/171)、[RoboTwin #477](https://github.com/RoboTwin-Platform/RoboTwin/issues/477)。额外核对#171所指旧`rt_renderer.cpp:377`，实际是图像初始化的`submitAndWait`，不是本次`getImage→waitForRender`同一行；不合并成同根因。[RLinf #712](https://github.com/RLinf/RLinf/issues/712)讨论rollout吞吐、placement、cache频率，没有提供本次scene-fence遗漏的修复。检索官方SAPIEN/svulkan2/RLinf/RoboTwin issues、相关文件提交历史及Vulkan/OIDN规范；未找到可直接认领为已验证本案的现成PR。

| 时间/操作 | 结果与原始证据 |
|---|---|
| 15:17 固定host-key普通用户SSH，复用状态脚本 | [GPU/RAM/进程/log](STEP2_CAUSAL_STATUS_20260904_1517.txt)；无写服务器 |
| 15:19 `sz_step2_causal_readonly_20260904.sh` | [新旧Git diff、真实camera/get_obs、非阻塞栈](STEP2_CAUSAL_SERVER_SOURCE_20260904.txt)；新RT与RLinf dirty为空，HEAD锁定 |
| 上游source/issue/commit HTTP只读 | [queue/scene/renderer sources](STEP2_CAUSAL_SOURCES_20260904.jsonl)、[首次sync核对](STEP2_SYNC_CONTRACT_SOURCES_20260904.jsonl)、[补充及目录确认](STEP2_SYNC_CONTRACT_FOLLOW_20260904.jsonl)、[TLAS实现](STEP2_SYNC_AS_20260904.jsonl)、[#712](STEP2_RLINF_712_20260904.jsonl) |
| 15:23 初次磁盘库核对 | [记录](STEP2_SYNC_BINARY_20260904.txt)；系统Python不支持hashlib.file_digest，身份/路径已读后停止，未改服务器 |
| 15:25 改用既有venv Python后重查 | [实际库hash与两render重载完整反汇编](STEP2_SYNC_BINARY_VERIFIED_20260904.txt)，exit0；未attach或暂停训练 |

本地工具小错误：初次`.cjs`顶层await语法不适配，去掉顶层await后原请求成功；猜测`acceleration_structure.cpp`/`rt.cpp`路径返回404，随后只读Git树定位真实`src/core/as.cpp`，未把404当代码不存在的证据。GitHub网页工具部分cache miss，已有Node HTTPS以正常TLS取回HTTP200。无生产实现变更。
