# Fast-WAM noOIDN Step2 停滞：原生等待链调查

调查时间：2026-09-04 14:50—15:01 CST。范围：普通账号只读现场、历史故障定向检索、上游源码/issue；另做独立 CPU-only 求解器探测。没有改训练源码/参数、安装升级、停止或重启训练/shared Ray。

## 1. 结论与最新状态

已定位“为什么整条训练不动”：Fast rank1 左腕相机读取 RGB，卡在 SAPIEN/NVIDIA 的渲染完成等待，并持续持有该进程的 Python GIL；其他环境线程、Python 超时处理都无法继续。不是模型计算慢，也不是仅凭 importlib 栈就能认定的 TOPPRA 导入死锁。

尚未唯一定位“为什么那次渲染不完成”：可能涉及原生渲染提交/同步、资源生命周期或线程交错；本轮没有取得出问题帧的 semaphore 计数与提交记录，因此不能宣布某个 C++ 根因已经完全证实。

| 15:01 现场 | 结果 |
|---|---|
| Fast 完整训练 | 仍只有 Step1；50/256=19.53%；Step2 未完成 |
| 最后日志进展 | 14:04:08，rank0 rollout 8/8；不是两rank都采完。到15:01已约57分钟无进展 |
| GPU6/7 | 17,699 / 58,727 MiB，均0%利用率；与14:19/14:23/14:50一致 |
| 进程/错误 | wrapper1052625、driver1052633及workers仍在，exit_code不存在；所查OIDN/key/fatal/OOM/Traceback为0 |
| Fast评估/保存 | 14:50文件与事件刷新：没有fixed eval、没有checkpoint；不能据一步评价学习效果 |
| Sidney | 完整Step47，继续Step48；最近训练149/256=58.20%，fixed45=14/32；Step40双shard/full_weights在，未做加载测试 |
| 主机 | 可用RAM约0.955TiB；14:50 CPU空闲95—96%、无swap-in/out；/data余1.27TiB，未见资源打满 |

GPU1仍有2次可纠正SRAM ECC计数（volatile/aggregate不是两组事件）；未见不可纠正ECC/row-remap。本轮普通账号无系统内核日志权限，不把“未见条目”当成完整排除Xid/驱动事故。SSH/mihomo正常、failed units=0。shared Ray仍321933/322685，Sidney driver仍3176215。

## 2. 等待链：证据已经比上一轮更深

```text
rank1 左腕相机 get_picture("Color")
  → SAPIEN getImage → waitForRender
  → NVIDIA libnvidia-eglcore → poll（等待渲染信号）
  → 此线程仍持有该进程GIL
  → TOPPRA导入/NumPy/EnvWorker的120秒超时均无法继续执行Python
  → Env不返回观测 → rollout等观测 → actor等轨迹 → 两卡空闲
```

- PID1053120 / TID1066198：14:23、14:51、14:54、14:56的Python位置稳定在`camera.py:335`；父帧`get_rgba:350`明确是`left_camera`。
- 14:54原生栈明确包含`SapienRenderCameraComponent::getImage → SapienRenderCameraInternal::waitForRender → libnvidia-eglcore.so.575.57.08 → poll`。
- 14:56 `py-spy --json --nonblocking`中，该线程是唯一`owns_gil=true`；其余Python线程均false。这是现场GIL所有者证据，不是只按库习惯猜测。
- 12个环境线程位于`TOPPRA → available_solvers:20 → import qpoases`相关锁；10个停在同一个`_ModuleLock`对象，另外2个在global import lock入口。还有3个线程在NumPy/TOPP计算位置等待。
- `VectorEnv.step:367`本来就有`future.result(timeout=120)`。超时不是操作系统强制中断C++：返回Python、抛异常仍要取得GIL，因此这次超时也被困住。再加Python watchdog线程不能解决这个边界。

锁定源码再次HTTP200核对：SAPIEN `d8228489...`的[`waitForRender:113–121`](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/sapien_renderer/camera_component.cpp#L113)使用`waitSemaphores(..., UINT64_MAX)`；[`get_picture` binding:1040–1045](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/python/pybind/sapien_renderer.cpp#L1040)直接调用`c.getImage`，没有释放GIL的作用域。
[pybind11官方说明](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)确认其不会隐式释放GIL。现场`master`对应binding也未见此改变，不应推断简单升级已有现成修复。

这证明持锁等待造成的进程级停滞；尚未证明完整的循环死锁，也未证明NVIDIA内部在等Python回调。

## 3. 是之前讨论的多训练并发问题吗？

目前证据不支持“两个RLinf任务名字/代码/输出撞了”。14:54读取实际环境与Ray注册：

| 隔离项 | Fast-WAM | Sidney |
|---|---|---|
| namespace | RLinf_1，15个named actors | RLinf，15个named actors |
| GPU | 6、7 | 4、5 |
| RLinf包 | `_ray_pkg_d10bda6d90b6e84f` | `_ray_pkg_75fd8455ab8169ae` |
| code worktree | fastwam-current-grpo | sidney-pi05-current-rlinf |
| RoboTwin | robotwin-clean-oidn-off-20260904 | RoboTwin-RLinf-support |
| 数据/视频 | 本run四条绝对路径 | Sidney run四条绝对路径 |

两条任务的进程不共享Python GIL/import lock；当前等待发生在Fast自己的EnvWorker内部。Sidney在Fast停住期间从46推进到47，也不支持shared Ray整体失效。

但这不等于排除一切整机并发影响：GPU驱动/原生同步资源仍可能有系统级约束。没有进行停Sidney的A/B，不会为验证猜测干扰它。现阶段“进程内渲染同步/线程归属”优先于重建Ray或改namespace。

## 4. 本机以前出现过吗？

定向搜索本地记录，并现场扫描本账号Fast-WAM目录下14个现存run的driver日志；没有遍历其他用户目录，也不能据日志没报错就声称从未有过静默hang。

| 已有故障 | 当时已知边界 | 与这次关系 |
|---|---|---|
| 深圳RLT/DSRL并发启动 | code package未显式传递出现`ModuleNotFoundError: rlinf`；相对`./data`落到共同cwd | 当前包、绝对输出路径已隔离；不是本次现场栈 |
| 深圳RLT Step25卡住 | 首次DCP提取fresh plural optimizer state；2行warmup修复后正式保存通过，旧偶发内部时序未唯一复现 | 同为GPU0%表象，但当前未到checkpoint，也未进入actor update |
| Fast旧128档Step15/34 | OIDN/key错误后Python线程状态fatal | 同属原生环境栈风险，但这次OIDN关闭、无key/fatal，卡在step取图而非关闭场景 |
| Fast旧256/常驻env档 | CUDA OOM；另有一次启动import失败 | 这次没有OOM，Step1已有完整更新；不能套用原OOM修法 |

深圳并发历史详见[原诊断§6–9](../../rlinf-shenzhen-rlt-dsrl-port/10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md#6-17217016389-到底是什么)。本轮未找到更早已保存、与当前`waitForRender + owns_gil`完全相同的本机证据。

## 5. 配置、旧补丁和求解器的核对

- 当前32env×rollout8=256轨迹，16env/rank，GB1024/MB2/update_epoch2；并行数没翻倍，串行轮数翻倍。它增加每个outer step接触环境/相机的次数，但并不直接制造跨job冲突；未证明是本次触发因素。
- OIDN train/eval均`none`；RT shader/SPP32/深度8未改。关闭降噪不是关闭SAPIEN/Vulkan渲染。
- live源码：RLinf `4faade1d...`、RoboTwin `f3e30a83...`、Sidney `f50e235c...`，均clean。新RT从`0008ae6...`仅增加BaseTask开关；旧生命周期分支仍保留，未混入。
- 旧补丁改了reset/close/cache；**SubEnv.step仍在ThreadPool里执行**。当前是step取图，不是reset；全部恢复旧补丁并没有已知充分理由，且不能保证解此问题。
- 现场版本是**TOPPRA0.6.9**、MPLib0.2.1、SAPIEN3.0.1、NumPy1.26.4；不沿用旧standalone环境的0.6.3记录。`qpoases`、`cvxpy`均未安装，这是可选依赖探测，实际solver为seidel，并非缺包导致必须补装。
- 本轮独立CPU进程，16线程×256次`available_solvers`，4096次在1.935秒完成，exit0；未使用GPU、模型或Ray worker，无训练更新。仅说明此最小探测未复现，不排除组合场景时序。
- 本版`ReachabilityAlgorithm.__init__:63`无条件探测可选solver，即便传`solver_wrapper="seidel"`也不会跳过该行。因此“指定seidel就修好”也不成立。

另核对svulkan2 `d8516a4f...`的[`RTRenderer::render:702–726`](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/rt_renderer.cpp#L702)：noOIDN跳过denoise，后处理提交和signalSemaphores仍在if分支外。没有从源码发现“关闭OIDN必定漏掉最终signal”的简单分支错误；实际未完成信号还需要单帧提交证据。

## 6. 上游搜索找到什么

| 一手来源 | 与现场的适用边界 |
|---|---|
| [SAPIEN #171](https://github.com/haosulab/SAPIEN/issues/171) | H100/L40也有人报告take_picture不报错而冻结；维护者把其中createFence OOM解释为驱动同步资源耗尽，但明确未确定无输出冻结的原因。我们没有createFence错误，不认领为同一资源耗尽 |
| [RoboTwin #477](https://github.com/RoboTwin-Platform/RoboTwin/issues/477) | 单进程、独占GPU、关OIDN、clear_cache_freq=1仍可能render hang。硬件为Blackwell，不能直接当成本H100同根因；足以说明关OIDN不保证消除全部渲染hang |
| [RLinf #1040](https://github.com/RLinf/RLinf/issues/1040) | LingBot/RoboTwin reset线程fatal；评论env8/32训练仍可失败。更接近旧事故，不是本次GIL/信号等待的根治方案 |
| [TOPPRA PR #298](https://github.com/hungpham2511/toppra/pull/298) | 已于08-23合并`80722685...`，用find_spec替代optional solver imports，主要解决探测开销。我们0.6.9尚是旧实现；它不会释放相机持有的GIL，不能当本次主修复 |
| [RoboTwin #83](https://github.com/RoboTwin-Platform/RoboTwin/issues/83) | H100等某些随机场景render几十分钟后继续；报告GPU仍在工作，当前长期0%且clean任务，不可凭此删除对象或更改任务 |
| [RoboTwin PR #471](https://github.com/RoboTwin-Platform/RoboTwin/pull/471) | XPolicyLab结果保存后的进程退出清理，不在当前RLinf step路径；强杀worker不是根因修复 |

检索同时覆盖官方仓库issue搜索、准确函数/错误关键词和源码；通用搜索有大量无关结果，未用于推断。搜索不是穷尽性证明，目前未找到可直接套用、已验证适用本现场的现成补丁。

## 7. 建议如何干净推进（本轮未实施）

**15:25后续研究已调整本节优先级：** [noOIDN与scene fence因果审查§4–5](FASTWAM_NOOIDN_SCENE_FENCE_CAUSAL_REVIEW_20260904.md#4-我建议的干净改动补齐现有scene-fence协议)核对了实际库，找到timeline重载漏接scene-access fence。当前主修候选是补齐这条已有协议，而不是先修超时/GIL或先换线程模型。下列内容保留为15:01时的建议历史，不作为并行实施计划。

先把两个目标分开：让真正的渲染错误可报告，以及修复首次导致渲染不完成的原因。

1. **确定的小修方向在SAPIEN等待边界**：为原生render semaphore等待提供有限上限并向上抛错；在纯C++等待/取图段正确释放GIL、构造Python返回对象前重新取得GIL。必须同时检查相机/缓冲区生命周期和线程互斥，不能对包含Python对象操作的整个lambda盲加release。仅在Fast独立构建/路径使用，不覆盖共用venv，不升级整套库/改变画面。此修正让超时能工作，**不保证渲染本身恢复正常**，不自动吞错/返回旧帧/补零或把故障记成策略失败。
2. **针对真正触发器只做一个高信息量复现**：沿用noOIDN、16env/rank及真实step/reset/offload，在隔离环境比较现有线程池路径与统一渲染线程归属；记录相机/帧ID、submit和wait的目标/实际semaphore值。现在`gen_sparse_reward_data`内部每个physics step也调用`_update_render`，所以只把最后`get_obs`搬出线程池未必覆盖全部交错。
3. 若验证是线程交错，窄修RoboTwin/SAPIEN渲染提交与读取的归属/同步；保留32env×8、G8、模型batch/LR/update预算。只有证据指向资源销毁顺序时，才逐块回移旧生命周期修复。若统一线程仍复现，继续查那一帧的Vulkan信号/资源，不继续叠Python GC或import补丁。

第2项需要新GPU复现/可能替换停滞run，应另行取得执行授权并展示命令/资源；本次诊断没有自动发起。当前run没有checkpoint，不能宣称可以从Step1严格恢复；重启需明确从SFT重新开始，不能无声吞掉Step2接着训。

## 8. 原始证据与操作账本

时间均按文件内部时间换算CST；文件名部分后缀是计划标签，不是准确采样分钟。

| 时间 | 操作 / 结果 |
|---|---|
| 14:50 | 复用`sz_current_training_health_20260904.py`经SSH stdin只读刷新：[两run/整机/源码/文件JSON](STEP2_STALL_HEALTH_20260904_1448.txt) |
| 14:51 | `sz_step2_deep_readonly_20260904.sh`：重复非阻塞Python栈、installed TOPPRA/MPLib、任务/相机源码、线程wchan/syscall：[证据](STEP2_STALL_STACK_SOURCE_20260904_1450.txt) |
| 14:53 | `sz_step2_lock_readonly_20260904.sh`：只保存导入帧局部变量；`--native --nonblocking`组合被py-spy拒绝；`/proc/task/stack`无权限，未升权：[证据](STEP2_STALL_LOCKS_20260904_1454.txt) |
| 14:54 | 向用户说明后，对故障EnvWorker短暂暂停采一次native栈（20秒工具上限、实际本段不足一秒），自动脱离；正确路径完整读取VectorEnv与旧diff；列named actors后仅shutdown诊断client：[原生栈/作用域](STEP2_STALL_NATIVE_SCOPE_20260904_1457.txt) |
| 14:56 | `sz_step2_cpu_probe_20260904.sh`：CPU-only4096次solver探测exit0；读取GIL所有者/参数绝对路径/14份Fast日志：[证据](STEP2_STALL_CPU_HISTORY_20260904_1500.txt) |
| 15:01 | 最终GPU/进程/log/fatal刷新，Fast仍卡、Sidney47：[状态](STEP2_STALL_FINAL_STATUS_20260904.txt) |
| 本轮 | 官方源码与issue/API只读查询：[主证据](STEP2_STALL_UPSTREAM_20260904.jsonl)、[PR298合并与真实diff](STEP2_STALL_TOPPRA_PR298_20260904.jsonl) |

读取错误已纠正：首次误用`envs/vector_env.py`得到文件不存在，且该错误路径的git diff为空，**未用于判定补丁**；实际文件是`robotwin/envs/vector_env.py`，14:54完整读取与diff成功。网页工具部分源码cache miss，PowerShell HTTPS报认证错误；随后既有Node fetch以正常TLS取回HTTP200，未绕过证书验证或增加凭据。既有本地源码缓存仅用于定位，关键GIL/无限等待源码已重新取回。
