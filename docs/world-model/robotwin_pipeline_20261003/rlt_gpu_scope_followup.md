# 深圳3 RLT 图形卡位：后续归还时处理

2026-10-03，本地只读源码审计；没有 SSH、没有 GPU 探针、没有修改服务器 profile、冻结 owner、借还计划或训练源码。本页是后续方案，不是深圳3修复成功回执。WM 优先，当前 WM 不因本项暂停。

## 结论

深圳1/2已经验证了同一类修复：CUDA 选卡之外，用 NVIDIA Application Profile 限制 EGL 辅助图形上下文。深圳3应复用**现有 GPU4 scope v5 的 Ray 兼容实现，扩成每个 RLT 任务自己的单卡 scope**；5/6/7分别只见自己的图形卡，不使用全账户默认规则。

**现在不能直接新增 profile。** GPU4 v5 的 `read_manifest()` 会拒绝原审计快照之外、含 `EGLVisibleDGPUDevices` 的任何新文件，规则即使只匹配5–7也一样。直接安装会使冻结借还链的 GPU4 归还/后续新子进程检查失败。应在后续明确的资源交接中，使用同时认识所有规则的新 scope revision；保留旧文件与历史回执。

**不能声称“5–7归还后0卡问题已解决”。** 当前冻结的5–7归还沿原启动环境；旧 RLT 退出可以消除它们原有的0卡上下文，但以后归还的新 RLT 是否再创建0卡上下文，必须实查。

## 已有实现分别能复用什么

| 来源 | 现成机制 | 在深圳3的限制 |
|---|---|---|
| 深圳1 Dojo | 每个独有 `commname` 对应单卡 EGL mask；私有 `sitecustomize.py` 早期设置进程名 | 单次 `prctl` 不能覆盖 RLinf/Ray 后续改进程名；其账户默认 mask240 不应照搬，可能把5–7的辅助上下文放到4 |
| 深圳2 EXPO | 私有 bootstrap、profile/主机/UID校验、真实 spawn 三相机探针、独立 owner 交接与回执 | EXPO 是4–7统一 UUID CVD；RLinf 单卡物理 placement/数字 CVD 和 CPU ChannelWorker 路径不同，不能直接替换 |
| 深圳3 GPU4 v5 | 私有 comm、Ray `setproctitle` 后恢复主线程 comm、CUDA身份核验、SAPIEN `RenderSystem("cuda:0")`、CPU ChannelWorker 全卡列表特例 | 当前硬编码物理4，profile快照冻结；仅更换环境变量或复制manifest不足以支持5–7 |

现成源码：

- [Dojo bootstrap](../../../local_scripts/dojo_parallel_20261003/gpu_scope/sitecustomize.py)与[逐卡profile](../../../local_scripts/dojo_parallel_20261003/gpu_scope/nvidia-application-profiles.json)。其已验收情况见[深圳1报告](../../robodojo-openwam/GPU_SCOPE_FIX_20261003.md)。
- [EXPO已发布bootstrap](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/568fd5a017597998f0c8ebe265d6f20b3e39b699/tools/expo_gpu4567_20261003/bootstrap/sitecustomize.py)、[profile](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/568fd5a017597998f0c8ebe265d6f20b3e39b699/tools/expo_gpu4567_20261003/graphics-profile.json)、[明确交接入口](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/568fd5a017597998f0c8ebe265d6f20b3e39b699/tools/expo_gpu4567_20261003/handoff_gpu4567.py)。本次读取的是本地对应 publication checkout；没有重查远端实时状态。验收摘要见[深圳2报告](../../methods/expo-ft/GPU4567_BINDING_20261003.md)。
- [GPU4 v5 runtime](../../../local_patches/opendw_smoke_20261003/drafts/scope_tools_v5/gpu_scope_runtime.py)：`read_manifest`第33行、新增规则拒绝第55行、Ray标题保护第82行、CVD第136行、渲染绑定第196行。SHA256 `02aaabc0d824c33bc837a761bdc5f5051c4c6cbe78451fd71fb28cd257b73709`。
- [GPU4原prepare](../../../local_patches/opendw_smoke_20261003/tools/gpu_scope_prepare.py)：`profile_snapshot()`直接拒绝已有EGL规则，不能当作可重复的多卡安装器运行。

## 最小新版本边界

保持现有GPU4机制，仅把硬编码目标参数化为manifest的 `physical_gpu in {4,5,6,7}`，覆盖manifest校验、数字CVD、回执字段及错误提示；GPU UUID/PCI和minor仍从目标服务器现查。每任务独立comm、profile、manifest、bootstrap和回执目录。

必须保留以下已有行为：

1. GPU actor/env/rollout只接受该任务自己的数字卡号或正确UUID；只有已定位的CPU ChannelWorker精确全卡列表 `0,1,2,3,4,5,6,7` 收窄到本任务目标。未设/空CVD的CPU发现进程仍保留原行为，其他错误卡位拒绝。
2. Ray改变argv标题后恢复**主线程**comm；不能只在bootstrap中设置一次，也不能把非主线程的 `prctl` 当作全进程绑定。
3. 仅在原生RoboTwin环境导入时绑定SAPIEN 3.0.1已核对的Scene接口，验证logical CUDA0的UUID/PCI；不提前在actor/rollout导入仿真。
4. profile mask按已核对的NVIDIA minor计算，不能从物理索引直接假定；规则只匹配本人任务私有comm。不给整账户追加无条件fallback，不修改共享Ray。
5. 新revision一次审计现有GPU4及新增5–7规则与优先级，并在**所有将继续使用的scope**中冻结同一已审清单。保留GPU4旧规则和回执；不能在旧v5 manifest未换代时悄悄增加全局profile文件。

本次没有额外生成部署patch，避免出现看似可直接运行、实际缺少深圳3身份和规则优先级审计的入口。

## 如何接在归还时

当前[多卡owner](../../../local_patches/opendw_smoke_20261003/multigpu/opendw_multigpu_owner.py)在 `finally` 内完成WM精确清理后直接调用 `H.resume`（第650行），没有外部替换归还scope的交接钩子。[合成cycle](../../../local_patches/opendw_smoke_20261003/multigpu/rlt_multigpu_cycle.py)同时校验子计划与源码SHA，再分别调用GPU4、GPU5–7既定resume。**因此不能靠一个外部文件或现场修改冻结源码，接管正在运行的owner归还。**

可行的独立交接边界：等待当前WM自然结束并生成唯一终态/释放/归还回执，先实查实际返回结果；在下一次已授权的正常切换中，新maintenance owner凭旧owner终态、精确PID/start/UID、namespace、checkpoint及配置摘要接管。新owner只加入各任务图形scope，保持任务、CP、N/G/步数、模型及优化参数，随后验证新首轮和全卡C/G记录。若下一步继续WM，优先继续WM，本项留在RLT下一次实际归还前完成；不要为本项打断健康WM。

若归还尚未发生且要替换本轮归还链，必须另做明确的owner交接协议；当前源码没有现成入口。这不是安装几个profile即可安全完成的小改动，本轮不展开。

## 何时可称解决

最少验收为：服务器CPU检查覆盖4–7各自mask及Ray主/后台线程改名；在已释放的获准卡上，真实环境reset、三图、少量动作和正常关闭通过；新RLT实际使用该scope，0–3没有本任务计算或图形PID，退出后本任务上下文消失。外用户上下文或驱动底层残余单独记录，不能以“无计算PID”代替“完全空闲”。

本页状态：**方案明确；深圳3 GPU5–7返回scope尚待实现与实机验收。** WM并行smoke、训练和云端发布继续作为主线。
