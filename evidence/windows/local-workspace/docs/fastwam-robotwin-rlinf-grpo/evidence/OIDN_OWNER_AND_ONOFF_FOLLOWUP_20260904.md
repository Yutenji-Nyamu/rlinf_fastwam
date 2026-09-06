# OIDN 所有权修复与有/无降噪小对照：讨论续记

2026-09-04。本轮为源码研究和方案讨论；没有刷新训练/资源状态，没有运行render/inference/smoke，没有改服务器。

## 1. 用户最新选择

- 旧Python生命周期补丁先保留，但有效性和必要性仍待证；最终修复完成时逐块审计，删除无必要修改，不永久遗留实验性补丁。
- 关闭OIDN不是优先的长期修法，但现在明确允许作为候选讨论；可先做一两次有/无OIDN画面与推理对照，判断观测代价，再决定是否尝试更长运行。
- 不必机械地等“修复失败若干次”才看画面；廉价对照可以前置。但本轮“继续讨论”不等于已批准实施、GPU测试、关闭现役降噪或续训。
- 不升级shared环境，不干扰Sidney/其他用户/shared Ray。若选择无OIDN试运行，必须显式命名渲染配置变化，不悄悄接入旧实验。

## 2. 对“要不要先看有无OIDN”的判断

**赞成把一次廉价的画面/动作对照前置，不必先修几轮失败才看。**“长期修法优先保留OIDN”与“先评估关闭OIDN的代价”并不矛盾。它能帮助决定后续是否值得承担原生构建/回移成本，但一两次对照不能证明无OIDN的长期成功率或稳定性。

建议的最小范围（讨论草案，未执行）：

- 1—2个相同场景状态；每个状态比较OIDN on/off，其他相机、光照、samples-per-pixel、path-depth、shader、尺寸保持原配置。不通过提高spp补偿，避免同时改变两个因素。
- 三路相机都保存同状态原图、差异图、Fast-WAM实际拼接/缩放后的模型输入；重点看夹爪边缘、订书机、接触区域和阴影，不能只看整图平均差异。
- 对两种观测用同一checkpoint、相同proprio、文本和初始推理噪声各query一次；1—2状态合计2—4次策略query，0次更新，不跑完整GRPO。既看画面也看C24动作偏移，关节和夹爪通道分开报告，不设无来源的硬阈值。
- 若渲染本身有随机噪声/累积采样，用一次相同OIDN-on设置的重复渲染作为噪声量级参照，避免将所有像素差异归于开关。无需把它扩成多臂大实验。
- “同seed”不自动等于同状态：应核对实际qpos、物体pose、camera参数；或在不推进物理状态的同一场景上得到匹配观测。若使用已有状态快照，按原快照恢复。

结果怎么用：画面与动作变化都小，可以把显式无OIDN短程试跑作为合理下一候选；只要动作明显变化，就不能用“肉眼差不多”论证无影响。若以后选择无OIDN训练，另记渲染配置，并在该配置下重测SFT/恢复点基线，不将两个渲染版本的指标无标记拼接。

关闭OIDN只是去掉后处理降噪，**不是关闭ray tracing**，也不会消除SAPIEN/Vulkan/相机生命周期的所有风险。即使能避开本次OIDN调用链，也不承诺其他原生故障随之消失。

## 3. 所有者现在能追到哪？

本轮核对精确源码：SAPIEN `d8228489...`、svulkan2 `d8516a4f...`、OIDN `v2.0.1`。这些对应前轮现场安装版本的源码锁；本轮没有再次登录服务器或发布新的动态状态。

```text
RenderCameraComponent
  └─ SapienRenderCameraInternal
      └─ unique_ptr<RendererBase / RTRenderer>
          └─ unique_ptr<DenoiserOidn>
              ├─ OIDN DeviceRef → ThreadLocal<ErrorState> → pthread key
              ├─ OIDN FilterRef / BufferRef
              └─ CUDA stream + Vulkan/CUDA共享buffer与semaphore
```

- `camera_component.cpp:91/108`每个相机内部创建renderer并按全局默认选择denoiser；`:279—283`移出scene时清内部camera，再unregister。
- `rt_renderer.cpp:858`创建DenoiserOidn；`denoiser_oidn.cpp:43`为它新建OIDN CUDA device。不是所有相机只共用一个OIDN device；单例Python Engine不能改变这条所有权链。
- `core/device.h:128`含ThreadLocal；`core/thread.h:39/54`在构造/析构处申请/删除key。**源码有释放函数，不是已经找到漏写一行delete。**问题是相关对象是否真的走到析构、何时走到，以及途中是否损坏。
- `DenoiserOidn::~DenoiserOidn`先free filter/buffer，再放掉device，再销毁stream；智能指针赋空只放掉一个引用，不保证对象立刻销毁。因此诊断必须记录原生对象最终析构及真实key删除结果，不能把API release调用次数当成实际释放数。

来源：[相机所有权](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/sapien_renderer/camera_component.cpp#L39)、[Denoiser实现](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/denoiser_oidn.cpp#L35)、[OIDN2.0.1 ThreadLocal](https://github.com/RenderKit/oidn/blob/v2.0.1/core/thread.h#L39)。

## 4. 真正的小范围修法：本轮新增的具体发现

### 4.1 不能只修init：上层现在会意外退到无降噪

`RTRenderer::enableDenoiser():860—863`在`init()`返回false时，只打印错误、reset denoiser、返回。后续render路径只在`mDenoiser`非空时denoise。于是若只给底层init补错误检查，上层可能在用户要求OIDN时继续输出未降噪图像，而不是明确失败。

应区分两个明确语义：

- **requested=none**：用户主动选择无OIDN，正常走原始渲染。
- **requested=oidn，但创建/执行失败**：报可识别错误，停止该次观测/rollout；不得悄悄变成none，更不能训练一半遇错自动切换。

不需要新建复杂fallback系统；在现有调用链补正确错误传播即可。[调用方源码](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/rt_renderer.cpp#L842)。

### 4.2 可以确定的窄改动面

| 文件/阶段 | 确认的问题或合同 | 建议修改，尚未实施 |
|---|---|---|
| `denoiser.h` | OIDN的mWidth/mHeight未初始化，但render会在allocate前getWidth/getHeight | 显式初始化为0，避免读不确定值；这是代码缺陷，但尚未证明导致本次key失败 |
| `DenoiserOidn::init` | CUDA stream成功后，OIDN device创建/commit没有充分检查 | 逐项检查句柄及OIDN错误；失败时只释放已建立的资源，不暴露半初始化对象 |
| `DenoiserOidn::allocate` | filter/buffer/commit及semaphore导入失败缺乏完整阻断 | 创建结果有效才进入可用状态；失败后清理已成功部分，阻止下一次denoise使用 |
| `DenoiserOidn::denoise` | OIDN错误仅打印，后续继续signal/copy输出；CUDA wait/signal结果未检查 | 传播首个错误，不提交依赖失败结果的后处理，不把坏观测交给策略；不能用黑图/旧图/关闭降噪掩盖 |
| `RTRenderer::enableDenoiser` | 初始化失败后移除denoiser并继续 | 对明确要求OIDN的调用向上传错；显式none仍保持合法 |
| 清理与析构 | 资源跨Vulkan/CUDA，正常路径已有等待和RAII | 保证相关在途操作完成后释放相应buffer/sem，析构不能再抛异常导致二次终止；不要每步添加全局同步/GC |

补充范围控制：header中`useNormal()`返回`mAlbedo`也是笔误，但现路径`init(true,true,true)`两者同为true，不能作为本次根因。本轮只记录，不把无关清理扩成大补丁。

### 4.3 根治泄漏/释放损坏仍需要哪一条证据？

隔离诊断应把scene/camera/denoiser/device的身份及构造/最终析构连起来，同时记录pthread key成功创建/删除、返回码及调用栈。原生对象地址和key可能复用，要按一次生命周期标识，不能只比较地址数。观察反复reset/offload后净存活数是否回到合理平台，不要求包含库级缓存的全部进程资源都归零。

- 若camera/denoiser已销毁但device未销毁：继续追filter/buffer/engine等最后引用持有者，在那个层级修释放。
- 若device/key收支平衡但仍报错：检查其他原生库的key持有和更早的非法使用，不能继续沿“必是OIDN泄漏”单一路径猜测。

这里的诊断由已经出现的key申请失败直接驱动，不是预设一堆测试分支。无需先跑完整策略训练来复现。

已知上游DeviceGuard强引用修复可参考，但不能盲打：该提交的回归用例专门提前释放外部device引用，再让filter释放触发device最终销毁；我们Denoiser正常清理已经先释放filter/buffer、后device，所以现有栈尚不足以证明恰好命中该顺序bug。若确认命中，则回移对应最小逻辑；否则不要拿无关上游patch冒充根治。[上游提交](https://github.com/RenderKit/oidn/commit/9f816f77eb3d6bddaf8d07a96c480444f3d0ee4b)。

另一个重要边界：OIDN **2.0.1** API文档明确支持线程安全调用、同device操作串行化；因此仅凭“reset在线程池”不能认定OIDN必须只在主线程用。这不保证Python绑定、SAPIEN对象生命周期或CUDA/Vulkan异步协作全部安全，需分别定位，不继续扩大旧“统一串行reset就根治”的说法。[2.0.1 API说明](https://github.com/RenderKit/oidn/blob/v2.0.1/doc/api.md)。

## 5. 有/无OIDN对照的实现注意与推进顺序

前轮现场`_base_task.py:216—219`每次初始化明确设置`shader=rt / spp=32 / path_depth=8 / denoiser=oidn`。相机创建时读取这些默认值。因此“在启动脚本先设none”可能随后被覆盖；“相机建完只改全局默认”又可能不影响已有相机。对照要核验每个相机**实际生效**的denoiser状态，不只记录一个配置字符串。

若获批准，在独立诊断进程/独立输出目录中做受控选择；不编辑共用`site-packages`或共享BaseTask，也不在现役EnvWorker里切换全局设置。先确定设置位置及生效方式，再给精确命令/配置/资源/输出/停止条件。这里尚未产出可执行packet，不能把本草案视为已运行或已批准。

建议下一步先做§2的小对照来评估备选成本；保留现有Python补丁不动。画面与动作差异可接受时，再讨论明确标记的无OIDN短程诊断；否则继续锁定OIDN追原生所有权。最终若采用原生修复，或最终选择无OIDN，都要回头审计旧Python补丁，而不是让它因为“暂留”变成永久。

## 6. 取证与记录账本

- 完整读取四份入口及唯一Fast-WAM current SSOT；复用前轮锁定源码路径，不遍历全部历史。
- 获取当前官方API文档作线索，同时读取v2.0.1 `doc/api.md`核对版本差异；没有把当前API新增的external semaphore功能套到旧库。
- `oidn_owner_ab_followup_20260904.cjs`读取svulkan2/SAPIEN/OIDN精确commit/tag文件。初次两个header猜测路径404，按实际include改为`src/renderer/denoiser.h`与`core/thread.h`，全部复取200；初次结果保留，未据404推断文件不存在。
- 本轮进一步核对normal destruction顺序、caller静默退无降噪、未初始化尺寸、OIDN线程安全合同。没有导入项目、构建原生库、GPU渲染/推理、训练、恢复或进程操作。
- 用户明确要求记住的新偏好已保存到长期记忆扩展note，内容限于补丁暂留/后续精简、无OIDN备选可提前廉价对照、讨论不等于运行授权。
- 当前专题/交接更新偏好与下一步；旧实验/资源快照保留原时间，不冒充本轮刷新。

证据：[初次源码采集](OIDN_OWNER_AND_ONOFF_FOLLOWUP_20260904.source.jsonl)、[修正路径后的完整锁定源码](OIDN_OWNER_AND_ONOFF_FOLLOWUP_20260904.source-verified.jsonl)、[前轮上游完整diff](OIDN_CLEAN_FIX_DISCUSSION_20260904.release-fixes.jsonl)、[前轮BaseTask现场源码](OIDN_CLEAN_FIX_DISCUSSION_20260904.live.txt)。
