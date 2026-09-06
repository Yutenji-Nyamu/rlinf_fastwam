# OIDN干净修复讨论：补丁去留、上游证据与代码边界

2026-09-04；服务器代码/配置/fatal于10:13 CST只读刷新。本轮讨论，不实施、不安装、不渲染测试、不续训。

## 1. 直接回答：补丁去留待证，收回“已修好根因”的说法

原补丁确实执行了，也把子环境清理与全局缓存清理分开了。**但现有证据不能证明它治好了旧事故，也不能证明延迟了崩溃。**目前没有证据表明本次事故由补丁引入，不因复发立即整体回撤；同样不因结构更清楚就永久保留。当前保留停机现场，去留待证；最终修复逐块确认必要性，不必要的部分应删除。详见本日后续[最小修改与学习效果复核 §1](FASTWAM_LEARNING_AND_MINIMAL_FIX_REVIEW_20260904.md#1-直接结论锁版本小范围旧补丁去留待证)。本轮没有执行回撤。

本轮有两项重要更正：

1. **`clear_cache()`不是强制销毁所有renderer。** SAPIEN binding调用svulkan2 `clearCachedResources(models,images,shaders)`，默认models/images=true、shaders=false；实现是带锁清理模型、图片、纹理等registry。与明确标记会让所有renderer失效的`releaseGPUResourcesUnsafe()`是两个函数。此前“兄弟场景还活着时clear一定导致invalid handle”的因果定性过强，不能再作为确定根因使用。
2. **本run实际清理频率是1，不是8。** `runtime/resolved.yaml`的train/eval `task_config.clear_cache_freq`都为1，`RoboTwinEnv._init_env`完整传入VectorEnv；8只是后者缺省值。因此full reset走每次整批关闭后GC/global clear，FAQ的“设成1”已经满足。

源码：[SAPIEN binding](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/python/pybind/sapien_renderer.cpp#L498)、[clearCachedResources实现](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/resource/manager.cpp#L380)、[与unsafe API的区别](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/include/svulkan2/resource/manager.h#L102)。

## 2. 真正还没解决的是什么？

通俗地说：环境反复拆建，底层降噪器也反复申请和归还“登记号”。现在先看到登记号申请失败，随后系统仍拿无效对象继续渲染，最后在清理场景时彻底退出。我们需要弄清的是：**谁没有归还资源，或谁在对象释放后仍使用它。**前者是积累/保留，后者是生命周期失配；两者不能仅凭相同报错词混为一谈。

已知事实：OIDN 2.0.1的`Device`含一个`ThreadLocal<ErrorState>`，其构造/析构申请/删除pthread key；日志的首错与此类key申请失败一致。1024是本机进程内pthread key槽上限，不是可创建线程数。尚无崩溃进程的活跃key计数、分配调用栈或泄漏对象归属。

新fatal与旧fatal均落在原生场景/渲染清理附近，但它们可能是同一个积累问题的不同表现，也可能包含两处缺陷。不能称为“旧问题已修好、只是新问题”，更不能直接归罪GRPO或Fast-WAM模型。

## 3. 广泛检索后，哪些线索真正相关？

检索覆盖RoboTwin官方FAQ/issue、SAPIEN issue/源码、svulkan2源码、OIDN官方版本史及逐提交diff；一般搜索存在不相关结果，未用那些结果作归因。关键来源如下。

| 来源 | 查到什么 | 对我们的意义与限制 |
|---|---|---|
| [RoboTwin官方FAQ](https://robotwin-platform.github.io/doc/common-issue/index.html) | invalid handle建议clear_cache_freq=1或查SAPIEN #243 | 我们已是1；不能反复调同一个开关当新修法 |
| [SAPIEN #243](https://github.com/haosulab/SAPIEN/issues/243) | 5090一开始渲染就失败；报告者通过替换OIDN解决 | 硬件/首错不同。评论中的直接覆盖shared site-packages、LD_PRELOAD不是我们的部署建议 |
| [SAPIEN #258](https://github.com/haosulab/SAPIEN/issues/258) | Ada平台illegal memory access，按#243升级后仍失败 | 相同组件不等于相同根因；升级不保证治百病 |
| [OIDN 2.1.0及后续版本史](https://www.openimagedenoise.org/) | 2.1修GPU对象释放崩溃，2.2.1/2.2.2继续修device释放泄漏 | 是有直接机制依据的候选，优于凭空添加Python清理；但未证实我们命中同一bug |
| [OIDN 9f816f77](https://github.com/RenderKit/oidn/commit/9f816f77eb3d6bddaf8d07a96c480444f3d0ee4b) | DeviceGuard从裸Device指针改为强引用，确保释放操作完成前device仍活着 | 具体生命周期修复，不是单纯增大缓存/内存 |
| [OIDN d4af2c66](https://github.com/RenderKit/oidn/commit/d4af2c667497a42a5c06c95e6d6b5d7f5cd35349) | 进一步修析构顺序、释放后指针状态与析构异常路径 | 同样与销毁时崩溃相关，适合作为原生修复候选审查 |
| [OIDN a179f0b9](https://github.com/RenderKit/oidn/commit/a179f0b9090cdde0398fb0ee422665830c6b2c7c)、[d9e6124a](https://github.com/RenderKit/oidn/commit/d9e6124aec543d86f368f7b513e7cbe8f59c820b) | 调整buffer/device引用关系及engine/缓存资源的销毁顺序 | 2.2.1/2.2.2确实做了所有权修复；不能把该版本的具体循环引用结构套到2.0.1 |

重要排除：2.0.1的Engine使用weak scratch-manager、USMBuffer持有Engine；2.2.x的缓存/buffer/device组织已有变化。故本轮**没有**下“2.0.1必然具有2.2.1修复的同一个引用环”结论。

还有一条版本约束：OIDN 2.2.0及2.3.0包含降噪画质/性能变化。换库可能改变策略看到的RGB，不能保证只改了稳定性。官方修复release可作隔离候选；若要求严格保持原图像生成语义，应回移已证实相关的生命周期修复并验证输出，而非悄悄替换所有库。

## 4. 深入我们的代码：下一刀应落在哪一层？

服务器10:13读取：Fast-WAM HEAD=`4faade1d50bf21d1caf1b8a4e5f89282a810208a`，RoboTwin fix=`8c7380c118ce7ca8a4ea4df53d753adc8fab0df2`；两工作树clean、补丁文件hash未变。

| 层 | 已核对路径/行为 | 修复含义 |
|---|---|---|
| RLinf | `env_worker.py:1074/1079`每wave bootstrap，train auto_reset=false；`1367`按配置offload；train/eval offload都true | 频繁创建/销毁是正式路径的一部分，不能为绕过故障随意改预算/常驻策略 |
| RoboTwin wrapper | `robotwin_env.py:88`传resolved task_config；`:251`reset进入VectorEnv | 清理频率1真实传递，无“配置未传进去”的证据 |
| VectorEnv | `vector_env.py:403—424`full reset先关所有child后按频率清理；partial reset不清global cache；step仍在线程池 | 保留明确所有权分工；reset线程迁移不等于所有渲染调用已有固定线程亲和性，不能称所有线程风险已排除 |
| BaseTask | `_base_task.py:630—663`清camera/robot/scene/renderer/engine引用；scene=None触发Python wrapper的Scene.__del__→C++Scene::clear | Python引用清理不是OIDN内部资源回收证明；C++Scene::clear会逐entity执行onRemoveFromScene |
| SAPIEN兼容Engine | `wrapper/engine.py:39`的create_scene实际返回新Scene，后者建立新的PhysX/RenderSystem | 单纯把Python `Engine()`改成单例不等于复用底层相机/denoiser，不能作为根治方案 |
| svulkan2 OIDN | `DenoiserOidn::init`建立CUDA stream/device，allocate建filter及buffer；free→device→stream有显式释放路径 | 不能说我们已找到某处完全缺失release；要核对真正活跃对象及错误路径 |
| 确定的错误处理缺口 | `init`调用newCUDADevice/commit后不检查OIDN结果即返回成功；`denoise`只日志报错然后继续输出 | 原生失败没有及时阻断使用无效对象；该缺口应修，但fail-fast本身不是泄漏根治 |

SAPIEN安装loader指向2.0.1；同venv仍在役的自有EnvWorker `/proc/maps`也显示OIDN core/API/CUDA三库均2.0.1。Fast-WAM崩溃进程已退出，不能伪称本轮读取到了其/proc/maps。

对应原生路径：[DenoiserOidn](https://github.com/haosulab/sapien-vulkan-2/blob/d8516a4f1467167122ae85f53a8532dbceb1eec2/src/renderer/denoiser_oidn.cpp#L35)、[Scene清理](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/scene.cpp#L110)。

## 5. 我的干净修复建议（尚未实施）

**当前主线：锁定原RGB与依赖，把根因修复放回SAPIEN/OIDN原生所有权与错误传播；旧上层补丁逐块待证、删除无必要改动，不继续堆Python GC、缓存频率和进程重启。**

1. 先做一个与本故障直接对应的隔离生命周期诊断，记录key/device/filter的创建、释放、净存活及调用来源；覆盖真实camera/render/reset/offload路径。目的是决定改哪一个所有者，不是重新跑一晚GRPO碰运气。先给精确预算和停止条件，经批准才运行。
2. 根因修复优先复用上游已审查、且由诊断确认相关的GPU释放逻辑。前轮提及OIDN 2.2.2为验证候选，不是升级决定；按用户本轮保持原观测的约束，**当前路线不整库升级**，优先在2.0.1基线回移必要释放修复并核对RGB。不能盲拷2.2.x不同所有权结构，也不能声称小补丁天然保证画面不变。
3. 同一批次收口`DenoiserOidn`：构造失败不返回成功、不继续建filter/复制输出；已建立资源在错误路径也按有效依赖顺序释放；错误向上传为可识别的原生渲染失败。析构路径不得以抛出异常取代安全清理。不要靠持续打印错误或跳seed继续训练。
4. 制成Fast-WAM独立、版本/hash锁定的原生runtime或wheel，OIDN API/core/device整套匹配；启动验证实际加载路径，**不覆盖Sidney共用venv**，不靠全局LD_PRELOAD混搭。
5. 验收看原生资源在重复生命周期后的平衡、错误是否消失及观测一致性，再进入正式调用链与真实checkpoint恢复；不是只看能启动或跨过Step15。若换库改变RGB，应明确把它当渲染环境版本变更，不能与旧环境无条件拼接比较。

暂不选：全局Ray重启、定期重建整个训练任务、调线程ulimit、关OIDN、减少env/rollout、直接train-env常驻、把Python renderer变单例。它们或不针对key/原生所有权，或改变预算/观测，或把可诊断错误改成隐藏的长期维护负担。

## 6. 本轮取证账本

- 完整读取四个入口、唯一Fast-WAM current SSOT；只沿故障对应源码和前轮证据追溯。
- `sz_oidn_cleanfix_discuss_20260904.sh`：普通账号固定指纹SSH只读身份、code/hash/Git、exact v2 terminal/fatal、resolved参数、实际库依赖。未刷新Sidney训练指标，故本轮不发布新训练状态。
- 官方网页搜索后，用GitHub公开API/raw补齐issue评论、逐提交diff与源文件；未克隆、构建、安装任何运行库。
- `oidn_fix_research_20260904.cjs`获取FAQ关联issue和2.2.1/2.2.2提交；`oidn_ownership_sources_20260904.cjs`核对2.0.1实际对象关系；`oidn_native_follow_sources_20260904.cjs`核对SAPIEN cache/scene与2.1.0变化；`oidn_release_fix_commits_20260904.cjs`读9f816/d4af完整补丁。
- 猜测路径`core/memory.h`返回404后，按2.0.1 `buffer.h`中Memory定义核对；没有依赖不存在的文件。后续检索结果均记录状态，未将无结果当成“上游没有问题”。
- 根据新源码纠正上一轮“clear必然使兄弟renderer失效”和“实际频率8”两项过强/错误表述，保留原事故时间线。只改本地文档/研究采集脚本；未实施候选修复、测试、续训或外部发布。

## 7. 可复核原始证据

- [现场代码、resolved、fatal与库映射](OIDN_CLEAN_FIX_DISCUSSION_20260904.live.txt)
- [issue、2.2.1/2.2.2提交与diff](OIDN_CLEAN_FIX_DISCUSSION_20260904.web.jsonl)
- [OIDN2.0.1所有权源码](OIDN_CLEAN_FIX_DISCUSSION_20260904.ownership.jsonl)
- [SAPIEN cache/scene及2.1.0 diff](OIDN_CLEAN_FIX_DISCUSSION_20260904.native-follow.jsonl)
- [两个GPU释放修复完整提交](OIDN_CLEAN_FIX_DISCUSSION_20260904.release-fixes.jsonl)
- [前轮事故调查与服务器快照](OIDN_RECURRENCE_INVESTIGATION_20260904.md)
