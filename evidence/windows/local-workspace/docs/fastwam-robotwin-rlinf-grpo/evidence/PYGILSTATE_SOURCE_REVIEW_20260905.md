# Fast-WAM PyGILState_Release：原生线程状态与修复范围审查

日期：2026-09-05。范围：现存日志/本地锁定源码、官方源码和issue只读审查；已接入主审计11:23:45—49现场补证。本审查没有运行GPU测试、修改服务器或重启训练。

## 1. 直接答案与归因顺序

**当前最有依据的解释是RoboTwin联合原生运行时的线程状态/对象生命周期隐患在长程运行中暴露；还没有证据把它指定为scene-fence补丁引入，也不能指定某个SAPIEN析构。**

三次处理面对的是不同边界：

| 阶段 | 已观察问题 | 对应改动 / 本次关系 |
|---|---|---|
| 旧OIDN run | OIDN `pthread_key_create failed`先出现，随后invalid handle和Python线程状态fatal | 关闭OIDN移除降噪执行；本run未检出该首错链 |
| clean/noOIDN v1 | Step2取图无限等待；实际svulkan2 timeline提交漏接已有scene-access fence | 补的是不同相机共享场景资源的GPU在途访问顺序；这不是Python线程状态修复 |
| scene-fence-v3 | 完整Step17后，Step18 reset阶段EnvGroup rank0首先发生`PyGILState_Release` fatal | 解释器发现当前OS线程的自动线程状态映射不存在，直接终止EnvWorker；后续NCCL/Gloo/Ray断连是传播结果 |

旧问题并未全部被证明根治；这次fatal是本条noOIDN/fence运行中新出现的失败表现。不能把“新出现的表现”等同“我们新制造的缺陷”。官方RLinf已有同名fatal和几乎相同RoboTwin reset栈，见§4。

## 2. 精确错误是什么意思

现场Python栈明确使用CPython **3.11.14**。已按该tag读取[`PyGILState_Release`实现](https://github.com/python/cpython/blob/v3.11.14/Python/pystate.c#L1744)：

1. 先用`PyThread_tss_get(&runtime->gilstate.autoTSSkey)`取**当前OS线程**的自动Python线程状态。
2. 返回NULL时立即触发本次fatal；之后才检查这个状态是否为当前持GIL的状态。

所以本次错误不是简单的“多个线程抢GIL”“忘记释放GIL”或“GIL锁超时”，也不等同旧`Couldn't create autoTSSkey mapping`。更具体地说，是某个原生调用路径要释放一份Python线程状态，但该线程已经没有它应有的自动映射。与之相容的机制包括：Ensure/Release不在同一线程或不配对、外层Release之前内层路径删除了状态、线程局部状态/内存被破坏。仅凭错误字符串不能在这些机制中唯一选择。

同版本[`tstate_delete_common`](https://github.com/python/cpython/blob/v3.11.14/Python/pystate.c#L1074)确实会清掉匹配的autoTSS映射。这提供了“提前删除状态后外层再Release”的具体代码可能性，不证明本次已经走中。故障记录是`Python runtime state: initialized`，没有依据套用解释器整体退出时的finalization故障解释。

## 3. reset线程池不等于同时重建多个scene

本run使用干净RoboTwin `f3e30a83365c`；11:23现场取回的[实际vector_env.py](discussion-audit-20260905/sources/robotwin/robotwin/envs/vector_env.py) SHA256=`dc55a587850ba58b3963e58173c2b6d952eb61e7a48cfb56b7874917b76a3c96`。其`SubEnv.reset:141–177`入口已有：

```python
with self.global_lock:
    with self.lock:
        ... close_env(...)
        ... create_instruction()
        ... setup_demo(...)
```

`VectorEnv`虽将每个reset投给线程池，但场景关闭/重建在进程级锁下串行。其他线程停在reset入口，很可能只是等待锁；不能据多条reset栈声称多个scene正在同时析构。**再加一把reset全局锁不是新的修复。**

这把锁也不等同“对象在同一个固定OS线程创建、调用和最终析构”：线程池任务可能由不同线程接手，原生库还有自己的工作线程。若实际证据指向线程归属，需修的是所有者与回调/清理边界，而非只把已串行的reset再串行一次。

`Base_Task.close_env()`会依次移除camera、robot、scene、renderer、engine引用，再按开关清资源cache。仅看该Python顺序，不能保证每个C++对象恰在当前行最终析构，也不能把`clear_cache()`当成强制销毁所有renderer。此前旧Python生命周期补丁没有进入本run，不建议在未定位此次原生调用者前整包套回。

### 3.1 未去重fatal原文进一步确认了什么

[完整driver首错原文](discussion-audit-20260905/fastwam_fatal_raw.txt)第151行开始fatal；此前第150行仍在Step18 rollout第1/8轮之后。Python栈显示：

- 第338/387/425行是3个线程停在`SubEnv.reset:142`，即等待`global_lock`。
- 第376—383行是唯一已进入reset的Python线程`0x7ef23f7fe640`，位于`create_instruction → load_task_instructions:169`；按实际源码顺序，它的`close_env:147`已经返回，`setup_demo:158`尚未开始。
- 第462—471行是环境控制线程在`bootstrap_step → reset → future.result`等待。
- 第154—155行存在一个`<no Python frame>`线程，但全文没有`Current thread`标签。**不能把列表第一条无帧线程就指定为fatal调用者**，也不能把正在读说明文件的线程直接指定为罪魁祸首。

因此可将失败时间边界收窄为“训练bootstrap reset中，已完成某次Python层关闭、正读任务说明时，有原生线程状态fatal”。它不支持“多个scene同时析构”“卡在新scene创建”或“fence wait再次挂住”。异步原生清理仍是候选，但需要native caller证明；`close_env`返回不保证所有原生后台清理已经完成。

`coredumpctl`工具不存在，系统`core_pattern`走Apport。11:27追加只读检查在`/var/crash`没有找到本账号条目；实际shared Ray日志目录已从gcs命令行定位，但该目录直接匹配PID1569541的err/out没有命中。**本轮没有取得可用core/native backtrace**；这不能升级为服务器所有路径都没有保存过原生证据。

## 4. 原生绑定与已有上游病例

锁定SAPIEN `d8228489`的[`cmake/pybind11.cmake`](https://github.com/haosulab/SAPIEN/blob/d8228489d05775b8615ef3edd1d47fadf25d6d7a/cmake/pybind11.cmake#L6)指定pybind11 **`06e8ee2e357fc2fd6e36de431fa0ca0049aafc7d`**；该源码版本标记为3.0.0.dev1。运行环境里`pip show pybind11`的版本不能代表已经编进SAPIEN wheel的头文件版本；上述构建锁仍需与实际二进制元信息对应，不能据此直接断言wheel存在某个已修复bug。

11:27实际磁盘二进制补证：[native metadata](discussion-audit-20260905/native_and_dtype_metadata.json)。SAPIEN package=3.0.1，安装的pybind package=3.1.0；实际`pysapien.so`包含`__pybind11_internals_v7_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1__`，SHA256=`d542454cfe57c593ac79039d6a43daa518af7f56302a99e48206dcd1dc731070`。ABI字符串不能反推出唯一pybind commit，但确证不能把pip3.1.0直接当它的构建版本。

同批核对原svulkan2 SHA仍`972eff8fc5fedfd59d4cd8794039d9bc850847a6c74dfbf7d549eff9d041b1b7`、shim仍`45e6cac35ea2edea0724fa4cd92ec82cab401b6abd6f8001805d22306f375df0`；shim只导出timeline `RTRenderer::render`一个符号，动态导入中没有PyGIL/PyThread/PyEval/pthread匹配。当前生产源码也与两处fence补线相符。这削弱了“shim直接错误操作Python线程状态”的猜测，不能据此完全排除ABI或时序改变带来的间接影响。

该锁定pybind中：

- [`get_internals()`局部RAII](https://github.com/pybind/pybind11/blob/06e8ee2e357fc2fd6e36de431fa0ca0049aafc7d/include/pybind11/detail/internals.h#L430)显式成对调用`PyGILState_Ensure/Release`。
- [`gil_scoped_acquire::dec_ref`](https://github.com/pybind/pybind11/blob/06e8ee2e357fc2fd6e36de431fa0ca0049aafc7d/include/pybind11/gil.h#L93)在自己拥有的状态计数到零时调用`PyThreadState_DeleteCurrent`。
- [`functional.h`](https://github.com/pybind/pybind11/blob/06e8ee2e357fc2fd6e36de431fa0ca0049aafc7d/include/pybind11/functional.h#L38)对Python回调句柄的析构也会获取GIL。

这些路径解释为何C++回调/最终析构值得调查；**这些代码本身并不是本案缺陷证明**。目前不能排除Torch、Cython或其他扩展库是实际Release调用者，不能因为附近有reset就默认caller为SAPIEN。

更有用的外部旁证是[RLinf #1040](https://github.com/RLinf/RLinf/issues/1040)：LingBotVLA/RoboTwin在CPython3.11.14的reset线程池中已报告同一fatal，根本不依赖本项目Fast-WAM scene-fence补丁。[后续评论](https://github.com/RLinf/RLinf/issues/1040#issuecomment-4250491839)指出降环境数对某人的评估有效，但另一人的训练即使env=8仍复现。因此，平台既有隐患优先于“Fast GRPO公式导致”，降并行不能作为已知根治办法。

pybind11检索出现的Python3.14自由线程/子解释器PR属于不同运行时；本机是普通3.11.14，没有依据照搬那类修复。

## 5. 我建议的窄修方向

**我最倾向的下一刀是环境对象的固定线程归属：每个SubEnv由一个固定owner线程负责创建、step/取图、reset和close，保留现有reset互斥，避免共享线程池把同一个原生场景的生命周期交给不同OS线程。** 这是一项具体的工程修复候选，仍不是已经证明能根治本次fatal的补丁。它处理“虽然串行，但对象来回换线程”这一当前实现确实允许的边界；比再加global_lock、GC或重新打开OIDN更贴近本次线程状态故障。

该候选可以只落在Fast隔离RoboTwin的环境调度/生命周期代码，保留32个环境、rollout8、256轨迹和全部GRPO参数，step仍可在不同SubEnv的owner之间并行；不应把整条训练改为单环境或偷偷缩小预算。所有使用该对象的入口必须统一，不能只把reset挪到某线程却继续在任意线程close。原生库自建线程若仍错误管理Python状态，固定外层owner也可能不足，这时需修实际native caller。

由于现有日志尚无native caller，实施这一候选时应把故障处原生backtrace与owner线程身份记录一起放进一次真实reset/close/offload的有界回归。若出现fatal，即按调用者修Ensure/Release或析构，不再叠第二套猜测补丁。当前不建议直接拿另一个100步GRPO作诊断，也不建议把降低env当根治。

若后续找到core，先直接读取core定位`PyGILState_Release`上一层。保持noOIDN和已确认的fence补线，让工程回归回答线程状态在哪一层丢失。此类新增代码/GPU复现需另按任务授权执行，本轮未执行。

针对定位后的修复形状：若是Python对象回调或析构跨线程，确保Ensure/Release在同一线程严格配对、Python对象在有效线程状态与GIL下销毁，必要时由固定owner线程承担相关创建与清理；若是线程状态提前删除，修对应嵌套计数/状态恢复；若是内存或TLS键破坏，回到实际破坏调用者，不能靠额外GIL释放掩盖。

不建议当前直接盲加`gil_scoped_release`、全局GC、重装/整库升级、再次串行reset、恢复全部旧Python补丁或撤回fence。scene-fence函数只补两处既有Vulkan同步操作，没有Python C API；这降低其直接造成线程状态fatal的解释力，但改变时序或ABI风险仍不能在没有原生栈的情况下完全排除。

## 6. 证据与本地操作账本

| 操作 | 结果 |
|---|---|
| 读取PROJECT_CONTEXT、HANDOFF、当前Fast SSOT、失败边界、scene-fence账本与本地shim | 完成；未连接或改动服务器 |
| 审阅现存失败抽样脚本 | 原脚本按关键词截片并去重，不能替代完整有序fatal栈；已请求主审计定点补取原始driver/worker err |
| 官方CPython、SAPIEN、pybind源码与RLinf issue只读检索 | [第一批来源](PYGILSTATE_PRIMARY_SOURCES_20260905.jsonl)、[锁定跟进源码与issue评论](PYGILSTATE_PINNED_FOLLOW_20260905.jsonl) |
| 本地源码读取辅助脚本首次执行 | PATH上的旧Node先受realpath限制、stdin路径后发现无fetch；改用既有bundled Node经stdin成功，未安装依赖或改变服务器 |
| 个别猜测pybind文件路径404 | 随后Git tree定位实际`python/pybind/sapien.cpp`等文件再读取；404未用于推断代码不存在 |
| 主审计11:23现场补证 | 已读取[live snapshot](discussion-audit-20260905/live_snapshot.json)、完整fatal与实际RT源码；确认reset已有全局锁、3线程等待、1线程在close返回后读取说明；尚无native caller |
| 主审计11:27追加原生元信息 | 实际ABI、原库/shim hash、唯一导出/无Python导入已核对；未取得core/native caller，不把空检索冒充穷尽排查 |

旧抽样首错入口：[失败边界](FASTWAM_EXIT_BOUNDARY_20260905.data.txt)；新结论优先依据上方未去重原文和实际源码。
