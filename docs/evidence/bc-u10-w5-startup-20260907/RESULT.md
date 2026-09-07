# BC8/U10与DVAC[0,5]切换结果

## 1. 执行结果与配置

2026-09-07 18:37:56 CST只读核验：GPU6/7两项新正式均已加载模型、创建各自workers并进入首轮采集；实际tensorboard配置与预先resolved逐叶差异为空，所查日志无fatal/OOM，driver未退出。只确认健康启动，不宣称首次更新、保存或长程稳定性已通过；按授权未做smoke，确认后结束主动轮询。

| 项目 | GPU6 BC | GPU7 BC-DVAC |
|---|---|---|
| 代码分支 | codex/sz-pi05-online-bc-u10 | codex/sz-pi05-online-bc-dvac-w5 |
| 已推源码 | 3dacac8bcdc43dbace82c8f1f3b87f6318a5d199 | 9ee82771088f61abb3c46bd7de92f09e7bd6c4da |
| driver / starttime | 1690678 / 166401761 | 1693397 / 166403597 |
| namespace / Ray job | RLinf_1 / 46020000 | RLinf_2 / 49020000 |
| Actor / Rollout / Env PID | 1691878 / 1691880 / 1691882 | 1694815 / 1694817 / 1694820 |
| 每轮采集 / 更新 | 8尝试 / U10 | 相同 |
| micro / global | 32 / 1024 | 相同 |
| 成功长度上限 | null，关闭 | 相同 |
| DVAC | 关闭 | bounded_linear，min0/max5，无chunk均重1 |

两项fresh同Sidney π0.5 SFT、空累计成功池/方法历史，100轮；M10/C50/D14、expert-only、LR2.5e-5 constant、无图像增强/demo0、rollout seed42与隔离评估RNG全部沿旧Control。8×1采集，固定32评估按8×4/每5轮，checkpoint每10轮。单卡每次Adam由32个micro累积，每轮10次Adam，合计从累计池有放回抽10240个chunk；池为空沿旧流程跳过。不是每轮遍历成功池十遍。

相对旧运行只改U2→10、DVAC映射及独立路径/名字；新增长度参数保持关闭。新两项之间只有DVAC方法字段、GPU及身份路径不同。种子文件内容逐字相同。完整命令/输出/预算：[正式合同](BC_U10_W5_FORMAL_CONTRACT_20260907.md)；实际配置证据：[startup](bc-u10-w5-cutover-20260907/startup.json)。

## 2. 两个小改动及依据

### 成功长度过滤

新增`algorithm.online_bc.max_success_chunks`，默认null。本轮没有启用3。后续传`algorithm.online_bc.max_success_chunks=3`，只把<=3chunk的完整成功episode入池；4chunk成功整条拒绝，不截断成假成功。参数为正整数；无效值直接报错。

过滤位于累计池入池/落盘前，不更改环境实际成功率、采样或collector。只作用新接纳数据，已有replay不追溯删除。新增过滤计数可保存恢复；关闭时原replay采样RNG/行为保持。仅知道第一次观察到成功的chunk数，不新增加chunk内物理成功步观测。

### DVAC边界参数化和取消局部归一

此前BC只有`w=1+alpha*(z-本chunk均值z)`，因此不能仅把上下界参数改为[0,5]；GRPO已有相应非局部有界映射，但此前尚未接入BC。这次把该映射形式加为BC显式选项，而非移植GRPO损失/优势/优化器。旧chunk_centered默认模式兼容保留，新实验选择bounded_linear。

`w = 1 + (1-min)/z_clip * min(z,0) + (max-1)/z_clip * max(z,0)`。

本轮min0/max5/zclip2，故z=-2,-1,0,1,2对应w=0,0.5,1,3,5；不减chunk均值，不再整体均重归一。高信号chunk可以整段大于1。旧alpha仍为兼容配置字段，在本映射不控制幅度。

保持原action-level V[50]、log(V+1e-12)、过去最多5轮全新有效query标尺（含失败）、clip±2、首轮全1、入池后权重固定；仍将detached w[50]乘原生FM逐动作误差，再按原mask/分母聚合。无额外推理或噪声调用，不改采样/RNG、FSDP、优化器或渲染依赖。

长度上限以后启用时，DVAC标尺仍包含被拒长成功，沿原全新query口径；权重指标汇总过滤前的新成功，接纳与过滤计数另看。强权重不保证收益，均重不再1也可能放大整体loss/梯度；本轮仅验证实现和启动。

## 3. 测试与发布

服务器CPU针对性回归：BC19 passed（11.27s），DVAC44 passed（11.98s）；包含基线、配置、长度<=3边界/默认兼容/整条拒绝/空池/保存恢复、新旧映射/端点/clip/无chunk中心/首轮等权/冻结/全1/partial-mask FM梯度。两组含重叠测试，不称为63个独立用例；没有真实GPU smoke。

生产更改限定为SuccessReplay、BC actor接参、一个默认配置；DVAC另加权重模块分支及10行配置组。代码发布/测试/实际配置/停止/启动的回执在`bc-u10-w5-cutover-20260907/`。源码和旧分支归档均已push并ls-remote核验一致；启动证据随后另一个文档提交，不改变已启动源码。

- 新BC代码：[3dacac8b](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/3dacac8bcdc43dbace82c8f1f3b87f6318a5d199)。
- 新DVAC代码：[9ee82771](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/9ee82771088f61abb3c46bd7de92f09e7bd6c4da)。
- 旧BC结果推回codex/sz-pi05-online-bc：12f5b442；旧DVAC结果推回codex/sz-pi05-online-bc-dvac：5dec6cb3。
- 实施/唯一操作账本：[ledger](BC_U10_W5_CUTOVER_LEDGER_20260907.md)。server stage=`/data/chenyiteng/results/server-maintenance-20260907/bc-u10-w5`；副作用attempt/receipt禁止盲重放。

## 4. 旧实验收尾、ZIP与可视化

| 旧8条/U2实验 | 完成轮次 | 最后训练 | MA10末点 | fixed75 | 最佳固定评估 | 最后文件齐checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| BC | 79 | 6/8 | 75.00%（S70–79） | 20/32 | 23/32，S65 | 70 |
| BC-DVAC[0,2] | 78 | 3/8 | 81.25%（S69–78） | 17/32 | 21/32，S65 | 70 |

主动停止，不是训练自然完成100轮。最后完成轮次高于checkpoint轮次，不将后续未保存权重冒充可恢复点；本轮核对文件集/归档结构，不做恢复测试。旧DVAC训练滑动值略高，固定评估没有显示优势；不同末点窗口已标明。

- [BC轻量ZIP](bc-u10-w5-cutover-20260907/bc-closeout-20260907.zip)、[DVAC轻量ZIP](bc-u10-w5-cutover-20260907/dvac-closeout-20260907.zip)：配置、日志、完整指标、资源、图表、checkpoint清单；不含大权重、replay或视频。ZIP已testzip及SHA记录，未删除旧大文件。
- [简图：MA10＋固定评估](bc-u10-w5-cutover-20260907/plots/brief.png)。
- [展开图：逐轮＋MA5＋MA10＋评估](bc-u10-w5-cutover-20260907/plots/overview.png)。全部滑动窗口只用过去完整轮次，无虚构训练前Step0。

## 5. 共享资源边界

只停止原GPU6/7 driver及其核准namespace/job；停止后其余named actors集合保持，GPU4/5 GRPO driver1082394/start165635111未变，本轮没有改GRPO、共享Ray或其他用户。

18:37:56资源快照：GPU6 28.05GiB、GPU7 25.78GiB（首轮启动阶段，不是训练峰值）；RAM available约1430GiB；/data剩526GiB、/home剩1245GiB。新run写本人/home，分别预留240GiB，同时为GRPO后续留余量；没有删除或移动旧产物。Swap已用约5.87GiB，但RAM available充足，不能仅凭累计swap占用认定正在交换或内存不足。
