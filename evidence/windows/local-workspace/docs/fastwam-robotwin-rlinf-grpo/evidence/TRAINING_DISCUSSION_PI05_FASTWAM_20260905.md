# π0.5接续与Fast-WAM故障、学习问题讨论

2026-09-05；本轮为只读审计与讨论。未停止或启动训练、未修改模型/优化器/环境、未运行新rollout或评估。

## 1. π0.5：等100完成，再从100接到200

**建议等当前100步自然结束；仍值得保持每轮合同接到200，不建议临近终点中断并回到90。**

11:34:51 CST现场：完整Step96/100，第97步采样2/4，wrapper存活；所查fatal/OOM/RuntimeError为0。最新训练167/256=65.23%，MA10=66.56%；最新fixed Step95=22/32=68.75%，全程最好Step70=26/32=81.25%。训练长期上升、近期fixed总体高于早期，但仍波动，没有充分证据说已到平台，也不能由非饱和成功率保证继续训练必涨。

最近10步平均25.318分钟；由Step96完成时间向前推剩余4步，点估计为**09-05 13:04，聊天取约13:05；合理范围12:50—13:30 CST**，条件是后续采样、更新和最终评估保存正常。不是取runner显示的全程平均ETA。原始输入：[最终只读ETA快照](discussion-audit-20260905/sidney_final_eta.json)。

从90重启会丢弃90后尚未进入checkpoint的训练状态，还需新跑110轮；等100再续只需100轮，能保留完整100步曲线及最新Adam状态。resume入口必须是`global_step_100`训练checkpoint目录，不能用`full_weights.pt`冒充完整续训。

部署源码已核对：local-shard恢复模型、Adam动量/内部step、scheduler及actor RNG；env/rollout重新初始化，因此不是逐轨迹等价的一次不间断200步。当前scheduler为constant/warmup0，**100→200仍保持LR5e-6**。准确表述是“方法、每轮采样/优化和评估频率不变，只改总步数、resume入口与新输出路径”，不是所有配置叶字面不变。

额外100步为25,600条训练轨迹、200次optimizer call、640条fixed评估、10代checkpoint；按近期速度粗估约42小时/84 GPU-hours，另有初始化与保存波动。现有一代Sidney训练checkpoint约26.85GiB，若再保留10代约新增268.5GiB；11:34 `/data`尚余约720.24GiB，实际接续前还需看其他任务与保留策略。本轮只是讨论，没有建立或执行续训合同。

精确恢复语义/源码SHA/预算：[Sidney接续审阅第2—6节](../../rlinf-shenzhen-multitask-pi05/evidence/RESUME200_SOURCE_REVIEW_20260905.md)。

## 2. Fast错误：三个不同边界，不能混成一次补丁反复失效

| 阶段 | 实际问题/改动 | 与本次关系 |
|---|---|---|
| 旧有OIDN运行 | `pthread_key_create failed`、invalid handle等原生错误；随后尝试关闭OIDN | 当前train/eval明确none，本run未检出这组OIDN首错；关OIDN会改变RGB和隐式同步时序 |
| 干净noOIDN运行 | Step2取图停滞；确认timeline render漏接既有scene-access fence，可能过早重用另一相机仍在访问的共享场景资源 | 补丁只给该timeline提交补reset/submit fence；保留干净vector_env，无旧Python生命周期补丁 |
| 当前scene-fence-v3 | Step17后出现`PyGILState_Release: auto-releasing thread-state, but no thread-state for this thread`，04:47:11退出255 | CPython在释放GIL状态时，当前OS线程的autoTSSkey中已没有线程状态；属于Python/原生扩展线程状态边界，随后peer关闭是连锁结果 |

**最合理的当前判断：此前未处理到的原生线程/对象生命周期隐患暴露，优先于“scene fence直接制造了新bug”；尚未锁定唯一原生caller。**

支持与限制：

- 干净`SubEnv.reset()`原本已有进程级global_lock。完整未去重栈里3个线程正等这个锁，唯一执行reset的Python线程在读取任务说明，其`close_env()`已返回、`setup_demo()`尚未开始。故“并发reset阶段”不能等同多个scene正在同时析构，也不能再建议给reset加同一种锁。
- `PyGILState_Release`这条fatal精确表示TLS中线程状态为空，发生在后续GIL持有状态检查之前；不是“某处忘了释放GIL”这么简单。
- 11:27实查shim只有一个timeline render导出，没有PyGIL/PyThread/PyEval/pthread导入，原库/shim hash与原锁一致；这降低直接补丁误改Python线程状态的嫌疑，但不能排除间接时序或ABI影响。
- [RLinf官方issue #1040](https://github.com/RLinf/RLinf/issues/1040)已有LingBotVLA/RoboTwin/CPython3.11.14的相同fatal病例，不需要我们的Fast-WAM补丁；它支持平台层已有隐患，不能证明两个病例的唯一原因相同。
- 没有拿到本次原生backtrace/core；Apport目录无本人条目，指定Ray目录的PID直接匹配为空。这不是全盘穷尽检索，也不能由此指认某个C++析构函数。

**我倾向的下一刀是给每个SubEnv固定owner线程，令创建、step/取图、reset、close归属同一个OS线程，保留现有reset互斥和各SubEnv并行。**现有锁只保证串行，不保证线程归属；该候选更贴近本次错误。它不要求降低32环境×8轮预算，也不需要撤scene fence、重开OIDN或堆全局GC。

这仍是工程候选：若真正问题来自原生库自己的后台线程，固定Python owner未必根治。首次验证应覆盖真实reset/close/offload并保留native backtrace，若仍失败就根据实际caller修Ensure/Release或提前删除线程状态的边界；不能拿短测通过当长期根治。本轮未实施候选或启动验证。

完整源码、二进制与调用栈依据：[原生线程状态审阅第2—5节](PYGILSTATE_SOURCE_REVIEW_20260905.md)。

## 3. Fast有无提升：目前没有令人信服的持续增益

当前noOIDN run只有17步：训练首10步均23.13%、最后10步26.33%；fixed Step5/10/15=11/11/13 /32。多成功2条是小幅变化，3个点不足以判稳定上升；首末10步还有重叠。旧有OIDN/128档跑到33步也基本持平，但它与本run起点/观测不同，不能拼成已训练50步。

当前Fast每轮已与两个π系列同为256条轨迹、32个G8组；Fast因C24产生2048个query records且每轮4次optimizer，两个π系列各1024 records/2次。**不能继续用旧128档“组数只有一半”解释当前运行。**不同query仍可能共享同一轨迹成败，并不自动带来更多独立outcome。

三个任务分别是Fast/stapler、Sidney/pillbottle、旧π0/adjust_bottle，SFT与动作表示也不同；绝对成功率不能归为纯模型或GRPO比较。Fast standalone的11/16没有对齐当前192-action/noOIDN/fixed32协议，也不是本run Step0。

## 4. 全面核对后，我怎样排序学习问题

| 优先级 | 候选与机制 | 目前证据与下一步方向 |
|---|---|---|
| 1 | **实际权重更新受BF16精度限制** | Fast整个1.021B action参数及Adam一二阶矩均BF16；选定6张量到Step10/40次优化后，参数大量不变而动量非零。优先考虑可训练action保留FP32主参数/Adam、BF16计算与更高精度累积，而不是直接调大LR |
| 2 | **真正有信息的G8组偏少、有效更新被稀释** | 全成/全败组不给相对优势；总体25%成功率不证明32组都有混合成败。现日志没有有效组计数，MB2还使全mask微批次的0值稀释logged ratio/KL；应先看真实保留组和valid加权变化 |
| 3 | **探索与模型可塑性、任务闭环不匹配** | Fast shift5/M10/noise0.3/C24、冻结proprio；π系列noise0.5、C50及不同可训练投影。相同noise数值不代表相同物理探索；先看抓取/搬运/放置失败阶段，不能直接照搬0.5 |
| 4 | **当前协议下SFT起点与观测分布** | noOIDN改变RGB、192-action与旧standalone不同；可能影响基础能力。但旧OIDN运行本来也未稳升，不能把全部问题归因关降噪 |
| 5 | **事故截断有效学习预算** | 当前只4352条轨迹/68次optimizer、旧128档33步也只有4224条/66次，均受中断；还不足以判上限。稳定性修复恢复的是可用预算，不自动解决学习机制 |

这里是排查价值排序，不是假装已有各根因概率。第1项新增实测如下；参照是同一release SFT先cast到实际BF16起点，按DCP元数据offset只读相关分片，未运行模型/optimizer/GPU。

| Step10，40次optimizer后的抽查张量 | 元素改变比例 | 相对L2变化 | Adam一阶矩非零比例 |
|---|---:|---:|---:|
| 动作输入投影 | 0.753% | 0.00173% | 100% |
| 动作输出head | 10.756% | 0.10881% | 100% |
| 时间投影第0层 | 28.990% | 0.17071% | 100% |
| 第0层attention Q | 4.727% | 0.02704% | 100% |
| 第29层attention Q | 5.789% | 0.03245% | 100% |
| 第0层modulation | 1.855% | 0.00287% | 100% |

共659万元素、约占action参数0.65%，DCP仅读39.6MB；不是整模型变化率。它支持“更新落地稀疏，精度很可疑”，**不证明所有未变权重本应变化，也不能单凭净变化排除抵消或确认无增益唯一根因**。两个π系列同样有大量BF16参数，但norm/AdaRMS、action/time/state等通路有FP32；不能说π全是FP32。底层dtype合同见[PyTorch 2.11 MixedPrecisionPolicy](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy)。

## 5. 已核对、当前不支持的“接入bug”说法

- 官方预处理/norm与一次decode、H32预测/C24执行、保存真实stochastic transition、旧/新logprob重放、GRPO组优势与mask、optimizer.step和权重同步路径未发现明显错接。历史08-31真实parity/strict-resume验收是已有证据，不是本轮重做。
- Fast **也使用batch共享denoise index**；旧review关于逐sample选index的说法应由本轮实际源码纠正。
- MB2没有代数上让梯度少16倍：Fast每rank累积256次、π系列16次；总均值由完整累积恢复。但日志total_loss记录在除256/16后，所以不能拿绝对total_loss横比。有限精度累积是另一项真实待查问题。
- Fast logged ratio约0.85不是valid样本真实ratio必然0.85：全mask微批次返回0后参与普通平均，现日志不能离线精确还原有效加权值。
- 未发现只训练824个元素：824是可训练张量名称数，实际约1.021B元素。scene-fence本身也不改模型或GRPO loss。

完整配置对比、dtype分类、调用链、日志聚合公式与全部排序依据：[学习源码复核第3—7节](GRPO_LEARNING_SOURCE_REVIEW_20260905.md)。

## 6. 操作与产物索引

- 根规则/交接与两专题SSOT先读；并行子审阅只读本地源与官方资料，服务器访问由主agent统一固定指纹Paramiko完成。
- 11:20/11:23读取3run完整resolved、TB、日志、checkpoint元数据、64个源文件与4树HEAD/dirty；远端无rg导致第一批提前结束后，改为Python文本检索。认证前banner EOF由既有helper限定重试后成功，未切换认证路线。
- 大stdout经终端转存被截断，保留第一批完整runs段，随后改为SSH直接写本地文件并用压缩流完整取回；最终JSON2.74MB、64源码，已成功解析。不将截断片段当全源证据。
- 11:27按DCP metadata/FakeTensorMode读取保存dtype、同版本so hash/导出、实际RoboTwin指令loader与shim源码。没有恢复模型训练，也不把FakeTensor元数据读取当恢复测试。[原始数据](discussion-audit-20260905/native_and_dtype_metadata.json)。
- 11:32在服务器CPU只读SFT mmap和6块DCP权重/动量，未改参数文件或调用optimizer。[参数差异证据](discussion-audit-20260905/weight_delta_probe.json)。
- 11:34刷新Sidney进度、TB与ETA，得到当前完整Step96；此前11:23 driver96/TB95是文件刷新时序差，不拼接伪造标量。
- 主原始快照：[live_snapshot.json](discussion-audit-20260905/live_snapshot.json)；精确采集命令分别保存在本地`local_scripts/remote_commands/sz_training_discussion_readonly_20260905.sh`、`sz_discussion_metadata_readonly_20260905.sh`、`sz_fastwam_weight_delta_readonly_20260905.sh`与`sz_sidney_final_eta_readonly_20260905.sh`。
- 本轮停点：讨论结论与证据已整理，不停止Sidney，不重启Fast，不干预另一窗口BC或共享Ray。本轮新修复/续训均未执行。
