# FQL真实执行数据：需要保存什么，现有代码缺什么

2026-10-25。仅契约与实施切口；采集尚未实现/启动。与[46模型方案](46_FQL_FIXED_TEACHER_PLAN_20261025.md)、[48预算](48_FQL_COLLECTION_AND_DATA_MIX_20261025.md)配套。

## 实际代码核对

已核当前`tools/pi05_policy_eval/run_batch.py`、`protocol.py`、`robotwin_env.py`以及服务器原生`RoboTwin-RLinf-support/envs/_base_task.py::gen_sparse_reward_data`。

- `run_batch.py`读最终成败；`obss, _, _, _, _ = env.chunk_step(...)`丢掉奖励、终止/截断和info。`recording_mode=none`不会自动留下可训Q的数据。
- 评估设置`auto_reset=false、ignore_terminations=true`；包装器可能把终止标记清零，不能直接当作Q的继续标记。
- `_cal_chunk_rewards`中的`n_steps_to_run`固定为0，会把成功放到矩形块末尾；不能把它解释为真实首次成功的物理动作位置。
- 原生控制器一次收到整个前缀，先做TOPP轨迹插值再执行。为“记录逐步奖励”而把C10拆成十次单步，会改变控制行为，不采用。
- 原生`take_action_cnt`在执行前增加提交前缀长度；成功可能在内部控制循环中提前发生。因此计数差表示提交的模型动作点数，不是实际完成的物理控制步数，更不是精确的首次成功动作位置。

结论：沿用既有完整C10调用，在原生返回/观察提取这一层捕获转移；独立开启`collect_transitions`。不打开U、四噪声、注意力或完整录像去间接凑字段。

## 每个决策块的数据契约

一个转移的意思是：在观察s下发起一个长度不超过10的动作前缀，执行或因成功提前结束，获得奖励和真实后继观察s'。

|字段|保存要求|用途|
|---|---|---|
|task_id、task_name、instruction|全局50任务表＋本回合实际指令|多任务条件与分层统计|
|episode_id、requested_seed、actual_seed、native_retry|实际重置成功的种子必须保留；同任务实际种子去重|复现、识别原生重试和训练/测试泄漏|
|当前/下一观察|头部＋双腕RGB、14维关节；同一回合连续边界|Q不能只读当前图，也不能拿reset后的图当s'|
|policy/model/config版本|原始权重、归一化、H/N/C、控制器/资产版本、官方step_limit|定义数据来自什么行为与动作接口|
|predicted_action_h50|原始模型预测的归一化H50动作，小体积，保留|核对推理/执行映射；不是老师答案训练缓存的前提|
|command_c10、mask、submitted_length|实际提交给控制器的绝对关节前缀及同坐标归一化副本|Q的动作输入；末块不足10时mask，非执行的H50尾段不入Q|
|reward_raw、success_before/after、reward|保留原生返回；本方案只在首次成功的决策转移记1，其余0|避免成功后重复记1|
|terminated_native、truncated_native、termination_reason|分开成功、上限、其他自然结束、错误|保留环境事实，不把不同原因揉成一个done|
|bootstrap_mask、remaining_actions_before/after|依据下述有限预算目标派生；保存口径版本|Q目标与有限时域状态|
|step counters、query_id、noise_seed|提交数、块序号、推理噪声种子；必要时保存初始noise数组|追踪时序；不要伪称提交数为物理步|
|valid、error、episode_success|错误回合单列；成功/正常失败均有效|只筛数据完整性，不筛掉失败行为|

训练不依赖旧老师每一层中间张量或全部去噪轨迹；同噪声老师/学生目标可在训练时即时生成。旧U采集NPZ和成绩CSV的字段不能未经核对就认定满足本契约。

## 奖励、终止和折扣的明确选择

本版拟学习“在该任务官方动作预算内能否成功”的有限时域目标：首次成功奖励1；正常失败或预算用尽为0，episode结束，不再往后估值。Q输入包含剩余预算；成功/预算末尾的`bootstrap_mask=0`。原始环境的`truncated_native`依然原样保存，派生mask另列，不修改原生标记。

这与“只是采集器临时截断、底层任务仍可无限继续”不同。后者应保留bootstrap，不能一见truncated就设0。[Gymnasium对有限时域与外部截断的区分](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。

按一次决策定义gamma=.99；目标为`r + gamma × bootstrap_mask × Q_target(s',G(s'))`。一次完整C10和末次短前缀都属于一次决策；终止转移没有后续项。此次不声称知道成功发生的精确物理动作位置，不构造未经记录的逐物理步折扣奖励。

环境报错、设备错误、缺末帧或进程超时，属于invalid；不能拿它们的未完成记录补成“失败终点0”。已原子提交的完整回合保留，未完成回合隔离，日志标原因。已经成功的slot不再写重复的吸收态转移；提交长度0不写。

## 存储与开关

建议每回合一个分块HDF5文件，完成后由`.partial`改名；并行worker各写自己的文件。约4800个回合文件，避免为每个动作/数组生成数百万小文件。

- 观察序列只存T+1份，相邻转移按索引共享，不双份保存obs/next_obs。
- 三相机存模型实际所见的224×224 RGB uint8，使用原来的resize/pad、相机顺序和颜色约定；不能另选一种缩放后声称等价。若在预处理后截取，读取器需避免重复归一化或改变图像处理。
- 实施smoke要核对“原推理输入”与“落盘后重载输入”一致；未通过时保留原分辨率决策帧并重新估容量，不能把100GiB估计当承诺。
- 轻量无损压缩，保留关节/动作float32与布尔标记。无整段视频、无observer、无逐物理步画面/控制日志、无U/DV/四噪声/Attn；这批目的是训练Q。
- 配置明确`collect_transitions=true、recording_mode=none、record_full_video=false、record_signals=false、record_decisions=false`；旧`collect_data`专家采集器开关不是这个新功能。

## 实施时最少验收

在已验收的8环境规模开始，不从1环境逐级试：取容易成功和常有失败的代表任务，尽量覆盖成功、正常上限及末次短前缀。检查保存后重读、首末状态、奖励总数与原生成败、T+1/T维度、动作mask及种子。回合不足以覆盖边界时使用CPU合成边界测试，不无限重跑等成功。

只新增一个原生返回捕获/转移writer接口，旧评估未开开关时行为保持；模型常驻、异常跳过与精确清场复用原调度。若有实现失败，优先修本小接口；不要重写推理、把全任务采集当smoke。
