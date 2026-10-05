# RynnValue 接入 RoboTwin WMRL：接口、顺序与待验证项

2026-10-05 调研时计划。后续已获授权并实现、实测；原文以下保留规划依据。**当前执行与失败诊断以[实施记录](rynnvalue_execution_20261005.md)为准**：Success接口已跑通，专家控制样本全No，正式尚未启动，正在单卡诊断。这里的“未实现/训练保持”等描述仅指16:12调研时点。

## 1. 决策与范围

先接 **视频 → 批量 Success 判断**，用于同一批原生/生成视频的旁路复核；验证判定质量与吞吐后，再接环境的 reward/done。保留 adjust_bottle、π0.5、OpenDW、GRPO、N64/G8/R8、C32/max384；第一版不引入剩余时间 shaping、量化、换任务或新策略算法。

RynnValue 不需要动作或14D state才能评分，因此本次不用重做策略/WM动作适配。它也不能替代 WM 动力学：即便准确识别了视频中的成功，也不能证明相同动作在物理仿真里成功。

配套：[batch与资源审查](rynnvalue_batch_audit_20261005.md)、[旧候选比较](candidate_comparison_20261005.md)、[当前奖励饱和证据](reward_saturation_discussion_20261005.md)。

## 2. 两边目前是什么接口

| 环节 | 已有 | 本次要接的切口 |
|---|---|---|
| OpenDW | 输入三图拼图、C32×14D动作、state，输出当前帧＋8张未来帧 | 直接取无损RGB；主图256×320，动作时刻0,4,…,32 |
| WM服务→env | 只传末帧三图和8个小RM分数 | 新增独立clip/Success通道，不能把一个Yes复制成8个帧分数 |
| Rynn旧服务 | 单轨迹/K4、逐前缀数值评分，返回剩余时间/差值 | 新增批量Analysis生成，解析Success；旧RA-BC数值入口保持 |
| env | 任一帧小RM≥0.9则在C32末done；奖励为分数差 | 后续明确区分成功判断、训练奖励、超时和模型错误 |
| GRPO | 有效reward求和，按G8均值[0.1,0.9]过滤 | 接新奖励时复核有效组，保留组内完整性；unknown不能伪装成奖励0 |

代码定位：`local_patches/opendw_smoke_20261003/batch16/opendw_service_batched.py:510`（主图评分）、`:548`（返回末图）；`multigpu/rlinf/envs/world_model/opendw_adapter.py:110`（reward/done）；`multigpu/rlinf/envs/world_model/opendw_robotwin_env.py:229`（请求）。

## 3. 第一版输入输出合同

独立使用现有 Rynn venv/权重，通过本机 HTTP 或离线 manifest 接入，不把 Rynn 依赖合并进 OpenDW 环境。新接口建议 `score_success_batch(items)`，外层可收512条队列，内部按测出的RM microbatch分批。

每条输入：

- `episode_uid`、`end_action_idx`、`source`（native/wm）、`instruction`、机器人/相机说明。
- 按时间排列的主相机RGB帧及实际动作序号；从起点到当前的历史均匀取K8，含首末帧。不足8帧的重复规则固定并记录。
- `sampling_version`、`camera_layout`、输入哈希。原生success标签仅供事后评测，**不能送进模型提示**。

每条输出：

```json
{
  "episode_uid": "run/rank/env/reset-generation",
  "end_action_idx": 128,
  "success": null,
  "match": null,
  "parse_status": "missing_success",
  "raw_text": "...",
  "model_fingerprint": "...",
  "input_hash": "...",
  "latency_ms": 0
}
```

`success`只取true/false/null；不是校准过的概率。只有完整Success字段才解析Yes/No；不能用“文本包含Yes”、Match Yes或剩余时间接近0代替。字段缺失、冲突或生成截断记unknown。保留提示、采样、processor、attention和源码版本；缓存身份包含这些变化。

先保留官方Analysis提示、greedy生成和128新token上限；Success在描述和Match后面，先不为省时强改成只答一个词。仅需Success时直接generate，无需另算全轨迹剩余时间曲线。生成期间模型内部仍可能执行其价值头，不能称“价值头已经关闭”。

## 4. 历史记录是当前明确缺口

现env只有当前图/state，没有完整历史；现保存媒体是早期带配额的C32片段，不是当前每轮512条完整视频。可用它们测加载/接口，不能冒充完整轨迹准确率或最新策略代表样本。

`reset_id`是50份起点之一，会重复；env slot也会重复。需新增 `run_id + rank + env_index + reset_generation` 组成的episode UID，每块带起止动作序号。reset清历史，保存恢复同时保存UID与历史/不可变文件引用。G8只共享起点，分叉后各自保留历史，不能共用另一个成员的评分。

建议历史归env管理：服务按需返回8张主图或不可变CPU文件引用，env追加；评分请求自包含选出的K8图像，Rynn服务不保存跨episode隐式历史。整段按stride4保存最多97张主图，N64约1.42GiB未压缩RGB（不计副本）；逐R消费/释放，不把所有轮留在内存。若将来需三视角，另算体积。起点＋各块边界去重，不重复插入上块末帧。

在线真正需要的改动是episode身份、短历史、独立Success接口和阶段装卸；不是合并三套完整训练框架。

在线控制点已核清：`opendw_robotwin_env.py:239–245`的WM响应到达后建立clip，在`:245–262`提交reward/terminated之前调用Rynn。rank0/1的CPU env分别请求固定物理4/5卡的外部Rynn服务；不能在绑WM6/7的env进程里直接用cuda:0加载。采样末的env offload要同时等待Rynn卸载，让runner原屏障覆盖它，再进入actor训练。服务PID、独立端口、GPU映射与清理加入原owner；现有owner有WM专用假设，需要显式区分服务类型。

## 5. 实验顺序与输出

1. **离线接口与判定核验**：先用带物理仿真success标签的原生成功/失败视频，并列加入生成视频。核原生误报、漏报、unknown；生成视频只报告新旧裁判分歧及人工可判断性，不把Rynn当真值。现存媒体不完整时，补有明确episode身份的小样本；不从旧配额片段拼出未经证实的完整轨迹。
2. **一次有代表性的batch测试**：同K8/分辨率/提示，以少量B1为参考，测B4/B8/B16及尾批；用512条队列测总耗时，分别报告首载、预热吞吐、生成长度、峰值与装卸。若重复输入用于吞吐，明确标注，不能充作512个独立质量样本。
3. **再选择在线模式**：先确认Success能区分正负、没有严重批处理错位/unknown，随后在独立实验接下面的reward/done合同。语义测试与吞吐测试分别报告；不以“能跑/有梯度”代替判定可靠。

这份文档是执行前计划，没有承诺B16可用、具体时长或判定准确率。显存和速度测出来后才能给整轮额外成本。

## 6. 上线模式：需要讨论的是奖励定义

| 选择 | 怎样工作 | 得到什么/限制 |
|---|---|---|
| 旁路复核（首步推荐） | 同视频同时记录旧RM与Rynn，原训练不改 | 回答旧RM是否可能误判、Rynn是否适合；本身不改善训练 |
| 候选成功复核 | 旧RM提出成功→Rynn Yes才在本C32末成功；No继续 | 减少候选中的误报；旧RM从未报出的成功仍可能漏掉；反复拒绝会多次调用 |
| Rynn主裁判 | 每C32末判断；首次Yes给reward=1并done，其余0，384超时 | 成功与学习目标一致；更多生成调用，稀疏信号也可能全成/全败 |

**推荐先旁路；若准确性合格，再比较候选复核与Rynn主裁判。** 候选复核若只改done、仍保留旧分数差奖励，策略仍在优化旧RM，不能声称解决了奖励饱和。因此真正用来训练时，建议另开明确的“首次确认成功=1”稀疏奖励实验；不要暗中混加两个RM。

二值reward下，G8均值0.1–0.9会保留1–7条成功组，滤掉0/8和8/8；数学上吻合GRPO，无需新增critic。但若仍几乎全Yes，依旧没有有效组内区别，大模型不会自动解决饱和。

unknown先重试一次；仍unknown的在线样本应标记无有效标签，并按完整G8排除，而非当作确定失败。当前框架尚无这条新validity通道，需要与接入一起实现。HTTP失败是服务异常，不应伪装为Success No。候选被拒后必须在env提交done之前继续原轨迹；事后改掉已结束轨迹的标签无法恢复后续探索。

## 7. 视角与成功语义仍需验证

- 当前原生摆瓶子success检查指定侧的位置及瓶子功能点高度；Rynn只看RGB和文字，看不到几何真值。提起的视觉印象未必达到阈值。
- 首版用主图方便对照旧RM；瓶子出主图、腕图还能看见的样本必须单列。三视角拼图可后续对照，不能把三相机串成三个时间点。
- K8均匀历史可能漏掉短暂成功；官方Success文本对“曾成功”和“当前仍成功”的实际行为要与原生锁存标签核对，不凭提示想当然。原生视频的任务真值与“这8帧能否看清”分开记录。
- 即使两个裁判都判成功，WM也可能把不合理动作画成成功。原生固定seed评估仍是最终依据。

## 8. 本轮状态与下一入口

16:12固定host-key、普通账户只读核验，读取了9个配置/源码文件；无模型加载或远端写入。实际Git `10e0d333f5f3811d0d130587e50f1faf48da49e5`；HF锁定`8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e`；Transformers4.57.6、Torch2.7.1。本次读取的7个HF小文件hash与旧锁一致；本轮没有重读19GB权重校验内容。

下一步入口：新独立Success批服务＋带标签视频manifest；先完成离线batch/语义结果，再决定在线奖励模式。本文和batch审查仅本地维护，未部署、未宣称已推Git。
