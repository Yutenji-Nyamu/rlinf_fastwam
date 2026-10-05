# click_bell、Robometer、RynnValue：接入距离与可靠性

后续选择：用户已选先试RynnValue；16:12只读完成实际版本batch/Success接口审查，见[具体接入计划](rynnvalue_integration_plan_20261005.md)与[批量成本](rynnvalue_batch_audit_20261005.md)。尚未实现或切换RM。下面保留候选比较依据。

2026-10-05讨论更新。15:54固定host-key、chenyiteng只读核验深圳3指定资产；没有下载权重、加载模型、占用GPU、改训练或切换任务。

`click_bell`是换任务并复用其任务RM；Robometer和RynnValue是换/增加奖励判定器。两种选择可以组合，不能按同一种“模型替换”理解。

## 紧凑比较

| 维度 | click_bell＋WorldArena小RM | Robometer-4B | RynnValue-8B |
|---|---|---|---|
| 输入与输出 | 主图＋任务文字→连续成功分数；要先核对应checkpoint结构 | 任务文字＋视频/帧序列→progress、success概率、轨迹偏好 | 任务文字＋图像序列/机器人相机说明→剩余时间、相对时间；语言生成另给Match和Success Yes/No |
| 已公开内容 | 对应RM权重约588MB、模型/训练源码、RoboTwin原生任务、RLinf任务示例 | 权重、独立推理、HTTP批服务、训练与LoRA教程 | 4B/8B权重、数值与Analysis推理、HTTP评测服务、pi-rl的IQL/SAC等示例 |
| 我们已有 | OpenDW/π0.5/GRPO/C32/三图/原生评估骨架可复用；当前WM目录没有click_bell RM和reset包 | 未见本项目部署回执；标准models目录未发现对应命名，未穷举所有缓存 | 刚核8B四分片、代码、独立环境仍在；已有旧数值HTTP客户端与装卸逻辑 |
| 距离离线出结果 | 近：下载并核加载、准备任务图像、核正负样本评分 | 近：独立依赖＋权重＋RGB历史采样＋读取success输出 | 近：复用资产；新增/复用官方Analysis生成、解析与unknown处理 |
| 距离接入当前GRPO | 最近：任务参数化、reset/指令、RM、原生种子与基线 | 中等：每环境历史、批量/队列、分数到reward/done、装卸和校准 | 中等：同前，另有文本生成延迟、输出缺失、版本合同 |
| 可靠性优势 | 专门面向该任务，模型轻 | 独立成功输出；训练利用成功与失败轨迹，不必生成文字 | 训练数据含RoboTwin，可同时看过程、匹配性和文本解释 |
| 关键未知 | 生成图上是否可靠、瞬间接触是否可观察、OpenDW动作响应 | 未核到OpenDW＋该任务的success精度；阈值需本任务标定 | 剩余时间/排序质量不是Success精度；解释可能错误；生成图精度未知 |

来源：[WorldArena任务权重](https://huggingface.co/WorldArena/WorldArena2.0/tree/main/reward_model/click_bell)、[Robometer模型与推理](https://huggingface.co/robometer/Robometer-4B)、[Robometer success输出](https://github.com/robometer/robometer/blob/main/robometer/evals/eval_server.py)、[LoRA教程](https://github.com/robometer/robometer/blob/main/FINETUNE_ROBOMETER.md)、[RynnValue官方](https://github.com/alibaba-damo-academy/RynnValue)。

## click_bell的主要难点是成功判据

公开原生`check_success()`检查正确夹爪闭合，以及夹爪接触位置相对铃铛接触点的几何距离；发生后`stage_success_tag`锁存为真。它不以“末帧看起来靠近铃铛”判成功，也不要求听音频。WM每4动作给一张图，短接触可能被稀疏采样或遮挡漏掉。

因此这个任务工程上近，却不一定比摆瓶子更适合单图判成功。WorldArena默认click_bell YAML使用LPIPS末帧相似度，HF同时另有分类器权重；计划优先核分类器，不把默认LPIPS当真实接触检测器。若需要短历史/腕图，这是后续针对可观察性的适配，不是直接提高阈值能解决的。

接入要补：对应权重strict load和评分核验、同刻三图/state的任务reset、任务指令、原生固定seed基线。当前reset生成器可复用，但回执task写死adjust_bottle，配置也需任务参数化；不是只换一个字符串。OpenDW先复用多任务Robotwin bundle，不预先承诺click_bell动力学有效，也不要求先重训WM。

[原生判据](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/click_bell.py)、[默认奖励配置](https://github.com/WorldArena2/WorldArena-2.0/blob/main/RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)。

## 成本与接线

沿已审权重统计，Robometer-4B BF16纯权重约8.29GiB，RynnValue-8B约17.84GiB；这不是总显存峰值。旧Rynn数值路径在B1/K4实测Torch allocated峰值18.02GiB，是9月9日旧实验，不能当新Success文本生成的峰值或吞吐。本轮未做GPU测量；细节和元数据来源见[成本专题](general_reward_integration_20261003.md)。

当前每轮512条轨迹，最长12个C32块：

- 离线抽32条轨迹复核，只需32个视频评分样本。
- 每条终局复核一次，完整一轮512个视频样本；若在线候选成功被否决后继续并再次复核，可能每条多次，不能始终按512估计。
- 每C32块评分一次，上限6144个视频样本；当前约113动作提前结束时约1800，但更换成功规则后长度会变，不能沿用旧耗时估算。
- 这些是逻辑样本数，不等于HTTP请求或GPU forward次数。Robometer可以一次对8帧输出8个分数，不必为每一帧另调用一次。

接入顺序建议：现有轨迹→独立评分→记录分歧；确认本任务成功精度后才进入奖励/终止链。GRPO可保留，不需导入两项目完整策略训练框架。

如果作为在线二级裁判，最好在现有RM提出成功、环境真正done之前复核。复核No/unknown后是否继续，需要明确环境状态与预算；事后把已结束的轨迹“改成失败”并不等于补齐后续探索。双方同意才成功通常减少误报，同时可能增加漏报。首次接入不建议同时改变RM、任务与过滤门限。

## RynnValue已有资产与版本边界

15:54实核：`/data/chenyiteng/models/RynnValue-8B-8738c5e4`、`projects/RynnValue-10e0d333`、`venvs/rynnvalue-8b-py310`均存在；模型有四个分片和manifest。目录存在不代表本轮重新验过权重内容或启动了服务。

现有[数值服务](../../../local_scripts/bc_rynnvalue_rabc_packet/SCORER_README.md)返回remaining_seconds和delta_seconds，不返回Success。官方Success是同一模型的自回归Analysis分支，没有独立success分类头；输出缺失/格式异常要记unknown，不能当成No或Yes。

新Git README强调quantile tokenizer/causal推理；本地旧权重config及本轮所读HF config仍显示symlog与`pred_slot_isolated_eager`。不能把最新README的默认行为或排名直接归给旧checkpoint。首次离线对照应锁同一权重、源码、processor和提示协议；若选新版另行审版本。[HF config](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B/blob/main/config.json)

## 建议排序

1. 最快获得独立诊断：先利用已有RynnValue资产，给保存轨迹跑Success复核；需要补接口，尚未执行。
2. 最值得考察的长期通用在线裁判：Robometer，输出更直接、权重更小；是否更准确仍需同一标注样本比较。
3. 最快扩展第二任务：click_bell＋现成小RM，保留当前框架与对照；先看原生基线和接触识别，避免复制摆瓶子的饱和问题。

三者都不能修复“WM把无效动作画成成功”的动力学偏差。若新裁判也几乎全Yes，GRPO对比仍可能饱和；必须同时看原生效果、假阳性与有效组比例。

本轮新内容仅本地保存。资产回执：[JSON](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/wm-reward-candidates-oct5-assets01.out)。raw click_bell演示数据本轮未全面盘点，不宣称不存在。
