# 原始Clean GRPO：同run六点采集

用户2026-11-02确认：原始GRPO Clean，药瓶，沿用原200轮预算；画图工作优先45，随后单任务FQL，最后RLT。

## 已启动的实际配置

- GPU：深圳1物理4/5。首轮已进入`Generating Rollout Epochs`，尚无完整首轮指标，不能把启动成功说成训练完成。
- 原始Sidneyπ0.5；`move_pillbottle_pad`，Clean；N10/H50/C50，最多200动作。
- 每轮64环境×2采样批=128轨迹，group8；两次更新，B512/micro32，学习率5e−6。冻结/训练范围、种子42/43、评估随机流隔离都沿用旧实配。
- 200轮；每5轮评32回合。只把旧每10轮保存改为每40轮保存，得到0/40/80/120/160/200六点；0复用原始基座，后五点保完整恢复文件。
- 训练源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-seed42-20260911`，原`formal_driver`和仅清理本namespace的Ray结束逻辑直接复用。
- 独立namespace `RLinf_grpo_clean_entropy_g45_20261102`；共享Ray与67训练未动。

部署：`/home/chenyiteng/deployment-20261102/grpo-clean-entropy-g45`。

训练结果：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/grpo-clean-entropy-g45-20261102`。

固定池/六点预测：`/data/chenyiteng/results/pi05-entropy/20261102-clean-sixpoint`。

## 训练后自动执行

1. CP200执行，按同任务评估种子池取32局；两卡各16，H50/C50/200。完整相机观察、指令、噪声、实际到达位置和视频沿用已跑通U/四噪声采集链。
2. 复用`pi05_u_video/run_batch.py`，新增仅两项：严格加载指定GRPO权重；在已有观察NPZ中同时保存指令。原始默认采集行为不变，新功能显式开启。
3. 六点都读取末代产生的同一批观察、相同K4噪声。每查询先B16主噪声、再B48补三噪声。CP200主噪声必须精确重现实际采集的动作。
4. 保存原始四份归一化D14动作与噪声、固定位置ID、逐位置熵CSV。初始熵十组只划一次，之后始终跟踪相同位置。熵公式/截尾沿用已验收实现。
5. 自动输出Base/Final Top20%重合表、初始十组的末代减初始熵、0–200轮的平均熵与50/80/95分位熵曲线，并保存PNG/SVG/PDF和CSV/JSON。

此次没有新增训练loss、奖励或U加权。观察池由末代执行；不拿旧Clean140执行的观察池冒充末代池。两卡各16是为了沿用主实验32局预算，并非从每卡32缩减总预算。

## 保存与接班

后五个完整GRPO点预计约134.3 GiB，0点不重复存基座。当前Data约452 GiB余量，按现有队列预算可容纳；其他用户继续增长时余量仍会变化。六点已在根`PROTECTED_CHECKPOINTS.json`登记排除清理；收集前将五个推理权重硬链接到`models/protected-eval/pillbottle/grpo-clean-entropy-g45-20261102/`，不复制占用。完整续训分片留原保存点。

45：已完成教师K3补算 → 本次Clean GRPO200 → 末代固定池与六点重放 → FQL v2从完整55更新继续到2200更新 → 原逐卡RLT。67仍为DMD new+GAD → Direct+GAD-OT → 原逐卡RLT。

训练沿用原4天超时上限；采集单批最多1小时，重放单点最多20分钟，后处理总上限4小时。失败留记录与已产出数据，精确清理本任务后推进FQL；不把缺测补0。训练首轮完成前仅给粗估2–4天，不把4天上限当测得ETA。

本轮GPU已验收的是旧药瓶补算：十query通过原教师主噪声精确复现，32局5725位置，73秒/卡。新六点流程源码和配置已准备、CPU语法检查通过；CP200尚未生成，末代采集与六点GPU重放须待训练完成，未提前宣称通过。

## 本轮药瓶配对新结果

两策略最高熵20%位置各1145处，重合977处（85.33%，独立随机参考20%）。逐位置双熵、距离、十组表和教师新增原动作均归档。该结果是Clean140与Uτ3CP70两策略比较，不是新Clean同run训练结果。
