# 七信号＋DV：服务器离线筛选与看图复核

2026-10-04；这是描述性展示筛选，不是显著性检验、失败预测性能或因果归因。

- 正式验收89批、1424条，其中751条成功；另有shake_bottle两批32条作为重复种子补充数据。
- 1077次批量决策已重算八条信号；除SR数值算法误差1.33e-15外，与采集分数一致。
- 50任务×8信号=400个条目；46任务有368组曲线/帧，4任务32项缺数据。
- 352项来自成功轨迹；move_stapler_pad、place_dual_shoes的16项为失败候选。
- 看过46张任务总图的全部8行，共368组。所有标记片段来自终止前的确定部分；橙色终止段不用于筛选。
- 图像只记录每个chunk前后，不能把某个峰精确叫作接触时刻。q/h/action索引均从0开始。

## 优先看什么

- DV/GEO/U-GROW：先看place_bread_basket，同一成功轨迹的入篮阶段共同升高；handover_mic、scan_object也有清楚阶段。
- Norm：先看place_object_basket的宽高区，含义是表示幅度；不要求它必须与DV同峰。
- GeoAAC：handover_mic可看前缀增长；大量任务的峰位于chunk前部，不能解释为独立动作难度。
- SR：pick_dual_bottles有较居中的局部峰；整体仍多边界形状，且stable rank非常接近1。
- Fresco/SHIFT：place_object_basket有宽峰，但路径长度混杂明显，保留作对照，不强推为不确定性/关键接触证据。

## 全量诊断

以下先在每条正式轨迹内、只用非终止已确认部分，计算Spearman相关，再取中位数；不足15个有效动作或常数序列不计。

| 信号 | 与初噪声到最终动作距离 | 与初噪声范数 | 与相邻动作变化量 |
|---|---:|---:|---:|
| dv | 0.064 | 0.034 | 0.262 |
| fresco | 0.998 | 0.499 | -0.110 |
| geo | -0.459 | -0.218 | 0.302 |
| geoaac | -0.509 | -0.210 | 0.301 |
| shift | 0.818 | 0.437 | -0.053 |
| norm | 0.032 | -0.116 | 0.219 |
| sr | 0.002 | 0.128 | 0.288 |
| ugrow | -0.074 | 0.023 | 0.312 |

Fresco的0.998意味着当前分数几乎按生成路径长度排序，并不等于观测条件下的随机不确定性。若去噪轨迹近似直线，跨去噪时间的草稿方差本来就与起终点距离平方成正比。SHIFT也较受该因素影响；这不等于原语言模型方法无效，只说明本次动作去噪适配存在混杂。

DV与GEO的相关中位数约0.643，DV与U-GROW约0.365；它们有重叠但并非一条信号。相关不能证明因果或训练收益。

## 逐任务逐信号看图记录

candidate=有可解释阶段候选；phase_only=形态有变化但事件证据弱；confounded=位置/遮挡/路径长度混杂；weak=不适合作为关键事件示例；no_data=没有完整数据。自动A/B/C仅衡量曲线形状，优先看人工复核这一列。

### adjust_bottle

[八信号总图](task_sheets/adjust_bottle.jpg) · [任务图册](tasks/adjust_bottle.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | adjust_bottle_b1_s14 | candidate | 候选：q2瓶身由横向转为竖直；峰在该chunk后段。 |
| fresco | adjust_bottle_b1_s02 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | adjust_bottle_b1_s01 | candidate | 候选：q2持瓶调整段有宽峰，精确接触不可定位。 |
| geoaac | adjust_bottle_b0_s10 | confounded | 谨慎：q1起始突峰，提瓶画面可见，但前缀位置效应明显。 |
| shift | adjust_bottle_b1_s07 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | adjust_bottle_b0_s13 | phase_only | 阶段差异：持瓶调整前后幅度变化，叠加chunk首部尖峰。 |
| sr | adjust_bottle_b1_s02 | confounded | 谨慎：抬瓶段末端升高，接近chunk边界。 |
| ugrow | adjust_bottle_b1_s01 | candidate | 候选：q2持瓶调整段集中升高。 |

### beat_block_hammer

[八信号总图](task_sheets/beat_block_hammer.jpg) · [任务图册](tasks/beat_block_hammer.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | beat_block_hammer_b1_s14 | candidate | 候选：q1由抓住锤柄到举锤/朝方块移动，局部峰清楚。 |
| fresco | beat_block_hammer_b1_s11 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | beat_block_hammer_b0_s01 | candidate | 候选：q0接近并抓取锤柄段高，不能称已经击打。 |
| geoaac | beat_block_hammer_b1_s14 | confounded | 谨慎：q1举锤段窄峰与chunk前段重合。 |
| shift | beat_block_hammer_b0_s03 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | beat_block_hammer_b1_s13 | phase_only | 阶段差异存在，标记峰含chunk首部，不能直接解释为重要性。 |
| sr | beat_block_hammer_b1_s03 | confounded | 谨慎：q1结束附近升高，动作邻域/边界影响。 |
| ugrow | beat_block_hammer_b1_s08 | candidate | 候选：持锤在方块附近移动的q2/q3有局部高区；敲击瞬间不明。 |

### blocks_ranking_rgb

[八信号总图](task_sheets/blocks_ranking_rgb.jpg) · [任务图册](tasks/blocks_ranking_rgb.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | blocks_ranking_rgb_b1_s05 | candidate | 候选：q2/q9抓取或移动不同颜色方块，各有清楚峰。 |
| fresco | blocks_ranking_rgb_b0_s11 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | blocks_ranking_rgb_b0_s11 | candidate | 候选：q4/q10手离开原位置并转向方块，阶段变化清楚。 |
| geoaac | blocks_ranking_rgb_b0_s05 | confounded | 谨慎：多次前缀尖峰；q8夹向蓝块，不能归结为唯一事件。 |
| shift | blocks_ranking_rgb_b1_s02 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | blocks_ranking_rgb_b0_s04 | candidate | 阶段候选：q6手抬起/离开方块附近时幅度高；中间也有其他波动。 |
| sr | blocks_ranking_rgb_b1_s06 | confounded | 谨慎：q6结束附近高峰突出，但正处chunk边界。 |
| ugrow | blocks_ranking_rgb_b0_s11 | candidate | 候选：q4/q6转移方块阶段升高，同时存在其他尖峰。 |

### blocks_ranking_size

[八信号总图](task_sheets/blocks_ranking_size.jpg) · [任务图册](tasks/blocks_ranking_size.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | blocks_ranking_size_b0_s06 | candidate | 候选：q9操作小方块时有主要峰，另有多次小峰。 |
| fresco | blocks_ranking_size_b0_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | blocks_ranking_size_b0_s04 | candidate | 候选：q3/q12抓取/移开不同尺寸方块，两个主要高区。 |
| geoaac | blocks_ranking_size_b1_s00 | weak | 弱：几乎每隔一个或几个chunk就有前缀峰，事件特异性不清。 |
| shift | blocks_ranking_size_b0_s04 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | blocks_ranking_size_b0_s13 | candidate | 阶段候选：q6抬起方块附近升高，相对变化不大。 |
| sr | blocks_ranking_size_b1_s10 | confounded | 谨慎：q6末端峰主要受边界影响。 |
| ugrow | blocks_ranking_size_b1_s07 | candidate | 候选：q3夹取并抬起方块段强升高。 |

### click_alarmclock

[八信号总图](task_sheets/click_alarmclock.jpg) · [任务图册](tasks/click_alarmclock.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | click_alarmclock_b1_s05 | candidate | 候选：q1手指靠近闹钟上部时局部峰；画面不能证实按下瞬间。 |
| fresco | click_alarmclock_b1_s14 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | click_alarmclock_b1_s05 | candidate | 候选：与DV同一轨迹q1峰，接近闹钟阶段。 |
| geoaac | click_alarmclock_b0_s04 | weak | 弱：q0最前几个动作的前缀峰，不足以定位按键事件。 |
| shift | click_alarmclock_b1_s13 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | click_alarmclock_b1_s13 | weak | 弱：chunk开头反复抬高，画面变化很小。 |
| sr | click_alarmclock_b1_s13 | weak | 弱：chunk末端/开头重复抬高。 |
| ugrow | click_alarmclock_b0_s06 | candidate | 阶段候选：q0靠近闹钟时持续升高，未定位接触。 |

### click_bell

[八信号总图](task_sheets/click_bell.jpg) · [任务图册](tasks/click_bell.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | click_bell_b1_s09 | candidate | 候选：q0手从远处到铃上方，末段集中峰。 |
| fresco | click_bell_b1_s01 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | click_bell_b1_s09 | candidate | 阶段候选：q1在铃上方调整时持续较高，非独立窄峰。 |
| geoaac | click_bell_b1_s09 | confounded | 谨慎：q1开始突峰与前缀边界重合。 |
| shift | click_bell_b1_s07 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | click_bell_b1_s08 | weak | 弱：每chunk开始重复高幅度，不能解释成每次都按铃。 |
| sr | click_bell_b1_s09 | weak | 弱：q0/q1交界高峰，位置模板影响。 |
| ugrow | click_bell_b1_s09 | candidate | 候选：q0接近和q1局部调整都有峰，画面仅确认靠近。 |

### dump_bin_bigbin

[八信号总图](task_sheets/dump_bin_bigbin.jpg) · [任务图册](tasks/dump_bin_bigbin.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | dump_bin_bigbin_b1_s00 | candidate | 候选：q1/q3抬起或放回小容器阶段出现两个峰；大垃圾桶部分出画。 |
| fresco | dump_bin_bigbin_b1_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | dump_bin_bigbin_b1_s06 | candidate | 阶段候选：q3/q5手离开/回到容器附近，画面不足以精确确认倾倒。 |
| geoaac | dump_bin_bigbin_b1_s02 | confounded | 谨慎：q1提起容器时增长峰，前缀开头也偏高。 |
| shift | dump_bin_bigbin_b0_s02 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | dump_bin_bigbin_b0_s11 | phase_only | 阶段差异：操作容器时较高，手离开视野后较低。 |
| sr | dump_bin_bigbin_b1_s14 | weak | 弱：多chunk边缘升高。 |
| ugrow | dump_bin_bigbin_b1_s00 | candidate | 候选：q3/q5容器操作附近两个局部高区。 |

### grab_roller

[八信号总图](task_sheets/grab_roller.jpg) · [任务图册](tasks/grab_roller.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | grab_roller_b0_s09 | candidate | 候选：q0双手接近滚筒的后半段升高。 |
| fresco | grab_roller_b0_s13 | weak | 弱阶段趋势：接近后幅度上升但抖动密集，生成路径长度混杂强。 |
| geo | grab_roller_b1_s14 | candidate | 候选：q0双手展开并接近滚筒时较高，后续下降。 |
| geoaac | grab_roller_b0_s03 | confounded | 谨慎：接近和夹取各有增长区，但chunk位置效应强。 |
| shift | grab_roller_b0_s03 | weak | 弱阶段趋势：接近后略抬高但抖动密集，路径长度混杂强。 |
| norm | grab_roller_b1_s11 | candidate | 阶段候选：q1双手贴近/夹取附近幅度变化；画面边缘遮挡。 |
| sr | grab_roller_b1_s10 | weak | 弱：小幅多峰，chunk位置模板较强。 |
| ugrow | grab_roller_b1_s14 | candidate | 候选：q0双手接近滚筒时集中高区。 |

### handover_block

[八信号总图](task_sheets/handover_block.jpg) · [任务图册](tasks/handover_block.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | handover_block_b1_s13 | candidate | 候选：q6两只夹爪在持块区域相遇前升高；终止后高值单独标橙。 |
| fresco | handover_block_b1_s04 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | handover_block_b1_s13 | candidate | 候选：同轨迹q6交接接近段升高。 |
| geoaac | handover_block_b0_s10 | confounded | 谨慎：q4/q5交接准备有增长峰，但夹杂多次chunk前段尖峰。 |
| shift | handover_block_b1_s04 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | handover_block_b0_s10 | candidate | 阶段候选：q3靠近接物位置和q5两手接近均形成高区。 |
| sr | handover_block_b1_s13 | weak | 弱：多chunk边界重复上升。 |
| ugrow | handover_block_b1_s13 | candidate | 候选：同轨迹q6两手交接接近段持续强升高。 |

### handover_mic

[八信号总图](task_sheets/handover_mic.jpg) · [任务图册](tasks/handover_mic.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | handover_mic_b0_s14 | candidate | 候选：q2两手接近并交接麦克风时升高。 |
| fresco | handover_mic_b0_s01 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | handover_mic_b1_s10 | candidate | 候选：q2两手相遇段出现主峰，q1搬运段较低。 |
| geoaac | handover_mic_b1_s10 | candidate | 阶段候选：q2交接动作前缀增长较大，需保留前缀含义。 |
| shift | handover_mic_b1_s05 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | handover_mic_b0_s14 | candidate | 阶段候选：q1抓起后搬运时形成宽峰，q2交接并非最高。 |
| sr | handover_mic_b1_s07 | weak | 弱：每chunk端点反复升高，58%位置模板效应。 |
| ugrow | handover_mic_b1_s07 | candidate | 候选：q2两手接近时持续高区，q1较低。 |

### hanging_mug

[八信号总图](task_sheets/hanging_mug.jpg) · [任务图册](tasks/hanging_mug.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | hanging_mug_b0_s00 | candidate | 候选：q3/q5持杯旋转、向挂架移动段升高；峰贴近chunk末端。 |
| fresco | hanging_mug_b0_s01 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | hanging_mug_b0_s00 | candidate | 候选：q5向挂架移动段升高；还未能确认挂上瞬间。 |
| geoaac | hanging_mug_b0_s09 | weak | 弱：每chunk前段反复出现尖峰，挂杯事件特异性不清。 |
| shift | hanging_mug_b0_s01 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | hanging_mug_b0_s01 | candidate | 阶段候选：q4拿起/转动杯体时幅度明显增大。 |
| sr | hanging_mug_b0_s09 | confounded | 谨慎：q4末端主要峰接近边界。 |
| ugrow | hanging_mug_b0_s01 | candidate | 候选：q3杯柄抓取/调整姿态附近局部高区。 |

### lift_pot

[八信号总图](task_sheets/lift_pot.jpg) · [任务图册](tasks/lift_pot.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | lift_pot_b1_s09 | candidate | 候选：q1双手对准锅两侧手柄时峰清楚。 |
| fresco | lift_pot_b0_s05 | weak | 弱阶段趋势：接近后整体抬高，逐动作抖动仍密集。 |
| geo | lift_pot_b0_s15 | candidate | 候选：q0接近锅和q1握持准备两段高区。 |
| geoaac | lift_pot_b1_s14 | confounded | 谨慎：q0接近阶段较高，同时各chunk开头有窄尖峰。 |
| shift | lift_pot_b1_s01 | weak | 弱阶段候选：接近锅时先低后高，但位置模板与路径长度混杂强。 |
| norm | lift_pot_b1_s08 | candidate | 阶段候选：双手靠近、握持锅的两段宽峰；位置效应也较大。 |
| sr | lift_pot_b1_s07 | candidate | 候选：q1提锅附近内部局部峰，但仍有边缘效应。 |
| ugrow | lift_pot_b1_s09 | candidate | 候选：q0双手伸向锅柄时集中高区。 |

### move_can_pot

[八信号总图](task_sheets/move_can_pot.jpg) · [任务图册](tasks/move_can_pot.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | move_can_pot_b1_s13 | confounded | 谨慎：q1峰清楚，但罐和手在画面右缘，语义证据不足。 |
| fresco | move_can_pot_b0_s04 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | move_can_pot_b1_s08 | candidate | 候选：q2把罐向锅边移动时宽峰。 |
| geoaac | move_can_pot_b1_s10 | weak | 弱：多数chunk起始尖峰，搬运阶段与前缀效应难分。 |
| shift | move_can_pot_b1_s08 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | move_can_pot_b0_s04 | phase_only | 阶段差异存在，选中片段在画面右缘，不能精确解释。 |
| sr | move_can_pot_b1_s10 | weak | 弱：chunk边界反复变化。 |
| ugrow | move_can_pot_b1_s08 | candidate | 候选：q2移动至锅边、q3接近放置各有高区。 |

### move_pillbottle_pad

[八信号总图](task_sheets/move_pillbottle_pad.jpg) · [任务图册](tasks/move_pillbottle_pad.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | move_pillbottle_pad_b0_s11 | candidate | 候选：q1提瓶、q2移向垫子有两个局部高区。 |
| fresco | move_pillbottle_pad_b0_s03 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | move_pillbottle_pad_b0_s00 | candidate | 候选：q1提瓶与q2放到垫子附近升高。 |
| geoaac | move_pillbottle_pad_b0_s02 | confounded | 谨慎：q2移到垫子阶段增长峰位于前缀开头。 |
| shift | move_pillbottle_pad_b1_s05 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | move_pillbottle_pad_b1_s06 | candidate | 阶段候选：q2持瓶到垫子上方比初始接近阶段幅度高。 |
| sr | move_pillbottle_pad_b1_s06 | confounded | 谨慎：q2起始突升，可能主要是chunk边界。 |
| ugrow | move_pillbottle_pad_b1_s05 | candidate | 候选：q2移向垫子、q3放置附近集中变化。 |

### move_playingcard_away

[八信号总图](task_sheets/move_playingcard_away.jpg) · [任务图册](tasks/move_playingcard_away.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | move_playingcard_away_b0_s06 | weak | 形态清楚但语义弱：q3/q4画面里的纸牌几乎不变，手在边缘。 |
| fresco | move_playingcard_away_b1_s12 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | move_playingcard_away_b0_s06 | candidate | 候选：q0手接近纸牌有明显高区；q4高区画面解释弱。 |
| geoaac | move_playingcard_away_b1_s04 | confounded | 谨慎：q1抬起/移动纸牌时峰在chunk开头，前缀效应不可忽略。 |
| shift | move_playingcard_away_b1_s12 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | move_playingcard_away_b0_s04 | candidate | 阶段候选：q1夹住纸牌移动时幅度高于初始阶段。 |
| sr | move_playingcard_away_b1_s05 | weak | 弱：边界与多处小峰混杂。 |
| ugrow | move_playingcard_away_b0_s06 | candidate | 候选：q5手重新接近纸牌有高峰，其他峰较多。 |

### move_stapler_pad

[八信号总图](task_sheets/move_stapler_pad.jpg) · [任务图册](tasks/move_stapler_pad.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | move_stapler_pad_b0_s10 | candidate | 失败例候选：q4手持订书机在垫子附近操作时孤立强峰。 |
| fresco | move_stapler_pad_b1_s03 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | move_stapler_pad_b0_s10 | candidate | 失败例候选：同轨迹q4明显宽峰。 |
| geoaac | move_stapler_pad_b1_s11 | confounded | 谨慎：q1/q2搬运订书机时峰靠近chunk开头。 |
| shift | move_stapler_pad_b1_s03 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | move_stapler_pad_b1_s01 | phase_only | 失败例阶段差异：搬运订书机的q1/q2幅度较高。 |
| sr | move_stapler_pad_b1_s07 | weak | 弱：q2首部峰突出，但边界效应强。 |
| ugrow | move_stapler_pad_b0_s10 | candidate | 失败例候选：同DV轨迹q4集中升高。 |

### open_laptop

[八信号总图](task_sheets/open_laptop.jpg) · [任务图册](tasks/open_laptop.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | 无 | no_data | 该任务未取得完整可复算批次。 |
| fresco | 无 | no_data | 该任务未取得完整可复算批次。 |
| geo | 无 | no_data | 该任务未取得完整可复算批次。 |
| geoaac | 无 | no_data | 该任务未取得完整可复算批次。 |
| shift | 无 | no_data | 该任务未取得完整可复算批次。 |
| norm | 无 | no_data | 该任务未取得完整可复算批次。 |
| sr | 无 | no_data | 该任务未取得完整可复算批次。 |
| ugrow | 无 | no_data | 该任务未取得完整可复算批次。 |

### open_microwave

[八信号总图](task_sheets/open_microwave.jpg) · [任务图册](tasks/open_microwave.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | open_microwave_b0_s05 | candidate | 候选：q1抓住门侧并拉动时孤立强峰。 |
| fresco | open_microwave_b0_s09 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | open_microwave_b1_s12 | candidate | 候选：q0接近微波炉门/把手时高，后续较低。 |
| geoaac | open_microwave_b0_s05 | weak | 弱：贯穿整条轨迹的chunk前段尖峰，不能当作多次开门事件。 |
| shift | open_microwave_b0_s05 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | open_microwave_b0_s02 | phase_only | 阶段差异：初始靠近时高，持续操作后整体较低。 |
| sr | open_microwave_b1_s04 | weak | 弱：周期性边界峰，最高点不能单独解释为开门。 |
| ugrow | open_microwave_b0_s13 | candidate | 候选：q0接近及q6门侧调整都有峰，开门瞬间无法定位。 |

### pick_diverse_bottles

[八信号总图](task_sheets/pick_diverse_bottles.jpg) · [任务图册](tasks/pick_diverse_bottles.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | pick_diverse_bottles_b0_s10 | candidate | 候选：q1瓶子从被夹住到抬离桌面时孤立峰。 |
| fresco | pick_diverse_bottles_b0_s02 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | pick_diverse_bottles_b1_s00 | candidate | 阶段候选：q0接近、q1提瓶各有高区。 |
| geoaac | pick_diverse_bottles_b1_s09 | confounded | 谨慎：接近/提瓶两段较高，但每段前部天然偏高。 |
| shift | pick_diverse_bottles_b1_s02 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | pick_diverse_bottles_b1_s07 | weak | 弱：缓慢幅度变化叠加chunk开头高值，事件分辨较弱。 |
| sr | pick_diverse_bottles_b1_s00 | confounded | 谨慎：q1内部提瓶附近有峰，同时存在固定边缘变化。 |
| ugrow | pick_diverse_bottles_b0_s13 | candidate | 候选：q0双手接近瓶子时集中高区。 |

### pick_dual_bottles

[八信号总图](task_sheets/pick_dual_bottles.jpg) · [任务图册](tasks/pick_dual_bottles.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | pick_dual_bottles_b0_s02 | candidate | 候选：q0接近与q1提瓶两个小峰；橙色终止段更大峰不用于筛选。 |
| fresco | pick_dual_bottles_b0_s15 | weak | 弱阶段趋势：接近/提瓶期间宽高区，但路径长度混杂。 |
| geo | pick_dual_bottles_b1_s01 | candidate | 候选：q0双手接近瓶子为主要高区。 |
| geoaac | pick_dual_bottles_b1_s01 | confounded | 谨慎：q0/q1前部出现双峰，与前缀位置相关。 |
| shift | pick_dual_bottles_b0_s15 | weak | 弱阶段趋势：接近/提瓶期间宽高区，抖动与路径长度混杂。 |
| norm | pick_dual_bottles_b1_s03 | weak | 弱：初段幅度较高但事件区分不明确。 |
| sr | pick_dual_bottles_b1_s06 | candidate | 候选：q1中间提瓶附近局部峰，较少依赖边界。 |
| ugrow | pick_dual_bottles_b1_s03 | candidate | 候选：q0接近两个瓶子时集中升高。 |

### place_a2b_left

[八信号总图](task_sheets/place_a2b_left.jpg) · [任务图册](tasks/place_a2b_left.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_a2b_left_b1_s07 | candidate | 候选：q1开始提起方盒时局部强峰；靠近chunk前缘。 |
| fresco | place_a2b_left_b0_s12 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_a2b_left_b1_s00 | candidate | 候选：q0抓取、q2移向目标附近两个高区。 |
| geoaac | place_a2b_left_b1_s04 | weak | 弱：每chunk前几个动作尖峰，不能独立解释为搬运难度。 |
| shift | place_a2b_left_b1_s11 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_a2b_left_b0_s10 | weak | 弱：各chunk开头重复抬高，幅度变化含边界模式。 |
| sr | place_a2b_left_b1_s10 | confounded | 谨慎：q1/q2交界升高，动作窗口语义不独立。 |
| ugrow | place_a2b_left_b1_s00 | candidate | 候选：q2物体移向目标区域时持续升高。 |

### place_a2b_right

[八信号总图](task_sheets/place_a2b_right.jpg) · [任务图册](tasks/place_a2b_right.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_a2b_right_b1_s09 | candidate | 候选：q0接近物体和q2移至目标两处峰。 |
| fresco | place_a2b_right_b1_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_a2b_right_b0_s14 | candidate | 候选：q0接近并夹取物体时宽高区。 |
| geoaac | place_a2b_right_b1_s10 | confounded | 谨慎：q1举起物体时前缀峰，chunk效应明显。 |
| shift | place_a2b_right_b1_s13 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_a2b_right_b0_s04 | weak | 弱：主要峰与chunk开头重合。 |
| sr | place_a2b_right_b0_s12 | confounded | 谨慎：q1末端突出，但位于chunk边界。 |
| ugrow | place_a2b_right_b1_s13 | candidate | 阶段候选：q2搬运物体高区；该例部分动作出画。 |

### place_bread_basket

[八信号总图](task_sheets/place_bread_basket.jpg) · [任务图册](tasks/place_bread_basket.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_bread_basket_b0_s05 | candidate | 优先候选：q4持面包移入篮子时单个主高区，同轨迹GEO/U-GROW也响应。 |
| fresco | place_bread_basket_b1_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_bread_basket_b0_s05 | candidate | 优先候选：同一q4面包移入篮子阶段明显升高。 |
| geoaac | place_bread_basket_b1_s06 | weak | 弱：主要响应集中在初始前缀/各chunk前部。 |
| shift | place_bread_basket_b1_s13 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_bread_basket_b0_s05 | phase_only | 阶段差异：抓面包和移入篮子幅度不同，但并非DV主峰位置。 |
| sr | place_bread_basket_b1_s03 | weak | 弱：多chunk边缘峰，不能直接解释为多个重要事件。 |
| ugrow | place_bread_basket_b0_s05 | candidate | 优先候选：与DV同轨迹q4移入篮子高区。 |

### place_bread_skillet

[八信号总图](task_sheets/place_bread_skillet.jpg) · [任务图册](tasks/place_bread_skillet.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_bread_skillet_b0_s11 | candidate | 候选：q2双手分别持面包/平底锅并靠近时强升高。 |
| fresco | place_bread_skillet_b0_s00 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_bread_skillet_b1_s07 | candidate | 阶段候选：q2开始移动面包/锅时高，具体接触被视角限制。 |
| geoaac | place_bread_skillet_b0_s00 | weak | 弱：前缀开头尖峰加多个小波动。 |
| shift | place_bread_skillet_b0_s15 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_bread_skillet_b0_s15 | phase_only | 阶段差异：q1持面包时高，之后较低；chunk首部尖峰重复。 |
| sr | place_bread_skillet_b0_s15 | weak | 弱：多chunk末端高值，位置模板较强。 |
| ugrow | place_bread_skillet_b1_s00 | candidate | 候选：q2锅与面包靠近阶段局部高峰。 |

### place_burger_fries

[八信号总图](task_sheets/place_burger_fries.jpg) · [任务图册](tasks/place_burger_fries.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_burger_fries_b1_s13 | candidate | 候选：q2汉堡从托盘外移到盘内时孤立主峰。 |
| fresco | place_burger_fries_b0_s15 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_burger_fries_b1_s06 | candidate | 候选：q0双手靠近食物时高，后续较低。 |
| geoaac | place_burger_fries_b0_s11 | confounded | 谨慎：q1提起汉堡时首部峰，前缀效应混杂。 |
| shift | place_burger_fries_b0_s13 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_burger_fries_b1_s06 | candidate | 阶段候选：q1提起食物和q3放下后另一只手靠近时较高。 |
| sr | place_burger_fries_b0_s14 | weak | 弱：chunk开头峰，后续小幅波动。 |
| ugrow | place_burger_fries_b0_s01 | candidate | 候选：q0接近、q1拿起食物时高区。 |

### place_can_basket

[八信号总图](task_sheets/place_can_basket.jpg) · [任务图册](tasks/place_can_basket.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_can_basket_b0_s13 | candidate | 候选：q2持罐越过篮口附近有主要峰。 |
| fresco | place_can_basket_b0_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_can_basket_b0_s13 | candidate | 候选：q1抬罐和q2移向篮口出现主要高区。 |
| geoaac | place_can_basket_b1_s00 | confounded | 谨慎：提罐/入篮段增长峰仍贴近chunk首部。 |
| shift | place_can_basket_b0_s11 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_can_basket_b1_s07 | candidate | 候选：q2/q3篮口对准并向内放置时宽高区。 |
| sr | place_can_basket_b1_s12 | confounded | 谨慎：q2放入篮子段末尾峰清楚，但接近边界。 |
| ugrow | place_can_basket_b0_s13 | candidate | 候选：q1抬罐及q3篮内操作有高区，伴随其他尖峰。 |

### place_cans_plasticbox

[八信号总图](task_sheets/place_cans_plasticbox.jpg) · [任务图册](tasks/place_cans_plasticbox.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_cans_plasticbox_b1_s12 | candidate | 候选：q4手在箱内罐子附近调整时高；同轨迹更早还有一次峰。 |
| fresco | place_cans_plasticbox_b0_s01 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_cans_plasticbox_b0_s12 | candidate | 候选：q2持罐移到箱内时宽高区。 |
| geoaac | place_cans_plasticbox_b1_s11 | confounded | 谨慎：q1抬罐的前缀开头峰，后续chunk也有类似尖峰。 |
| shift | place_cans_plasticbox_b1_s04 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_cans_plasticbox_b1_s12 | candidate | 阶段候选：q1双手握罐抬起时幅度高，放好后较低。 |
| sr | place_cans_plasticbox_b1_s03 | confounded | 谨慎：两个主要峰靠近q0/q1末端，位置效应仍在。 |
| ugrow | place_cans_plasticbox_b1_s13 | candidate | 候选：q2入箱和q3第二只手跟进时局部高区。 |

### place_container_plate

[八信号总图](task_sheets/place_container_plate.jpg) · [任务图册](tasks/place_container_plate.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_container_plate_b1_s04 | candidate | 候选：q1容器由桌面抬起时局部峰；画面在右缘。 |
| fresco | place_container_plate_b1_s03 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_container_plate_b1_s04 | candidate | 候选：同一轨迹q1抬容器时峰清楚。 |
| geoaac | place_container_plate_b1_s07 | weak | 弱：q0前几个动作峰，不能解释为放到盘子的时刻。 |
| shift | place_container_plate_b0_s15 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_container_plate_b0_s12 | candidate | 阶段候选：q1持容器靠近盘子时幅度高，但末端也有固定上升。 |
| sr | place_container_plate_b1_s02 | weak | 弱：chunk末端升高，无法独立对应放置。 |
| ugrow | place_container_plate_b1_s04 | confounded | 谨慎：q0接近容器有峰，部分动作靠近画面右缘。 |

### place_dual_shoes

[八信号总图](task_sheets/place_dual_shoes.jpg) · [任务图册](tasks/place_dual_shoes.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_dual_shoes_b0_s13 | candidate | 失败例候选：q3鞋从箱内/箱边提起并调整时高，后面低。 |
| fresco | place_dual_shoes_b1_s03 | candidate | 失败例阶段候选：q3有强宽峰，画面见鞋抬起；但仍高度受生成路径长度影响。 |
| geo | place_dual_shoes_b0_s13 | candidate | 失败例候选：同轨迹q3鞋的调整段高，低段仍未成功。 |
| geoaac | place_dual_shoes_b0_s00 | confounded | 失败例谨慎：初始接近较高，夹杂前缀开头尖峰。 |
| shift | place_dual_shoes_b0_s07 | candidate | 失败例阶段候选：q3后多个chunk整体抬高，不是少量事件峰；路径长度混杂明显。 |
| norm | place_dual_shoes_b0_s12 | phase_only | 失败例阶段差异：早期拿鞋幅度高，后续低；不能解释为成功程度。 |
| sr | place_dual_shoes_b0_s01 | confounded | 失败例谨慎：q4开始附近强峰与chunk边界重合。 |
| ugrow | place_dual_shoes_b1_s08 | candidate | 失败例候选：q3鞋在箱边旋转/调整时明显峰。 |

### place_empty_cup

[八信号总图](task_sheets/place_empty_cup.jpg) · [任务图册](tasks/place_empty_cup.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_empty_cup_b1_s13 | candidate | 候选：q1抬杯时局部峰；终止段不用于解释。 |
| fresco | place_empty_cup_b1_s08 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_empty_cup_b1_s08 | candidate | 候选：q0夹爪接近杯口时高，之后降低。 |
| geoaac | place_empty_cup_b1_s15 | weak | 弱：q0/q1最前位置尖峰，前缀结构主导。 |
| shift | place_empty_cup_b1_s08 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_empty_cup_b0_s00 | confounded | 谨慎：q1末端幅度上升，位置效应显著。 |
| sr | place_empty_cup_b0_s01 | weak | 弱：chunk末端系统性上升。 |
| ugrow | place_empty_cup_b0_s02 | candidate | 候选：q0接近杯子阶段孤立主峰。 |

### place_fan

[八信号总图](task_sheets/place_fan.jpg) · [任务图册](tasks/place_fan.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | 无 | no_data | 该任务未取得完整可复算批次。 |
| fresco | 无 | no_data | 该任务未取得完整可复算批次。 |
| geo | 无 | no_data | 该任务未取得完整可复算批次。 |
| geoaac | 无 | no_data | 该任务未取得完整可复算批次。 |
| shift | 无 | no_data | 该任务未取得完整可复算批次。 |
| norm | 无 | no_data | 该任务未取得完整可复算批次。 |
| sr | 无 | no_data | 该任务未取得完整可复算批次。 |
| ugrow | 无 | no_data | 该任务未取得完整可复算批次。 |

### place_mouse_pad

[八信号总图](task_sheets/place_mouse_pad.jpg) · [任务图册](tasks/place_mouse_pad.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_mouse_pad_b1_s08 | candidate | 候选：q3鼠标贴近垫子调整时末段强升高。 |
| fresco | place_mouse_pad_b0_s12 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_mouse_pad_b1_s08 | candidate | 候选：同轨迹q3垫子上操作时高。 |
| geoaac | place_mouse_pad_b0_s11 | weak | 弱：多chunk开始尖峰，画面不足以区分具体放置事件。 |
| shift | place_mouse_pad_b0_s12 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_mouse_pad_b0_s12 | weak | 弱阶段差异：q1拿鼠标较高、之后较低，重复边界尖峰不少。 |
| sr | place_mouse_pad_b0_s11 | weak | 弱：小幅多峰，主要标记在chunk边界。 |
| ugrow | place_mouse_pad_b1_s08 | candidate | 候选：同轨迹q3垫子上调整时窄峰。 |

### place_object_basket

[八信号总图](task_sheets/place_object_basket.jpg) · [任务图册](tasks/place_object_basket.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_object_basket_b1_s01 | candidate | 候选：q3入篮和q4篮内调整有高区。 |
| fresco | place_object_basket_b0_s07 | confounded | 阶段候选但混杂强：q2/q3存在清楚宽峰，和物体入篮同时发生，仍主要测路径长度。 |
| geo | place_object_basket_b1_s02 | candidate | 候选：q2持物到篮口、q3放入篮内各有峰。 |
| geoaac | place_object_basket_b1_s06 | confounded | 谨慎：q1提物的前缀开头孤立峰，未直接指向入篮。 |
| shift | place_object_basket_b0_s07 | confounded | 阶段候选但混杂强：与Fresco同轨迹入篮时宽高区，不足以证明是语义难度。 |
| norm | place_object_basket_b0_s07 | candidate | 优先阶段候选：q2/q3持物越过篮口并放入时宽高区，之后回落。 |
| sr | place_object_basket_b1_s07 | confounded | 谨慎：q2接近篮口的高值主要在chunk开头。 |
| ugrow | place_object_basket_b1_s12 | candidate | 候选：q3物体由篮口上方落入篮内时明显高区。 |

### place_object_scale

[八信号总图](task_sheets/place_object_scale.jpg) · [任务图册](tasks/place_object_scale.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | 无 | no_data | 该任务未取得完整可复算批次。 |
| fresco | 无 | no_data | 该任务未取得完整可复算批次。 |
| geo | 无 | no_data | 该任务未取得完整可复算批次。 |
| geoaac | 无 | no_data | 该任务未取得完整可复算批次。 |
| shift | 无 | no_data | 该任务未取得完整可复算批次。 |
| norm | 无 | no_data | 该任务未取得完整可复算批次。 |
| sr | 无 | no_data | 该任务未取得完整可复算批次。 |
| ugrow | 无 | no_data | 该任务未取得完整可复算批次。 |

### place_object_stand

[八信号总图](task_sheets/place_object_stand.jpg) · [任务图册](tasks/place_object_stand.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_object_stand_b0_s01 | candidate | 候选：q1提起并移向台子时末段升高；只有一批数据。 |
| fresco | place_object_stand_b0_s05 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_object_stand_b0_s13 | candidate | 候选：q0接近物体时高，q1移动较低。 |
| geoaac | place_object_stand_b0_s00 | confounded | 谨慎：q1物体移上台子的前缀开头峰。 |
| shift | place_object_stand_b0_s12 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_object_stand_b0_s14 | candidate | 阶段候选：q1拿物体靠近台面时较高，边界影响不可忽略。 |
| sr | place_object_stand_b0_s01 | weak | 弱：q1末端固定抬高。 |
| ugrow | place_object_stand_b0_s13 | candidate | 候选：q0接近物体时有单个主要峰。 |

### place_phone_stand

[八信号总图](task_sheets/place_phone_stand.jpg) · [任务图册](tasks/place_phone_stand.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_phone_stand_b1_s14 | confounded | 谨慎：q6握着手机/支架附近末段升高，遮挡大，接触不可辨。 |
| fresco | place_phone_stand_b1_s14 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_phone_stand_b1_s14 | candidate | 候选：q3手机竖直姿态调整时宽峰。 |
| geoaac | place_phone_stand_b1_s14 | candidate | 阶段候选：同q3姿态调整前缀增长较高，另有多处尖峰。 |
| shift | place_phone_stand_b1_s14 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_phone_stand_b0_s11 | candidate | 候选：较短成功轨迹q1手机向支架移动时宽峰。 |
| sr | place_phone_stand_b1_s14 | weak | 弱：每chunk开头重复上升。 |
| ugrow | place_phone_stand_b1_s14 | candidate | 候选：q3手机姿态调整时形成主要高区。 |

### place_shoe

[八信号总图](task_sheets/place_shoe.jpg) · [任务图册](tasks/place_shoe.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | place_shoe_b1_s13 | confounded | 谨慎：q8鞋在垫子上调整时末端升高；画面变化小，另有早期峰。 |
| fresco | place_shoe_b0_s13 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | place_shoe_b1_s13 | candidate | 候选：q0抓鞋、q2移向垫子两段高区。 |
| geoaac | place_shoe_b1_s13 | confounded | 谨慎：q2向垫子移动增长峰，但正位于chunk开头。 |
| shift | place_shoe_b0_s13 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | place_shoe_b1_s13 | phase_only | 阶段差异较弱：提鞋前后幅度不同，不能单独定位重要事件。 |
| sr | place_shoe_b0_s13 | weak | 弱：多处重复边界峰。 |
| ugrow | place_shoe_b1_s13 | candidate | 候选：q2持鞋移到垫子附近有局部主峰。 |

### press_stapler

[八信号总图](task_sheets/press_stapler.jpg) · [任务图册](tasks/press_stapler.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | press_stapler_b0_s03 | candidate | 候选：q1手在订书机上方对准/下压附近局部峰；按下瞬间不可辨。 |
| fresco | press_stapler_b1_s02 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | press_stapler_b0_s02 | candidate | 候选：q0手接近订书机时持续高区，后续低。 |
| geoaac | press_stapler_b1_s05 | weak | 弱：初始前缀为最高，后面每chunk开始仍有窄峰。 |
| shift | press_stapler_b1_s05 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | press_stapler_b1_s11 | weak | 弱：chunk起点反复高值。 |
| sr | press_stapler_b1_s13 | weak | 弱：小幅多波动，缺少清楚事件对应。 |
| ugrow | press_stapler_b1_s03 | candidate | 候选：q0接近订书机时强宽峰，后续大部分低。 |

### put_bottles_dustbin

[八信号总图](task_sheets/put_bottles_dustbin.jpg) · [任务图册](tasks/put_bottles_dustbin.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | put_bottles_dustbin_b0_s12 | confounded | 谨慎：q2有强峰，但夹爪/桶位于画面边缘，无法确认入桶动作。 |
| fresco | put_bottles_dustbin_b0_s02 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | put_bottles_dustbin_b0_s02 | candidate | 候选：q24手重新接近倒下的瓶子时高区，长轨迹仍有多峰。 |
| geoaac | put_bottles_dustbin_b1_s05 | weak | 弱：初始前缀及其他chunk重复尖峰，事件特异性弱。 |
| shift | put_bottles_dustbin_b0_s02 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | put_bottles_dustbin_b1_s05 | confounded | 谨慎：q5高峰画面主要在边缘，语义证据不够。 |
| sr | put_bottles_dustbin_b1_s15 | weak | 弱：边界高峰反复出现。 |
| ugrow | put_bottles_dustbin_b0_s00 | candidate | 阶段候选：q8/q9持瓶操作时高，但入桶目标出画，不标成入桶瞬间。 |

### put_object_cabinet

[八信号总图](task_sheets/put_object_cabinet.jpg) · [任务图册](tasks/put_object_cabinet.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | 无 | no_data | 该任务未取得完整可复算批次。 |
| fresco | 无 | no_data | 该任务未取得完整可复算批次。 |
| geo | 无 | no_data | 该任务未取得完整可复算批次。 |
| geoaac | 无 | no_data | 该任务未取得完整可复算批次。 |
| shift | 无 | no_data | 该任务未取得完整可复算批次。 |
| norm | 无 | no_data | 该任务未取得完整可复算批次。 |
| sr | 无 | no_data | 该任务未取得完整可复算批次。 |
| ugrow | 无 | no_data | 该任务未取得完整可复算批次。 |

### rotate_qrcode

[八信号总图](task_sheets/rotate_qrcode.jpg) · [任务图册](tasks/rotate_qrcode.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | rotate_qrcode_b0_s03 | weak | 形态清楚但语义弱：标记片段手和二维码牌大部分出画。 |
| fresco | rotate_qrcode_b0_s08 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | rotate_qrcode_b0_s10 | candidate | 候选：q0夹爪靠近并夹住牌架时高区，物体在画面右缘。 |
| geoaac | rotate_qrcode_b1_s05 | confounded | 谨慎：q0/q2首部峰，夹起/转动牌架画面可见但前缀混杂。 |
| shift | rotate_qrcode_b0_s08 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | rotate_qrcode_b0_s10 | candidate | 阶段候选：q2牌面被转到镜头前时幅度高，q3保持时低。 |
| sr | rotate_qrcode_b0_s07 | weak | 弱：每chunk边缘重复高值。 |
| ugrow | rotate_qrcode_b0_s11 | candidate | 候选：q2转动二维码牌到可见方向时持续高区。 |

### scan_object

[八信号总图](task_sheets/scan_object.jpg) · [任务图册](tasks/scan_object.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | scan_object_b0_s10 | candidate | 候选：q2扫描器在夹爪内转动、另一手持物时末段孤立峰。 |
| fresco | scan_object_b1_s03 | weak | 弱阶段趋势：扫描器改变姿态后升高，路径长度混杂。 |
| geo | scan_object_b0_s10 | candidate | 候选：同轨迹q2扫描器转向物体时宽高区。 |
| geoaac | scan_object_b0_s10 | confounded | 谨慎：q1/q2抓取与调整段增长峰位于前缀开头。 |
| shift | scan_object_b0_s10 | weak | 弱阶段趋势：拿起扫描器后基线升高，不能解释为扫描成功。 |
| norm | scan_object_b1_s03 | candidate | 阶段候选：q1持扫描器准备转向时幅度高，q2转向后较低。 |
| sr | scan_object_b0_s02 | confounded | 谨慎：q1/q2交界高峰，边界效应仍需区分。 |
| ugrow | scan_object_b0_s10 | candidate | 候选：同DV轨迹q2扫描器朝目标转向时峰明显。 |

### shake_bottle

[八信号总图](task_sheets/shake_bottle.jpg) · [任务图册](tasks/shake_bottle.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | shake_bottle_b1_s09 | candidate | 补充数据候选：q0接近瓶子有宽峰；该任务有重复种子，不计正式验收。 |
| fresco | shake_bottle_b1_s15 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | shake_bottle_b1_s08 | candidate | 补充数据候选：q3重新接近横卧瓶子时高，并非摇动瞬间。 |
| geoaac | shake_bottle_b1_s05 | weak | 弱：初始前缀强峰；种子重复的补充数据。 |
| shift | shake_bottle_b0_s02 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | shake_bottle_b1_s05 | candidate | 补充数据阶段候选：q1提起瓶子时幅度高。 |
| sr | shake_bottle_b1_s08 | weak | 弱：chunk边界反复上升；补充数据。 |
| ugrow | shake_bottle_b1_s08 | candidate | 补充数据候选：q3重新接近瓶子时局部峰，终止段另有高值但不用。 |

### shake_bottle_horizontally

[八信号总图](task_sheets/shake_bottle_horizontally.jpg) · [任务图册](tasks/shake_bottle_horizontally.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | shake_bottle_horizontally_b1_s02 | candidate | 候选：q1夹住并抬起瓶子时末段高峰，未定位摇动。 |
| fresco | shake_bottle_horizontally_b1_s01 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | shake_bottle_horizontally_b1_s01 | confounded | 谨慎：q0接近并夹住瓶子时高，位置模板效应较强。 |
| geoaac | shake_bottle_horizontally_b0_s06 | weak | 弱：q0最前几个动作尖峰。 |
| shift | shake_bottle_horizontally_b0_s07 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | shake_bottle_horizontally_b0_s03 | candidate | 候选：q1抬起瓶子附近宽峰清楚，尚不能对应摇动周期。 |
| sr | shake_bottle_horizontally_b0_s08 | weak | 弱：初段与chunk边缘高，不足以解释摇动。 |
| ugrow | shake_bottle_horizontally_b1_s13 | candidate | 候选：q0接近并抓瓶时主要峰，后续较低。 |

### stack_blocks_three

[八信号总图](task_sheets/stack_blocks_three.jpg) · [任务图册](tasks/stack_blocks_three.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | stack_blocks_three_b1_s13 | candidate | 候选：q14第三个方块接近已有堆叠时主峰；另有早期强峰。 |
| fresco | stack_blocks_three_b1_s07 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | stack_blocks_three_b0_s14 | candidate | 阶段候选：q3/q5拿起、搬运方块形成高区，仍有其他峰。 |
| geoaac | stack_blocks_three_b1_s13 | weak | 弱：几乎全程反复chunk前缀峰，难指认唯一堆叠事件。 |
| shift | stack_blocks_three_b1_s14 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | stack_blocks_three_b1_s07 | candidate | 阶段候选：q3拿起方块时幅度高，q8移开后低。 |
| sr | stack_blocks_three_b1_s07 | weak | 弱：标记高区在chunk边缘，峰多。 |
| ugrow | stack_blocks_three_b0_s14 | candidate | 候选：q7堆叠/释放附近有局部主峰，q3等也有其他峰。 |

### stack_blocks_two

[八信号总图](task_sheets/stack_blocks_two.jpg) · [任务图册](tasks/stack_blocks_two.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | stack_blocks_two_b0_s07 | candidate | 候选：q7在两块积木上方反复调整时高，另有后续峰。 |
| fresco | stack_blocks_two_b1_s15 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | stack_blocks_two_b0_s07 | candidate | 候选：q8夹持方块调整姿态时宽高区。 |
| geoaac | stack_blocks_two_b1_s02 | weak | 弱：多chunk开头重复尖峰，难分辨堆叠事件。 |
| shift | stack_blocks_two_b0_s12 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | stack_blocks_two_b1_s01 | candidate | 阶段候选：q3从原位置提起方块时宽峰。 |
| sr | stack_blocks_two_b0_s04 | weak | 弱：多峰且主要高点位于chunk边界。 |
| ugrow | stack_blocks_two_b0_s12 | candidate | 候选：q3提起方块和q5移到另一方块上方两个局部高区。 |

### stack_bowls_three

[八信号总图](task_sheets/stack_bowls_three.jpg) · [任务图册](tasks/stack_bowls_three.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | stack_bowls_three_b0_s14 | candidate | 候选：q6手从已叠碗边缘移开时主峰，q8再抓碗有次峰。 |
| fresco | stack_bowls_three_b0_s05 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | stack_bowls_three_b0_s13 | candidate | 候选：q0最初接近碗时高，后续大部分低。 |
| geoaac | stack_bowls_three_b0_s13 | confounded | 谨慎：初始前缀及q6有增长，另有多处边界尖峰。 |
| shift | stack_bowls_three_b0_s14 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | stack_bowls_three_b1_s13 | candidate | 阶段候选：q2搬碗、q6堆叠附近有两个较宽高区。 |
| sr | stack_bowls_three_b1_s14 | confounded | 谨慎：q6放下/离开碗附近有局部峰，但临近边界。 |
| ugrow | stack_bowls_three_b1_s12 | candidate | 候选：q3夹爪把碗移向另一碗时局部高区。 |

### stack_bowls_two

[八信号总图](task_sheets/stack_bowls_two.jpg) · [任务图册](tasks/stack_bowls_two.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | stack_bowls_two_b1_s08 | candidate | 候选：q3夹爪从碗边抬起/再次调整时强峰。 |
| fresco | stack_bowls_two_b1_s10 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | stack_bowls_two_b0_s02 | candidate | 候选：q0最初接近碗时宽高区。 |
| geoaac | stack_bowls_two_b0_s14 | confounded | 谨慎：q4提碗时增长峰贴近chunk前缘。 |
| shift | stack_bowls_two_b0_s04 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | stack_bowls_two_b0_s09 | candidate | 候选：q2搬碗时宽高区，幅度变化可见但不是重要性的证明。 |
| sr | stack_bowls_two_b1_s04 | weak | 弱：q1/q2边界高峰，后续多处重复。 |
| ugrow | stack_bowls_two_b1_s04 | candidate | 候选：q4抓取/抬起第二只碗时集中宽高区。 |

### stamp_seal

[八信号总图](task_sheets/stamp_seal.jpg) · [任务图册](tasks/stamp_seal.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | stamp_seal_b1_s09 | weak | 形态清楚但语义弱：q1强峰片段手和印章出画较多。 |
| fresco | stamp_seal_b0_s06 | weak | 弱：逐动作抖动较密，未见可靠独立事件对应；全数据与噪声到最终动作距离高度相关。 |
| geo | stamp_seal_b1_s09 | weak | 形态清楚但语义弱：与DV同片段峰，无法可靠看清盖章动作。 |
| geoaac | stamp_seal_b1_s08 | candidate | 候选：q1拿起并向印面移动时局部增长区，仍要区分前缀效应。 |
| shift | stamp_seal_b0_s01 | weak | 弱：小幅抖动/阶段基线变化，未见可靠独立事件对应；路径长度混杂强。 |
| norm | stamp_seal_b0_s05 | candidate | 候选：q1印章移向印面时逐渐升高。 |
| sr | stamp_seal_b0_s12 | weak | 弱：宽小波动夹杂末端升高，无法单独定位盖章。 |
| ugrow | stamp_seal_b1_s12 | candidate | 候选：q4/q5印章在印面附近调整的两个峰；接触瞬间不可定位。 |

### turn_switch

[八信号总图](task_sheets/turn_switch.jpg) · [任务图册](tasks/turn_switch.html)

| 信号 | 候选 | 复核 | 观察 |
|---|---|---|---|
| dv | turn_switch_b0_s15 | candidate | 候选：q0夹爪从远处伸到开关处时集中峰；不是后期拨动的唯一高点。 |
| fresco | turn_switch_b1_s00 | weak | 弱阶段趋势：靠近开关后基线升高但抖动密集，生成路径长度混杂强。 |
| geo | turn_switch_b1_s00 | candidate | 候选：q0接近、q1开关附近调整两个高区。 |
| geoaac | turn_switch_b1_s00 | confounded | 谨慎：q0接近有宽区，q4窄峰在chunk首部，后者不宜解释为新接触。 |
| shift | turn_switch_b1_s00 | weak | 弱阶段趋势：靠近开关后基线略抬高，抖动与路径长度混杂强。 |
| norm | turn_switch_b0_s12 | weak | 弱：每chunk开头反复高，难解释为每次都发生关键事件。 |
| sr | turn_switch_b1_s10 | weak | 弱：chunk边界重复高、小幅多波动。 |
| ugrow | turn_switch_b1_s10 | candidate | 候选：q0接近开关时孤立主峰，其余较低。 |
