# GeoAAC-growth：全部任务

[返回总目录](../README.md)

主图保留逐动作原始值；细虚线为chunk边界，橙色为终止段实际执行长度不明，未用于筛选。帧只对应chunk前后，不能精确定位某个h的接触瞬间。A/B/C是形态筛选等级，优先读人工备注。

| 任务 | 原始曲线缩略图 | 复核 |
|---|---|---|
| [adjust_bottle](../tasks/adjust_bottle.md) | [![adjust_bottle](../figures/adjust_bottle/geoaac_curve.png)](../figures/adjust_bottle/geoaac.png) | 受其他因素影响：谨慎：q1起始突峰，提瓶画面可见，但前缀位置效应明显。 |
| [beat_block_hammer](../tasks/beat_block_hammer.md) | [![beat_block_hammer](../figures/beat_block_hammer/geoaac_curve.png)](../figures/beat_block_hammer/geoaac.png) | 受其他因素影响：谨慎：q1举锤段窄峰与chunk前段重合。 |
| [blocks_ranking_rgb](../tasks/blocks_ranking_rgb.md) | [![blocks_ranking_rgb](../figures/blocks_ranking_rgb/geoaac_curve.png)](../figures/blocks_ranking_rgb/geoaac.png) | 受其他因素影响：谨慎：多次前缀尖峰；q8夹向蓝块，不能归结为唯一事件。 |
| [blocks_ranking_size](../tasks/blocks_ranking_size.md) | [![blocks_ranking_size](../figures/blocks_ranking_size/geoaac_curve.png)](../figures/blocks_ranking_size/geoaac.png) | 较弱：弱：几乎每隔一个或几个chunk就有前缀峰，事件特异性不清。 |
| [click_alarmclock](../tasks/click_alarmclock.md) | [![click_alarmclock](../figures/click_alarmclock/geoaac_curve.png)](../figures/click_alarmclock/geoaac.png) | 较弱：弱：q0最前几个动作的前缀峰，不足以定位按键事件。 |
| [click_bell](../tasks/click_bell.md) | [![click_bell](../figures/click_bell/geoaac_curve.png)](../figures/click_bell/geoaac.png) | 受其他因素影响：谨慎：q1开始突峰与前缀边界重合。 |
| [dump_bin_bigbin](../tasks/dump_bin_bigbin.md) | [![dump_bin_bigbin](../figures/dump_bin_bigbin/geoaac_curve.png)](../figures/dump_bin_bigbin/geoaac.png) | 受其他因素影响：谨慎：q1提起容器时增长峰，前缀开头也偏高。 |
| [grab_roller](../tasks/grab_roller.md) | [![grab_roller](../figures/grab_roller/geoaac_curve.png)](../figures/grab_roller/geoaac.png) | 受其他因素影响：谨慎：接近和夹取各有增长区，但chunk位置效应强。 |
| [handover_block](../tasks/handover_block.md) | [![handover_block](../figures/handover_block/geoaac_curve.png)](../figures/handover_block/geoaac.png) | 受其他因素影响：谨慎：q4/q5交接准备有增长峰，但夹杂多次chunk前段尖峰。 |
| [handover_mic](../tasks/handover_mic.md) | [![handover_mic](../figures/handover_mic/geoaac_curve.png)](../figures/handover_mic/geoaac.png) | 可解释候选：阶段候选：q2交接动作前缀增长较大，需保留前缀含义。 |
| [hanging_mug](../tasks/hanging_mug.md) | [![hanging_mug](../figures/hanging_mug/geoaac_curve.png)](../figures/hanging_mug/geoaac.png) | 较弱：弱：每chunk前段反复出现尖峰，挂杯事件特异性不清。 |
| [lift_pot](../tasks/lift_pot.md) | [![lift_pot](../figures/lift_pot/geoaac_curve.png)](../figures/lift_pot/geoaac.png) | 受其他因素影响：谨慎：q0接近阶段较高，同时各chunk开头有窄尖峰。 |
| [move_can_pot](../tasks/move_can_pot.md) | [![move_can_pot](../figures/move_can_pot/geoaac_curve.png)](../figures/move_can_pot/geoaac.png) | 较弱：弱：多数chunk起始尖峰，搬运阶段与前缀效应难分。 |
| [move_pillbottle_pad](../tasks/move_pillbottle_pad.md) | [![move_pillbottle_pad](../figures/move_pillbottle_pad/geoaac_curve.png)](../figures/move_pillbottle_pad/geoaac.png) | 受其他因素影响：谨慎：q2移到垫子阶段增长峰位于前缀开头。 |
| [move_playingcard_away](../tasks/move_playingcard_away.md) | [![move_playingcard_away](../figures/move_playingcard_away/geoaac_curve.png)](../figures/move_playingcard_away/geoaac.png) | 受其他因素影响：谨慎：q1抬起/移动纸牌时峰在chunk开头，前缀效应不可忽略。 |
| [move_stapler_pad](../tasks/move_stapler_pad.md) | [![move_stapler_pad](../figures/move_stapler_pad/geoaac_curve.png)](../figures/move_stapler_pad/geoaac.png) | 【失败例】受其他因素影响：谨慎：q1/q2搬运订书机时峰靠近chunk开头。 |
| [open_laptop](../tasks/open_laptop.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [open_microwave](../tasks/open_microwave.md) | [![open_microwave](../figures/open_microwave/geoaac_curve.png)](../figures/open_microwave/geoaac.png) | 较弱：弱：贯穿整条轨迹的chunk前段尖峰，不能当作多次开门事件。 |
| [pick_diverse_bottles](../tasks/pick_diverse_bottles.md) | [![pick_diverse_bottles](../figures/pick_diverse_bottles/geoaac_curve.png)](../figures/pick_diverse_bottles/geoaac.png) | 受其他因素影响：谨慎：接近/提瓶两段较高，但每段前部天然偏高。 |
| [pick_dual_bottles](../tasks/pick_dual_bottles.md) | [![pick_dual_bottles](../figures/pick_dual_bottles/geoaac_curve.png)](../figures/pick_dual_bottles/geoaac.png) | 受其他因素影响：谨慎：q0/q1前部出现双峰，与前缀位置相关。 |
| [place_a2b_left](../tasks/place_a2b_left.md) | [![place_a2b_left](../figures/place_a2b_left/geoaac_curve.png)](../figures/place_a2b_left/geoaac.png) | 较弱：弱：每chunk前几个动作尖峰，不能独立解释为搬运难度。 |
| [place_a2b_right](../tasks/place_a2b_right.md) | [![place_a2b_right](../figures/place_a2b_right/geoaac_curve.png)](../figures/place_a2b_right/geoaac.png) | 受其他因素影响：谨慎：q1举起物体时前缀峰，chunk效应明显。 |
| [place_bread_basket](../tasks/place_bread_basket.md) | [![place_bread_basket](../figures/place_bread_basket/geoaac_curve.png)](../figures/place_bread_basket/geoaac.png) | 较弱：弱：主要响应集中在初始前缀/各chunk前部。 |
| [place_bread_skillet](../tasks/place_bread_skillet.md) | [![place_bread_skillet](../figures/place_bread_skillet/geoaac_curve.png)](../figures/place_bread_skillet/geoaac.png) | 较弱：弱：前缀开头尖峰加多个小波动。 |
| [place_burger_fries](../tasks/place_burger_fries.md) | [![place_burger_fries](../figures/place_burger_fries/geoaac_curve.png)](../figures/place_burger_fries/geoaac.png) | 受其他因素影响：谨慎：q1提起汉堡时首部峰，前缀效应混杂。 |
| [place_can_basket](../tasks/place_can_basket.md) | [![place_can_basket](../figures/place_can_basket/geoaac_curve.png)](../figures/place_can_basket/geoaac.png) | 受其他因素影响：谨慎：提罐/入篮段增长峰仍贴近chunk首部。 |
| [place_cans_plasticbox](../tasks/place_cans_plasticbox.md) | [![place_cans_plasticbox](../figures/place_cans_plasticbox/geoaac_curve.png)](../figures/place_cans_plasticbox/geoaac.png) | 受其他因素影响：谨慎：q1抬罐的前缀开头峰，后续chunk也有类似尖峰。 |
| [place_container_plate](../tasks/place_container_plate.md) | [![place_container_plate](../figures/place_container_plate/geoaac_curve.png)](../figures/place_container_plate/geoaac.png) | 较弱：弱：q0前几个动作峰，不能解释为放到盘子的时刻。 |
| [place_dual_shoes](../tasks/place_dual_shoes.md) | [![place_dual_shoes](../figures/place_dual_shoes/geoaac_curve.png)](../figures/place_dual_shoes/geoaac.png) | 【失败例】受其他因素影响：失败例谨慎：初始接近较高，夹杂前缀开头尖峰。 |
| [place_empty_cup](../tasks/place_empty_cup.md) | [![place_empty_cup](../figures/place_empty_cup/geoaac_curve.png)](../figures/place_empty_cup/geoaac.png) | 较弱：弱：q0/q1最前位置尖峰，前缀结构主导。 |
| [place_fan](../tasks/place_fan.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_mouse_pad](../tasks/place_mouse_pad.md) | [![place_mouse_pad](../figures/place_mouse_pad/geoaac_curve.png)](../figures/place_mouse_pad/geoaac.png) | 较弱：弱：多chunk开始尖峰，画面不足以区分具体放置事件。 |
| [place_object_basket](../tasks/place_object_basket.md) | [![place_object_basket](../figures/place_object_basket/geoaac_curve.png)](../figures/place_object_basket/geoaac.png) | 受其他因素影响：谨慎：q1提物的前缀开头孤立峰，未直接指向入篮。 |
| [place_object_scale](../tasks/place_object_scale.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_object_stand](../tasks/place_object_stand.md) | [![place_object_stand](../figures/place_object_stand/geoaac_curve.png)](../figures/place_object_stand/geoaac.png) | 受其他因素影响：谨慎：q1物体移上台子的前缀开头峰。 |
| [place_phone_stand](../tasks/place_phone_stand.md) | [![place_phone_stand](../figures/place_phone_stand/geoaac_curve.png)](../figures/place_phone_stand/geoaac.png) | 可解释候选：阶段候选：同q3姿态调整前缀增长较高，另有多处尖峰。 |
| [place_shoe](../tasks/place_shoe.md) | [![place_shoe](../figures/place_shoe/geoaac_curve.png)](../figures/place_shoe/geoaac.png) | 受其他因素影响：谨慎：q2向垫子移动增长峰，但正位于chunk开头。 |
| [press_stapler](../tasks/press_stapler.md) | [![press_stapler](../figures/press_stapler/geoaac_curve.png)](../figures/press_stapler/geoaac.png) | 较弱：弱：初始前缀为最高，后面每chunk开始仍有窄峰。 |
| [put_bottles_dustbin](../tasks/put_bottles_dustbin.md) | [![put_bottles_dustbin](../figures/put_bottles_dustbin/geoaac_curve.png)](../figures/put_bottles_dustbin/geoaac.png) | 较弱：弱：初始前缀及其他chunk重复尖峰，事件特异性弱。 |
| [put_object_cabinet](../tasks/put_object_cabinet.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [rotate_qrcode](../tasks/rotate_qrcode.md) | [![rotate_qrcode](../figures/rotate_qrcode/geoaac_curve.png)](../figures/rotate_qrcode/geoaac.png) | 受其他因素影响：谨慎：q0/q2首部峰，夹起/转动牌架画面可见但前缀混杂。 |
| [scan_object](../tasks/scan_object.md) | [![scan_object](../figures/scan_object/geoaac_curve.png)](../figures/scan_object/geoaac.png) | 受其他因素影响：谨慎：q1/q2抓取与调整段增长峰位于前缀开头。 |
| [shake_bottle](../tasks/shake_bottle.md) | [![shake_bottle](../figures/shake_bottle/geoaac_curve.png)](../figures/shake_bottle/geoaac.png) | 【补充数据】较弱：弱：初始前缀强峰；种子重复的补充数据。 |
| [shake_bottle_horizontally](../tasks/shake_bottle_horizontally.md) | [![shake_bottle_horizontally](../figures/shake_bottle_horizontally/geoaac_curve.png)](../figures/shake_bottle_horizontally/geoaac.png) | 较弱：弱：q0最前几个动作尖峰。 |
| [stack_blocks_three](../tasks/stack_blocks_three.md) | [![stack_blocks_three](../figures/stack_blocks_three/geoaac_curve.png)](../figures/stack_blocks_three/geoaac.png) | 较弱：弱：几乎全程反复chunk前缀峰，难指认唯一堆叠事件。 |
| [stack_blocks_two](../tasks/stack_blocks_two.md) | [![stack_blocks_two](../figures/stack_blocks_two/geoaac_curve.png)](../figures/stack_blocks_two/geoaac.png) | 较弱：弱：多chunk开头重复尖峰，难分辨堆叠事件。 |
| [stack_bowls_three](../tasks/stack_bowls_three.md) | [![stack_bowls_three](../figures/stack_bowls_three/geoaac_curve.png)](../figures/stack_bowls_three/geoaac.png) | 受其他因素影响：谨慎：初始前缀及q6有增长，另有多处边界尖峰。 |
| [stack_bowls_two](../tasks/stack_bowls_two.md) | [![stack_bowls_two](../figures/stack_bowls_two/geoaac_curve.png)](../figures/stack_bowls_two/geoaac.png) | 受其他因素影响：谨慎：q4提碗时增长峰贴近chunk前缘。 |
| [stamp_seal](../tasks/stamp_seal.md) | [![stamp_seal](../figures/stamp_seal/geoaac_curve.png)](../figures/stamp_seal/geoaac.png) | 可解释候选：候选：q1拿起并向印面移动时局部增长区，仍要区分前缀效应。 |
| [turn_switch](../tasks/turn_switch.md) | [![turn_switch](../figures/turn_switch/geoaac_curve.png)](../figures/turn_switch/geoaac.png) | 受其他因素影响：谨慎：q0接近有宽区，q4窄峰在chunk首部，后者不宜解释为新接触。 |
