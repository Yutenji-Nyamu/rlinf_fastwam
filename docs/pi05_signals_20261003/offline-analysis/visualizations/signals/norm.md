# Tell-Tale Norm：全部任务

[返回总目录](../README.md)

主图保留逐动作原始值；细虚线为chunk边界，橙色为终止段实际执行长度不明，未用于筛选。帧只对应chunk前后，不能精确定位某个h的接触瞬间。A/B/C是形态筛选等级，优先读人工备注。

| 任务 | 原始曲线缩略图 | 复核 |
|---|---|---|
| [adjust_bottle](../tasks/adjust_bottle.md) | [![adjust_bottle](../figures/adjust_bottle/norm_curve.png)](../figures/adjust_bottle/norm.png) | 阶段变化：阶段差异：持瓶调整前后幅度变化，叠加chunk首部尖峰。 |
| [beat_block_hammer](../tasks/beat_block_hammer.md) | [![beat_block_hammer](../figures/beat_block_hammer/norm_curve.png)](../figures/beat_block_hammer/norm.png) | 阶段变化：阶段差异存在，标记峰含chunk首部，不能直接解释为重要性。 |
| [blocks_ranking_rgb](../tasks/blocks_ranking_rgb.md) | [![blocks_ranking_rgb](../figures/blocks_ranking_rgb/norm_curve.png)](../figures/blocks_ranking_rgb/norm.png) | 可解释候选：阶段候选：q6手抬起/离开方块附近时幅度高；中间也有其他波动。 |
| [blocks_ranking_size](../tasks/blocks_ranking_size.md) | [![blocks_ranking_size](../figures/blocks_ranking_size/norm_curve.png)](../figures/blocks_ranking_size/norm.png) | 可解释候选：阶段候选：q6抬起方块附近升高，相对变化不大。 |
| [click_alarmclock](../tasks/click_alarmclock.md) | [![click_alarmclock](../figures/click_alarmclock/norm_curve.png)](../figures/click_alarmclock/norm.png) | 较弱：弱：chunk开头反复抬高，画面变化很小。 |
| [click_bell](../tasks/click_bell.md) | [![click_bell](../figures/click_bell/norm_curve.png)](../figures/click_bell/norm.png) | 较弱：弱：每chunk开始重复高幅度，不能解释成每次都按铃。 |
| [dump_bin_bigbin](../tasks/dump_bin_bigbin.md) | [![dump_bin_bigbin](../figures/dump_bin_bigbin/norm_curve.png)](../figures/dump_bin_bigbin/norm.png) | 阶段变化：阶段差异：操作容器时较高，手离开视野后较低。 |
| [grab_roller](../tasks/grab_roller.md) | [![grab_roller](../figures/grab_roller/norm_curve.png)](../figures/grab_roller/norm.png) | 可解释候选：阶段候选：q1双手贴近/夹取附近幅度变化；画面边缘遮挡。 |
| [handover_block](../tasks/handover_block.md) | [![handover_block](../figures/handover_block/norm_curve.png)](../figures/handover_block/norm.png) | 可解释候选：阶段候选：q3靠近接物位置和q5两手接近均形成高区。 |
| [handover_mic](../tasks/handover_mic.md) | [![handover_mic](../figures/handover_mic/norm_curve.png)](../figures/handover_mic/norm.png) | 可解释候选：阶段候选：q1抓起后搬运时形成宽峰，q2交接并非最高。 |
| [hanging_mug](../tasks/hanging_mug.md) | [![hanging_mug](../figures/hanging_mug/norm_curve.png)](../figures/hanging_mug/norm.png) | 可解释候选：阶段候选：q4拿起/转动杯体时幅度明显增大。 |
| [lift_pot](../tasks/lift_pot.md) | [![lift_pot](../figures/lift_pot/norm_curve.png)](../figures/lift_pot/norm.png) | 可解释候选：阶段候选：双手靠近、握持锅的两段宽峰；位置效应也较大。 |
| [move_can_pot](../tasks/move_can_pot.md) | [![move_can_pot](../figures/move_can_pot/norm_curve.png)](../figures/move_can_pot/norm.png) | 阶段变化：阶段差异存在，选中片段在画面右缘，不能精确解释。 |
| [move_pillbottle_pad](../tasks/move_pillbottle_pad.md) | [![move_pillbottle_pad](../figures/move_pillbottle_pad/norm_curve.png)](../figures/move_pillbottle_pad/norm.png) | 可解释候选：阶段候选：q2持瓶到垫子上方比初始接近阶段幅度高。 |
| [move_playingcard_away](../tasks/move_playingcard_away.md) | [![move_playingcard_away](../figures/move_playingcard_away/norm_curve.png)](../figures/move_playingcard_away/norm.png) | 可解释候选：阶段候选：q1夹住纸牌移动时幅度高于初始阶段。 |
| [move_stapler_pad](../tasks/move_stapler_pad.md) | [![move_stapler_pad](../figures/move_stapler_pad/norm_curve.png)](../figures/move_stapler_pad/norm.png) | 【失败例】阶段变化：失败例阶段差异：搬运订书机的q1/q2幅度较高。 |
| [open_laptop](../tasks/open_laptop.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [open_microwave](../tasks/open_microwave.md) | [![open_microwave](../figures/open_microwave/norm_curve.png)](../figures/open_microwave/norm.png) | 阶段变化：阶段差异：初始靠近时高，持续操作后整体较低。 |
| [pick_diverse_bottles](../tasks/pick_diverse_bottles.md) | [![pick_diverse_bottles](../figures/pick_diverse_bottles/norm_curve.png)](../figures/pick_diverse_bottles/norm.png) | 较弱：弱：缓慢幅度变化叠加chunk开头高值，事件分辨较弱。 |
| [pick_dual_bottles](../tasks/pick_dual_bottles.md) | [![pick_dual_bottles](../figures/pick_dual_bottles/norm_curve.png)](../figures/pick_dual_bottles/norm.png) | 较弱：弱：初段幅度较高但事件区分不明确。 |
| [place_a2b_left](../tasks/place_a2b_left.md) | [![place_a2b_left](../figures/place_a2b_left/norm_curve.png)](../figures/place_a2b_left/norm.png) | 较弱：弱：各chunk开头重复抬高，幅度变化含边界模式。 |
| [place_a2b_right](../tasks/place_a2b_right.md) | [![place_a2b_right](../figures/place_a2b_right/norm_curve.png)](../figures/place_a2b_right/norm.png) | 较弱：弱：主要峰与chunk开头重合。 |
| [place_bread_basket](../tasks/place_bread_basket.md) | [![place_bread_basket](../figures/place_bread_basket/norm_curve.png)](../figures/place_bread_basket/norm.png) | 阶段变化：阶段差异：抓面包和移入篮子幅度不同，但并非DV主峰位置。 |
| [place_bread_skillet](../tasks/place_bread_skillet.md) | [![place_bread_skillet](../figures/place_bread_skillet/norm_curve.png)](../figures/place_bread_skillet/norm.png) | 阶段变化：阶段差异：q1持面包时高，之后较低；chunk首部尖峰重复。 |
| [place_burger_fries](../tasks/place_burger_fries.md) | [![place_burger_fries](../figures/place_burger_fries/norm_curve.png)](../figures/place_burger_fries/norm.png) | 可解释候选：阶段候选：q1提起食物和q3放下后另一只手靠近时较高。 |
| [place_can_basket](../tasks/place_can_basket.md) | [![place_can_basket](../figures/place_can_basket/norm_curve.png)](../figures/place_can_basket/norm.png) | 可解释候选：候选：q2/q3篮口对准并向内放置时宽高区。 |
| [place_cans_plasticbox](../tasks/place_cans_plasticbox.md) | [![place_cans_plasticbox](../figures/place_cans_plasticbox/norm_curve.png)](../figures/place_cans_plasticbox/norm.png) | 可解释候选：阶段候选：q1双手握罐抬起时幅度高，放好后较低。 |
| [place_container_plate](../tasks/place_container_plate.md) | [![place_container_plate](../figures/place_container_plate/norm_curve.png)](../figures/place_container_plate/norm.png) | 可解释候选：阶段候选：q1持容器靠近盘子时幅度高，但末端也有固定上升。 |
| [place_dual_shoes](../tasks/place_dual_shoes.md) | [![place_dual_shoes](../figures/place_dual_shoes/norm_curve.png)](../figures/place_dual_shoes/norm.png) | 【失败例】阶段变化：失败例阶段差异：早期拿鞋幅度高，后续低；不能解释为成功程度。 |
| [place_empty_cup](../tasks/place_empty_cup.md) | [![place_empty_cup](../figures/place_empty_cup/norm_curve.png)](../figures/place_empty_cup/norm.png) | 受其他因素影响：谨慎：q1末端幅度上升，位置效应显著。 |
| [place_fan](../tasks/place_fan.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_mouse_pad](../tasks/place_mouse_pad.md) | [![place_mouse_pad](../figures/place_mouse_pad/norm_curve.png)](../figures/place_mouse_pad/norm.png) | 较弱：弱阶段差异：q1拿鼠标较高、之后较低，重复边界尖峰不少。 |
| [place_object_basket](../tasks/place_object_basket.md) | [![place_object_basket](../figures/place_object_basket/norm_curve.png)](../figures/place_object_basket/norm.png) | 可解释候选：优先阶段候选：q2/q3持物越过篮口并放入时宽高区，之后回落。 |
| [place_object_scale](../tasks/place_object_scale.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_object_stand](../tasks/place_object_stand.md) | [![place_object_stand](../figures/place_object_stand/norm_curve.png)](../figures/place_object_stand/norm.png) | 可解释候选：阶段候选：q1拿物体靠近台面时较高，边界影响不可忽略。 |
| [place_phone_stand](../tasks/place_phone_stand.md) | [![place_phone_stand](../figures/place_phone_stand/norm_curve.png)](../figures/place_phone_stand/norm.png) | 可解释候选：候选：较短成功轨迹q1手机向支架移动时宽峰。 |
| [place_shoe](../tasks/place_shoe.md) | [![place_shoe](../figures/place_shoe/norm_curve.png)](../figures/place_shoe/norm.png) | 阶段变化：阶段差异较弱：提鞋前后幅度不同，不能单独定位重要事件。 |
| [press_stapler](../tasks/press_stapler.md) | [![press_stapler](../figures/press_stapler/norm_curve.png)](../figures/press_stapler/norm.png) | 较弱：弱：chunk起点反复高值。 |
| [put_bottles_dustbin](../tasks/put_bottles_dustbin.md) | [![put_bottles_dustbin](../figures/put_bottles_dustbin/norm_curve.png)](../figures/put_bottles_dustbin/norm.png) | 受其他因素影响：谨慎：q5高峰画面主要在边缘，语义证据不够。 |
| [put_object_cabinet](../tasks/put_object_cabinet.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [rotate_qrcode](../tasks/rotate_qrcode.md) | [![rotate_qrcode](../figures/rotate_qrcode/norm_curve.png)](../figures/rotate_qrcode/norm.png) | 可解释候选：阶段候选：q2牌面被转到镜头前时幅度高，q3保持时低。 |
| [scan_object](../tasks/scan_object.md) | [![scan_object](../figures/scan_object/norm_curve.png)](../figures/scan_object/norm.png) | 可解释候选：阶段候选：q1持扫描器准备转向时幅度高，q2转向后较低。 |
| [shake_bottle](../tasks/shake_bottle.md) | [![shake_bottle](../figures/shake_bottle/norm_curve.png)](../figures/shake_bottle/norm.png) | 【补充数据】可解释候选：补充数据阶段候选：q1提起瓶子时幅度高。 |
| [shake_bottle_horizontally](../tasks/shake_bottle_horizontally.md) | [![shake_bottle_horizontally](../figures/shake_bottle_horizontally/norm_curve.png)](../figures/shake_bottle_horizontally/norm.png) | 可解释候选：候选：q1抬起瓶子附近宽峰清楚，尚不能对应摇动周期。 |
| [stack_blocks_three](../tasks/stack_blocks_three.md) | [![stack_blocks_three](../figures/stack_blocks_three/norm_curve.png)](../figures/stack_blocks_three/norm.png) | 可解释候选：阶段候选：q3拿起方块时幅度高，q8移开后低。 |
| [stack_blocks_two](../tasks/stack_blocks_two.md) | [![stack_blocks_two](../figures/stack_blocks_two/norm_curve.png)](../figures/stack_blocks_two/norm.png) | 可解释候选：阶段候选：q3从原位置提起方块时宽峰。 |
| [stack_bowls_three](../tasks/stack_bowls_three.md) | [![stack_bowls_three](../figures/stack_bowls_three/norm_curve.png)](../figures/stack_bowls_three/norm.png) | 可解释候选：阶段候选：q2搬碗、q6堆叠附近有两个较宽高区。 |
| [stack_bowls_two](../tasks/stack_bowls_two.md) | [![stack_bowls_two](../figures/stack_bowls_two/norm_curve.png)](../figures/stack_bowls_two/norm.png) | 可解释候选：候选：q2搬碗时宽高区，幅度变化可见但不是重要性的证明。 |
| [stamp_seal](../tasks/stamp_seal.md) | [![stamp_seal](../figures/stamp_seal/norm_curve.png)](../figures/stamp_seal/norm.png) | 可解释候选：候选：q1印章移向印面时逐渐升高。 |
| [turn_switch](../tasks/turn_switch.md) | [![turn_switch](../figures/turn_switch/norm_curve.png)](../figures/turn_switch/norm.png) | 较弱：弱：每chunk开头反复高，难解释为每次都发生关键事件。 |
