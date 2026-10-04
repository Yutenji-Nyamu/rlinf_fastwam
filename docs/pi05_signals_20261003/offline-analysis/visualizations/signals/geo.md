# GEO-full：全部任务

[返回总目录](../README.md)

主图保留逐动作原始值；细虚线为chunk边界，橙色为终止段实际执行长度不明，未用于筛选。帧只对应chunk前后，不能精确定位某个h的接触瞬间。A/B/C是形态筛选等级，优先读人工备注。

| 任务 | 原始曲线缩略图 | 复核 |
|---|---|---|
| [adjust_bottle](../tasks/adjust_bottle.md) | [![adjust_bottle](../figures/adjust_bottle/geo_curve.png)](../figures/adjust_bottle/geo.png) | 可解释候选：候选：q2持瓶调整段有宽峰，精确接触不可定位。 |
| [beat_block_hammer](../tasks/beat_block_hammer.md) | [![beat_block_hammer](../figures/beat_block_hammer/geo_curve.png)](../figures/beat_block_hammer/geo.png) | 可解释候选：候选：q0接近并抓取锤柄段高，不能称已经击打。 |
| [blocks_ranking_rgb](../tasks/blocks_ranking_rgb.md) | [![blocks_ranking_rgb](../figures/blocks_ranking_rgb/geo_curve.png)](../figures/blocks_ranking_rgb/geo.png) | 可解释候选：候选：q4/q10手离开原位置并转向方块，阶段变化清楚。 |
| [blocks_ranking_size](../tasks/blocks_ranking_size.md) | [![blocks_ranking_size](../figures/blocks_ranking_size/geo_curve.png)](../figures/blocks_ranking_size/geo.png) | 可解释候选：候选：q3/q12抓取/移开不同尺寸方块，两个主要高区。 |
| [click_alarmclock](../tasks/click_alarmclock.md) | [![click_alarmclock](../figures/click_alarmclock/geo_curve.png)](../figures/click_alarmclock/geo.png) | 可解释候选：候选：与DV同一轨迹q1峰，接近闹钟阶段。 |
| [click_bell](../tasks/click_bell.md) | [![click_bell](../figures/click_bell/geo_curve.png)](../figures/click_bell/geo.png) | 可解释候选：阶段候选：q1在铃上方调整时持续较高，非独立窄峰。 |
| [dump_bin_bigbin](../tasks/dump_bin_bigbin.md) | [![dump_bin_bigbin](../figures/dump_bin_bigbin/geo_curve.png)](../figures/dump_bin_bigbin/geo.png) | 可解释候选：阶段候选：q3/q5手离开/回到容器附近，画面不足以精确确认倾倒。 |
| [grab_roller](../tasks/grab_roller.md) | [![grab_roller](../figures/grab_roller/geo_curve.png)](../figures/grab_roller/geo.png) | 可解释候选：候选：q0双手展开并接近滚筒时较高，后续下降。 |
| [handover_block](../tasks/handover_block.md) | [![handover_block](../figures/handover_block/geo_curve.png)](../figures/handover_block/geo.png) | 可解释候选：候选：同轨迹q6交接接近段升高。 |
| [handover_mic](../tasks/handover_mic.md) | [![handover_mic](../figures/handover_mic/geo_curve.png)](../figures/handover_mic/geo.png) | 可解释候选：候选：q2两手相遇段出现主峰，q1搬运段较低。 |
| [hanging_mug](../tasks/hanging_mug.md) | [![hanging_mug](../figures/hanging_mug/geo_curve.png)](../figures/hanging_mug/geo.png) | 可解释候选：候选：q5向挂架移动段升高；还未能确认挂上瞬间。 |
| [lift_pot](../tasks/lift_pot.md) | [![lift_pot](../figures/lift_pot/geo_curve.png)](../figures/lift_pot/geo.png) | 可解释候选：候选：q0接近锅和q1握持准备两段高区。 |
| [move_can_pot](../tasks/move_can_pot.md) | [![move_can_pot](../figures/move_can_pot/geo_curve.png)](../figures/move_can_pot/geo.png) | 可解释候选：候选：q2把罐向锅边移动时宽峰。 |
| [move_pillbottle_pad](../tasks/move_pillbottle_pad.md) | [![move_pillbottle_pad](../figures/move_pillbottle_pad/geo_curve.png)](../figures/move_pillbottle_pad/geo.png) | 可解释候选：候选：q1提瓶与q2放到垫子附近升高。 |
| [move_playingcard_away](../tasks/move_playingcard_away.md) | [![move_playingcard_away](../figures/move_playingcard_away/geo_curve.png)](../figures/move_playingcard_away/geo.png) | 可解释候选：候选：q0手接近纸牌有明显高区；q4高区画面解释弱。 |
| [move_stapler_pad](../tasks/move_stapler_pad.md) | [![move_stapler_pad](../figures/move_stapler_pad/geo_curve.png)](../figures/move_stapler_pad/geo.png) | 【失败例】可解释候选：失败例候选：同轨迹q4明显宽峰。 |
| [open_laptop](../tasks/open_laptop.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [open_microwave](../tasks/open_microwave.md) | [![open_microwave](../figures/open_microwave/geo_curve.png)](../figures/open_microwave/geo.png) | 可解释候选：候选：q0接近微波炉门/把手时高，后续较低。 |
| [pick_diverse_bottles](../tasks/pick_diverse_bottles.md) | [![pick_diverse_bottles](../figures/pick_diverse_bottles/geo_curve.png)](../figures/pick_diverse_bottles/geo.png) | 可解释候选：阶段候选：q0接近、q1提瓶各有高区。 |
| [pick_dual_bottles](../tasks/pick_dual_bottles.md) | [![pick_dual_bottles](../figures/pick_dual_bottles/geo_curve.png)](../figures/pick_dual_bottles/geo.png) | 可解释候选：候选：q0双手接近瓶子为主要高区。 |
| [place_a2b_left](../tasks/place_a2b_left.md) | [![place_a2b_left](../figures/place_a2b_left/geo_curve.png)](../figures/place_a2b_left/geo.png) | 可解释候选：候选：q0抓取、q2移向目标附近两个高区。 |
| [place_a2b_right](../tasks/place_a2b_right.md) | [![place_a2b_right](../figures/place_a2b_right/geo_curve.png)](../figures/place_a2b_right/geo.png) | 可解释候选：候选：q0接近并夹取物体时宽高区。 |
| [place_bread_basket](../tasks/place_bread_basket.md) | [![place_bread_basket](../figures/place_bread_basket/geo_curve.png)](../figures/place_bread_basket/geo.png) | 可解释候选：优先候选：同一q4面包移入篮子阶段明显升高。 |
| [place_bread_skillet](../tasks/place_bread_skillet.md) | [![place_bread_skillet](../figures/place_bread_skillet/geo_curve.png)](../figures/place_bread_skillet/geo.png) | 可解释候选：阶段候选：q2开始移动面包/锅时高，具体接触被视角限制。 |
| [place_burger_fries](../tasks/place_burger_fries.md) | [![place_burger_fries](../figures/place_burger_fries/geo_curve.png)](../figures/place_burger_fries/geo.png) | 可解释候选：候选：q0双手靠近食物时高，后续较低。 |
| [place_can_basket](../tasks/place_can_basket.md) | [![place_can_basket](../figures/place_can_basket/geo_curve.png)](../figures/place_can_basket/geo.png) | 可解释候选：候选：q1抬罐和q2移向篮口出现主要高区。 |
| [place_cans_plasticbox](../tasks/place_cans_plasticbox.md) | [![place_cans_plasticbox](../figures/place_cans_plasticbox/geo_curve.png)](../figures/place_cans_plasticbox/geo.png) | 可解释候选：候选：q2持罐移到箱内时宽高区。 |
| [place_container_plate](../tasks/place_container_plate.md) | [![place_container_plate](../figures/place_container_plate/geo_curve.png)](../figures/place_container_plate/geo.png) | 可解释候选：候选：同一轨迹q1抬容器时峰清楚。 |
| [place_dual_shoes](../tasks/place_dual_shoes.md) | [![place_dual_shoes](../figures/place_dual_shoes/geo_curve.png)](../figures/place_dual_shoes/geo.png) | 【失败例】可解释候选：失败例候选：同轨迹q3鞋的调整段高，低段仍未成功。 |
| [place_empty_cup](../tasks/place_empty_cup.md) | [![place_empty_cup](../figures/place_empty_cup/geo_curve.png)](../figures/place_empty_cup/geo.png) | 可解释候选：候选：q0夹爪接近杯口时高，之后降低。 |
| [place_fan](../tasks/place_fan.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_mouse_pad](../tasks/place_mouse_pad.md) | [![place_mouse_pad](../figures/place_mouse_pad/geo_curve.png)](../figures/place_mouse_pad/geo.png) | 可解释候选：候选：同轨迹q3垫子上操作时高。 |
| [place_object_basket](../tasks/place_object_basket.md) | [![place_object_basket](../figures/place_object_basket/geo_curve.png)](../figures/place_object_basket/geo.png) | 可解释候选：候选：q2持物到篮口、q3放入篮内各有峰。 |
| [place_object_scale](../tasks/place_object_scale.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [place_object_stand](../tasks/place_object_stand.md) | [![place_object_stand](../figures/place_object_stand/geo_curve.png)](../figures/place_object_stand/geo.png) | 可解释候选：候选：q0接近物体时高，q1移动较低。 |
| [place_phone_stand](../tasks/place_phone_stand.md) | [![place_phone_stand](../figures/place_phone_stand/geo_curve.png)](../figures/place_phone_stand/geo.png) | 可解释候选：候选：q3手机竖直姿态调整时宽峰。 |
| [place_shoe](../tasks/place_shoe.md) | [![place_shoe](../figures/place_shoe/geo_curve.png)](../figures/place_shoe/geo.png) | 可解释候选：候选：q0抓鞋、q2移向垫子两段高区。 |
| [press_stapler](../tasks/press_stapler.md) | [![press_stapler](../figures/press_stapler/geo_curve.png)](../figures/press_stapler/geo.png) | 可解释候选：候选：q0手接近订书机时持续高区，后续低。 |
| [put_bottles_dustbin](../tasks/put_bottles_dustbin.md) | [![put_bottles_dustbin](../figures/put_bottles_dustbin/geo_curve.png)](../figures/put_bottles_dustbin/geo.png) | 可解释候选：候选：q24手重新接近倒下的瓶子时高区，长轨迹仍有多峰。 |
| [put_object_cabinet](../tasks/put_object_cabinet.md) | 无完整数据 | 缺数据：该任务未取得完整可复算批次。 |
| [rotate_qrcode](../tasks/rotate_qrcode.md) | [![rotate_qrcode](../figures/rotate_qrcode/geo_curve.png)](../figures/rotate_qrcode/geo.png) | 可解释候选：候选：q0夹爪靠近并夹住牌架时高区，物体在画面右缘。 |
| [scan_object](../tasks/scan_object.md) | [![scan_object](../figures/scan_object/geo_curve.png)](../figures/scan_object/geo.png) | 可解释候选：候选：同轨迹q2扫描器转向物体时宽高区。 |
| [shake_bottle](../tasks/shake_bottle.md) | [![shake_bottle](../figures/shake_bottle/geo_curve.png)](../figures/shake_bottle/geo.png) | 【补充数据】可解释候选：补充数据候选：q3重新接近横卧瓶子时高，并非摇动瞬间。 |
| [shake_bottle_horizontally](../tasks/shake_bottle_horizontally.md) | [![shake_bottle_horizontally](../figures/shake_bottle_horizontally/geo_curve.png)](../figures/shake_bottle_horizontally/geo.png) | 受其他因素影响：谨慎：q0接近并夹住瓶子时高，位置模板效应较强。 |
| [stack_blocks_three](../tasks/stack_blocks_three.md) | [![stack_blocks_three](../figures/stack_blocks_three/geo_curve.png)](../figures/stack_blocks_three/geo.png) | 可解释候选：阶段候选：q3/q5拿起、搬运方块形成高区，仍有其他峰。 |
| [stack_blocks_two](../tasks/stack_blocks_two.md) | [![stack_blocks_two](../figures/stack_blocks_two/geo_curve.png)](../figures/stack_blocks_two/geo.png) | 可解释候选：候选：q8夹持方块调整姿态时宽高区。 |
| [stack_bowls_three](../tasks/stack_bowls_three.md) | [![stack_bowls_three](../figures/stack_bowls_three/geo_curve.png)](../figures/stack_bowls_three/geo.png) | 可解释候选：候选：q0最初接近碗时高，后续大部分低。 |
| [stack_bowls_two](../tasks/stack_bowls_two.md) | [![stack_bowls_two](../figures/stack_bowls_two/geo_curve.png)](../figures/stack_bowls_two/geo.png) | 可解释候选：候选：q0最初接近碗时宽高区。 |
| [stamp_seal](../tasks/stamp_seal.md) | [![stamp_seal](../figures/stamp_seal/geo_curve.png)](../figures/stamp_seal/geo.png) | 较弱：形态清楚但语义弱：与DV同片段峰，无法可靠看清盖章动作。 |
| [turn_switch](../tasks/turn_switch.md) | [![turn_switch](../figures/turn_switch/geo_curve.png)](../figures/turn_switch/geo.png) | 可解释候选：候选：q0接近、q1开关附近调整两个高区。 |
