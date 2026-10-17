# 已保存的强策略候选

2026-10-17深圳三机只读核验：仅核文件存在、大小、配置和检查点目录，未加载模型或运行新评估。下表是候选，不是已通过同协议强弱验证的专家。

优先：pillbottle历史GRPO Clean CP160/140、当前U CP180/140/110；turn_switch历史Clean CP100/60。使用旧Clean有助于避免用U训练出的强策略再验证U选择价值的循环论证。

| 机 | 任务/路径线索 | 保存点 | 大小GiB | 运行标识 |
|---|---|---:|---:|---|
| sz1 | 未从配置确认 | latest | 8.18 | `expo-fm-u-tau2-sz1-g67-60000-1010-v2` |
| sz1 | 未从配置确认 | latest | 8.18 | `expo-fm-u-tau1-sz1-g45-60000-1010-v2` |
| sz1 | 未从配置确认 | latest | 8.18 | `expo-clean-entropy-b64-sz1-g45-60000-1015` |
| sz1 | 未从配置确认 | latest | 8.18 | `expo-editor-u-tau2-sz1-g45-60000-1013` |
| sz1 | 未从配置确认 | 180 | 7.94 | `grpo-u-tau1-notrick-pillbottle256-sz1-g67-200-1014` |
| sz1 | move_pillbottle_pad | 60 | 7.94 | `pi05_sidney_pillbottle_grpo_dvac_adv_w05to15_formal200_phys45_20260906` |
| sz1 | move_pillbottle_pad | 140 | 7.94 | `pi05_grpo_clean_half128_seed42_formal200_phys45_20260911_v1` |
| sz1 | move_pillbottle_pad | 160 | 7.94 | `pi05_sidney_move_pillbottle_pad_grpo_formal100_2gpu64x4_g8_b1024_u2_m10_noise0p5_h200_fixed32_phys45_v1` |
| sz2 | 未从配置确认 | 140 | 7.94 | `grpo-u-tau2-notrick-pillbottle256-sz2-g45-200-1014` |
| sz2 | move_can_pot | 10 | 7.94 | `sz2-pi05-grpo-move-can-pot-clean-256-seed42-restart200-phys45-20260926-v1` |
| sz2 | turn_switch | 60 | 7.94 | `sz2-pi05-grpo-turn-switch-clean-half128-seed42-formal200-phys45-20260922-v2` |
| sz2 | move_can_pot | 30 | 7.94 | `sz2-pi05-grpo-move-can-pot-clean-256-resume10to200-phys45-n32-20260927-v1` |
| sz2 | turn_switch | 100 | 7.94 | `sz2-pi05-grpo-turn-switch-clean-256-seed42-formal200-phys45-20260923-v1` |
| sz2 | 未从配置确认 | 110 | 7.94 | `grpo-u-tau3-notrick-pillbottle256-sz2-g67-200-1014` |
| sz2 | 未从配置确认 | latest | 8.18 | `expo-ubc-b64-tau2-sz2-g67-60000-1016` |
| sz3 | 未从配置确认 | latest | 8.18 | `expo-ubc-b64-tau3-sz3-g67-60000-1016` |
| sz3 | 未从配置确认 | latest | 8.18 | `expo-ubc-b64-tau1-sz3-g45-60000-1016` |
| sz3 | 未从配置确认 | latest | 8.18 | `expo-editor-u-tau3-sz3-g67-60000-1013` |
| sz3 | 未从配置确认 | latest | 8.18 | `expo-editor-u-tau1-sz3-g45-60000-1013` |

精确服务器路径、配置证据及全部候选见[checkpoint-inventory.json](checkpoint-inventory.json)。其中部分GRPO配置未在本次浅层候选探测找到，任务名只是路径线索，正式使用前必须从原resolved配置与归一化文件复核。

RLT还保留：adjust_bottle、move_can_pot、turn_switch、move_stapler_pad、move_pillbottle_pad、pick_dual_bottles、pick_diverse_bottles、place_mouse_pad、place_fan、place_phone_stand、click_bell、stamp_seal、press_stapler、place_object_scale、place_object_stand、move_playingcard_away、place_a2b_right/left、rotate_qrcode、beat_block_hammer、lift_pot。具体机器、轮次和独立运行都在JSON中；不把同任务不同恢复段拼成新权重。

RLT通常需要token/critic等配套；EXPO latest内含基础策略与Editor/Q等状态。表中EXPO先保留turn_switch系统策略候选，不能把N8+Editor成绩当成仅基础π0.5的成绩。

原始W应是对应训练使用的同一任务可用π0.5底座，不是随意下载的另一个π0.5版本。候选S经同协议全程评估后，再固定checkpoint进行High/Low/Random切换；选checkpoint的种子与最终报告种子分开。
