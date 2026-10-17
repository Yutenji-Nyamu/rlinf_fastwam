# 已保存的强策略候选

2026-10-17深圳三机只读核验：仅核文件存在、大小、配置和检查点目录，未加载模型或运行新评估。下表是候选，不是已通过同协议强弱验证的专家。

优先复验：pillbottle历史GRPO Clean CP140/160。当前U CP180/140/110也存在；turn_switch历史Clean CP100/60先作候选库存，不默认更强。使用旧Clean有助于减少用U训练出的强策略再验证U选择价值的混杂。

## 历史成绩是否支持“强”

以下来自10月14日已保存的固定32条评估曲线，本轮按配置中的experiment_name与实际checkpoint路径核对关联。评估原始step从0开始，表中轮次为step+1；只是同轮日志，尚未重载保存权重复验。

| 候选 | 现存点 | 初5次→末5次评估均值 | 保存点同轮评估 | 判断 |
|---|---|---|---|---|
| pillbottle Clean C17 | CP140 | 31.25%→54.38% | 56.25% | 优先复验 |
| pillbottle Clean C15 | CP160 | 46.25%→63.75% | 53.13% | 优先复验；R165的75%不能记到CP160 |
| turn_switch Clean C27 | CP10/20/30/40/50/60 | 30.00%→46.25% | CP60：50.00% | 次选 |
| turn_switch Clean C26 | CP10至100，每10轮 | 38.13%→37.50% | CP100：37.50% | 没有明显后期提升，不优先当强策略 |

“初5次”已含训练更新，不是原始策略基线；这些趋势不能替代Weak-only/Strong-only同协议比较。历史曲线和对应证据见[checkpoint-history.json](checkpoint-history.json)。上述四组使用同一底座路径`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`；新评估前仍须核验底座、归一化、动作接口和实际加载结果。

## 保存点清单

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

精确服务器路径、配置证据及每组保留轮次retained_steps见[checkpoint-inventory.json](checkpoint-inventory.json)。上表主要列每组最新保存点，并非所有文件只剩这一个点。其中部分GRPO配置未在本次浅层候选探测找到，任务名只是路径线索，正式使用前必须从原resolved配置与归一化文件复核。

RLT还保留：adjust_bottle、move_can_pot、turn_switch、move_stapler_pad、move_pillbottle_pad、pick_dual_bottles、pick_diverse_bottles、place_mouse_pad、place_fan、place_phone_stand、click_bell、stamp_seal、press_stapler、place_object_scale、place_object_stand、move_playingcard_away、place_a2b_right/left、rotate_qrcode、beat_block_hammer、lift_pot。具体机器、轮次和独立运行都在JSON中；不把同任务不同恢复段拼成新权重。

RLT通常需要token/critic等配套；EXPO latest内含基础策略与Editor/Q等状态。表中EXPO先保留turn_switch系统策略候选，不能把N8+Editor成绩当成仅基础π0.5的成绩。

原始W应是对应训练使用的同一任务可用π0.5底座，不是随意下载的另一个π0.5版本。候选S经同协议全程评估后，再固定checkpoint进行High/Low/Random切换；选checkpoint的种子与最终报告种子分开。
