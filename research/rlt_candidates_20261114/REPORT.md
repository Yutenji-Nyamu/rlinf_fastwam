# RLT 候选与任务整理 · 2026-11-14

[打开交互曲线](index.html)。6张图覆盖全部11份候选，另附1机11个RT2任务的训练和评估曲线。

## 现有候选

11份小策略权重均存在、CPU重新加载成功、40个张量均有限，每份约8.3 MiB；状态复核时间15:32。候选的评估成绩来自当时的记录，本次刷新曲线没有新增策略评估。

- 按铃：Clean CP625与加权CP1125均20/20，后续末5评仍100%。
- 拨开关：旧CP270为12/12，CP614为11/12，新从头训练CP62为12/12。新组末5评均76.7%，单次满分之后仍有波动。
- 放风扇：留存的旧Clean CP300为1/20、加权CP250为0/20；后来Clean最高6/20，末5评均7%。这些早期候选用于历史对照。
- Dojo丢瓶子：旧RL CP30为10/32，后来下降；纯BC CP16为63/64，下一次59/64，证明小策略的模仿能力得到修复。
- Dojo倒液体：CP2为0/56，CP22为12/56。11月14日从CP34接续价值修复，曲线保留各段边界。

候选标识及张量检查结果见data.json；$ARTIFACT_ROOT为用户产物根目录占位符。每份小策略需要匹配的基础VLA、Stage1压缩器与配置。

## RT2任务效果

按代表性Clean运行的末5次固定评估均值排序，逐轮训练成功率同时在网页展示。相同任务的不同模型、恢复段、参数版本分别保留；这里整理经验。

| 任务 | 末5评均值 | 单次最高 | 末次 | 判断 |
|---|---:|---:|---:|---|
| 按铃 `click_bell` | 100.0% | 100.0% | 100.0%（R775） | 稳定成功 |
| 调瓶子 `adjust_bottle` | 100.0% | 100.0% | 100.0%（R800） | 稳定成功 |
| 按订书机 `press_stapler` | 98.0% | 100.0% | 100.0%（R1475） | 稳定成功 |
| 拨开关 `turn_switch` | 76.7% | 100.0% | 100.0%（R108） | 成功，但仍波动 |
| 移罐到锅旁 `move_can_pot` | 50.0% | 80.0% | 40.0%（R2550） | 有收益，尚不稳定 |
| 双手抓瓶 `pick_dual_bottles` | 30.0% | 60.0% | 35.0%（R1125） | 部分学会 |
| 鼠标放垫上 `place_mouse_pad` | 11.0% | 15.0% | 5.0%（R1050） | 偏弱 |
| 放风扇 `place_fan` | 7.0% | 30.0% | 5.0%（R850） | 偏弱 |
| 手机放支架 `place_phone_stand` | 5.0% | 35.0% | 0.0%（R1125） | 偏弱 |
| 盖章 `stamp_seal` | 4.0% | 10.0% | 10.0%（R1000） | 失败 |
| 订书机移到垫上 `move_stapler_pad` | 1.0% | 5.0% | 0.0%（R450） | 失败 |

## 历史覆盖与筛选

三机历史有21个去重RT2 RLT任务的正式采集记录。1机11项见表；另外两机10项为：pick_diverse_bottles, place_object_scale, place_a2b_left, place_a2b_right, move_pillbottle_pad, rotate_qrcode, place_object_stand, move_playingcard_away, beat_block_hammer, lift_pot。本轮读取1机留存事件；2、3机这10项依据既有正式派发、采集和恢复记录登记。

原筛选规则：官方任务上限400步；π0、π0.5、Fast-WAM、Motus、LingBot-VA各Clean/Randomized共十个公开成功率等权平均，按低到高选。实际RLT采用200动作。排序前20项已有训练记录，另做过第22名adjust_bottle。短任务表剩余click_alarmclock、place_container_plate、grab_roller暂未找到RLT记录，公开十项均值93.1%、97.2%、99.2%，偏易。

优先复用失败样本明确的place_fan、place_phone_stand、stamp_seal、move_stapler_pad。扩展新难任务时可进入400步以外任务，先用当前π0.5接口的小评估确定初始成功率。move_stapler_pad与press_stapler是不同任务，历史表现也完全不同。

## 数据与来源

- 本次1机扫描155份含实际成功率事件的RLT运行目录；筛掉smoke后选择上述正式运行。
- TensorBoard训练轮为step+1；明确早于训练的step0评估标为训练前。重复step使用时间更新的记录。各恢复段分别绘制。
- data.json包含全部绘图点、候选位置、来源运行目录和历史24任务排序；网页离线可打开，悬停查看成绩，点击任务切换曲线。
- 历史依据：TASK_SELECTION_FOUR_RLT_20260928.md、FOUR_TASKS_RLT_20260928.md、RLT_NEXT_SIX_20261002.md、RLT_SZ3_TIMEOUT_RECOVERY_20261003.md、OVERALL_REFRESH_20261005_1500.md（原RL项目docs/server-admin）。
- [官方任务步数表](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/env_cfg/task_config/_eval_step_limit.yml)、[Fast-WAM Table 3](https://arxiv.org/html/2603.16666v2#A1.T3)、[LingBot-VA Table S1](https://arxiv.org/html/2601.21998v2)。
