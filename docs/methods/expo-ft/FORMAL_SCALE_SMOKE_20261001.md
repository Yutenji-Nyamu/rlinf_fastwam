# EXPO 正式规模短 smoke · 2026-10-01

用户本轮确认：EXPO在深圳2；action expert不用LoRA；方法预算和Q视觉结构贴近论文/固定作者代码；保留RoboTwin三相机/原基座，先不用图像增强。允许短smoke，明确不进入正式训练。

## 本次配置与旧 smoke 的差别

| 项目 | 旧闭环 smoke | 本次正式候选规模 smoke |
|---|---|---|
| GPU | 深圳2物理4，1×H100 80GB | 深圳2物理4–7，4×H100 80GB，数据batch分卡 |
| 环境 | 1个，200动作上限 | 同样1个、200动作上限 |
| 每观测提案 | 8 base＋8 edit | 保持8＋8 |
| TD/Q全局batch | 4 | 64，每卡约16观察 |
| TD π候选批 | B4×候选micro4=16，候选分两段 | B64×候选8=512，分4卡，每卡128；观测micro64 |
| 一次Q更新的串行提案段数 | 2 | 1；一次call的20次Q更新仍须先后执行 |
| Base FM | B1 | 全局B64，每卡16，一次全局AdamW |
| Q/editor计算微批 | 小batch4 | 全局64，分4卡；备用缩micro需新合同/再测，不改变B64 |
| 训练VLA参数 | action expert＋action/time投影 | 保持，无LoRA，视觉/语言冻结 |
| Q视觉 | 逐相机共享Torch50/GN32/平均特征 | 作者结构等价PyTorch：joint9ch、V2 basic(3,4,6,3)、GN4、空间flatten→512 |
| 图像增强 | Q/editor无；FM原生接口自带随机增强 | 按用户本轮决定全部关闭随机图像增强；FM仍采真实noise/time并回传expert梯度 |
| Q replay | 只在线对齐chunk，demo仅供FM | demo＋online同物理步起点、同episode真实完整C10重叠窗 |
| FM数据 | 最后成功窗或固定demo窗 | 所有成功demo/online的真实H50起点均匀采样，全局64 |

GPU合计4×80=320GB，对比论文2×141=282GB，单卡和带宽/算力仍不同。数据并行复制模型、分训练batch，不能自动合并显存；本版PyTorch临时replica及master optimizer分布也不声称等同作者JAX placement。深圳2只读拓扑为各GPU间NV18。

## 一次更新到底做什么

采集时每到重规划点：π0.5批量产生8份H50；取前C10，edit各产生一个改动候选；Q从16份中选一份执行最多10真实动作。

一次learner call：重复20次『从replay新采B64 → 当前π0.5对64个next观察各批量产生8份 → edit为每份产生改动 → Q选优给TD目标 → 更新Q』；随后各1次成功H50的FM更新、editor和temperature更新。

因此一个call有20×64×8=10,240份base proposals及相同数量edit；它们是批量推理，不是10,240个独立Python调用。本次每次TD将512份base批分四卡，并行不能消除20次顺序学习更新。

正式配方记录：前10在线episode warmup（不积欠补训），以后每累计40真实动作1call、episode末执行；Q20/B64，FM/editor/温度各1，base AdamW2.5e-5，小组件Adam3e-4。**本次直接调用一次完整learner作规模测试，不运行10回合warmup或正式cadence，不声称这些调度已验收。** 显存、每call算子/并行参数对齐上表候选方案；以后更改并行参数须再测。

## 短尾问题与任务依据

这不是论文和RoboTwin基座的观点分歧：旧port只把每10步边界写成Q样本。第25步成功时最后边界窗只含5真实动作；补零参与Q/editor会出现训练假尾。按作者保存每个真实动作/观测后，可取第16–25步完整终止窗学到成功奖励。不足C10或跨回合窗口不伪造；真实成功末窗bootstrap=0/valid=1，timeout-touching窗排除。FM只使用真实完整H50。

官方clean50 parquet没有reward/done/final_obs字段；demo末步成功reward=1是基于成功演示来源的明确离线标注，terminal next_obs仅用最后真实obs作吸收态占位且bootstrap=0，不称实录final_obs。在线保存真实final observation。初始50集6060帧提供5610个C10起点、3610个真实H50起点，demo/online单位相同，无固定50/50。

pick_diverse_bottles为延续SZ2已跑通任务/clean50/norm/seed/API而选，非确认当前200动作初始成功率最合适。公开模型卡Easy100回合为59%但未给horizon；旧EXPO在线0/2也不是基座单独评估。正式选任务仍需相同200/C10的base固定seed初评。

## 执行合同与范围

- 独立worktree：`/data/chenyiteng/projects/expo-ft-sz2-20261001/scale-smoke-20261001/source`；branch `codex/sz2-expo-ft-scale-20261001`，从已交付c969b851e36bf5fe4d2b4696f17863370f8ae5ff建立。
- 输入：该scope根`inputs.json`，全部执行源码SHA固定；B64/4GPU/micro/augmentation进入strict restore合同。
- 新cycle：该scope根`rlt-cycle-scale-20261001-v1`，前驱为旧scope已RESTORED的`rlt-cycle`；仅借物理4–7原四RLT，新guardian唯一暂停/归还。旧guardian/helper/cycle不重放，既有helper原字节复用。最初CPU准备使用同名rlt-cycle发现namespace冲突，未执行stop；失败准备目录保留。
- 当前重试命令（已启动）：`/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python -u -B /data/chenyiteng/projects/expo-ft-sz2-20261001/scale-smoke-20261001/source/tools/expo_smoke_owner.py --attempt smoke-v2 --rlt-paused`。该文件为本scope的四卡owner，指定四UUID；调用 `examples/embodiment/smoke_expo_formal_scale.py --inputs …/inputs.json --run …/runs/smoke-v2/fresh`；第二进程`--resume …/runs/smoke-v2/fresh/checkpoint-one-call.pt --verify-only`仅恢复验哈希，不增加第二次学习call。
- 当前输出：`…/runs/smoke-v2`，启动回执`scale-retry-launch-v2.json`、日志`scale-retry-owner-v2.log`；首尝试`…/runs/scale-one-call-v1`与`inputs-scale-one-call-v1.json`保留。原始真实在线replay与checkpoint留服务器。每卡PyTorch allocated/reserved峰值、主进程CPU RSS、来源混合、组件真实变化、完整restore与精确后代cleanup分别留回执。
- 停止：完成1call立即退出；非finite、OOM、绑定/来源/形状/梯度/哈希不一致或超时立即停止，不进入正式循环。失败日志保留，修改新合同再试；旧终止intent不重放。
- 归还：guardian在租约内允许失败后重试；本次复用仍为PAUSED的`rlt-cycle-scale-20261001-v1`，不重启guardian或新建cycle。完成、结束租约或心跳失联后，按唯一冻结CP归还原四RLT，保持任务/方法/seed/原累计3000，验证四组恢复后首轮真正更新。共享Ray/其他用户及物理0–3保持原状。

## 执行记录

2026-10-01：服务器CPU7项核心检查通过；新joint视觉真实RGB224 forward输出1×512；真实demo FM批64×50×14/三相机224形状、同盘replay/RNG恢复通过，CUDA未初始化。修正native sample_actions实例bound-method在DP副本仍指向master的问题；FM按固定原生SFT use_rlt=False分支等价调用PI0Pytorch.forward.mean，避免DP replica的next(parameters)设备查询；随机图像增强仅在EXPO进程关闭，不改安装文件。正式规模GPU结果待实际回执，不能沿用旧14.98GiB当新结果。

首尝试`scale-one-call-v1`在`critic_observation`退出：原生`resize_with_pad_torch`会将B1输出压成三维，随后四维`permute`报维度错误，尚未发生学习。该attempt的owned后代cleanup与物理4–7释放检查已通过。backend只补回被压掉的singleton batch轴，并断言输出形状；服务器以实际原生resize分别检查B1/B2通过，检查未初始化CUDA。

另发现guardian的收集合同只匹配`runs/smoke-vN`及`tools/expo_smoke_owner.py`，首尝试的attempt/owner命名不匹配。已将同版四卡owner在独立scale source保存为预期文件名，并以`smoke-v2`重试；源码SHA重新固定，RLT冻结归还计划及四卡/全局B64等参数不变。

**smoke-v2已完整PASS**：真实200动作＋一次完整B64 learner call（Q20、FM/editor/temp各1），随后独立进程恢复base/core/replay/candidate RNG哈希一致，未做第二call；两阶段owned后代清理与四卡释放通过。Q视觉、critic、target、editor、温度均实际改变；expert抽样参数最大变化2.5004e-5，冻结prefix梯度数0且抽样哈希不变。FM B64均来自成功demo，本在线episode未成功，不能用0/1判断基座成功率或方法效果。

学习call207.0秒；采集/更新/保存合计367.6秒（不含初始化）。物理4–7的2秒间隔nvidia-smi采样峰值46034/38323/38323/38323MiB（约45.0/37.4/37.4/37.4GiB；采样可能漏掉更短尖峰）；PyTorch allocated峰值26.91/26.14/26.14/26.14GiB、reserved峰值39.43/35.88/35.88/35.88GiB，主进程RSS峰值17.70GiB。无OOM或非finite。

用户在恢复核验阶段追加授权：smoke通过直接正式，EXPO退出才恢复低优先级RLT；按论文规模20,000在线真实动作含warmup。旧smoke guardian已先派发四RLT并等待首轮，后续资源交接另走独立正式owner，不重放旧stop/resume。正式合同与现场状态见[正式运行](FORMAL_RUN_20261001.md)，本页的单call通过不代表warmup/cadence长训已验收。

## 来源

- [论文更新/设备](https://arxiv.org/html/2605.25477v2#A4.SS2)
- [NVIDIA H200](https://www.nvidia.com/en-au/data-center/h200/)、[H100](https://www.nvidia.com/en-sg/data-center/h100/)
- [作者多卡mesh](https://github.com/pd-perry/openpi/blob/46407a41183b037313a383ff679683f2773b766d/src/openpi/training/sharding.py#L17)
- [作者更新顺序](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L888)
- [物理步回放](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/data/replay_buffer.py#L328)
- [基座公开评估](https://huggingface.co/SidneyXie/pi05_robotwin#evaluation-results)
