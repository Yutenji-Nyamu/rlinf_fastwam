# π0 在线成功 BC：实现与训练预算复核（2026-09-05）

历史预算讨论：下方10:01的未运行/未推、M10/正式32×8建议，已被用户后续32×1与[GPU6执行合同](GPU6_SMOKE_CONTRACT_20260905.md)覆盖；当前M4、GPU6 smoke和Git状态只看[本轮账本](GPU6_SMOKE_LEDGER_20260905.md)。本页保留早间审计证据，不据其启动。

本页是[专题 SSOT](../00_RESEARCH_AND_PLAN.md)的证据附件，不是另一份实施计划。此次用户要求恢复上下文、解释代码、讨论并发和 smoke，并只读检查服务器；本轮未运行 GPU 测试、未修改训练配置、未启动或重启任务。

## 1. 昨晚的实现没有丢

10:01 CST 服务器核对：独立树 `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc`，分支 `codex/sz-pi0-online-bc`，HEAD 为官方 `dc9b87cc49334c7516487ead68ebeb060fd7c090`；9个实现文件尚未 commit/push，但本地与服务器 SHA256 逐个相同。不能据此诊断聊天界面为何缺消息，只能确认工程与文档仍在。

现场 `online-bc` 结果目录仅有 `implementation-20260904`；原拟 `pi0-adjust-bottle-smoke-v1` 不存在，所查进程也没有 BC 训练。昨晚通过的是7项基础测试，以及 Ruff、AST、import、Hydra/validate_cfg；**没有真实 GPU smoke，也没有 BC 显存/RAM实测**。

原始核对：[源码/配置/资源审计数据](BC_IMPLEMENTATION_CONFIG_REVIEW_20260905.data.txt)；检查脚本：[只读脚本](../../../local_scripts/sz_online_bc_review_readonly_20260905.py)。

## 2. 哪些文件改了，是否干净

以下路径相对独立 RLinf 树；本地对应 `worktrees/pi0-online-bc`。

| 文件 | 改动及必要性 |
|---|---|
| 新增 `rlinf/data/online_bc.py`，141行 | 收完整 episode，成功后将决策前输入＋提交的动作 chunk 加入累计池；有放回抽 query；归档、回放/RNG恢复、FM mask |
| 新增 `rlinf/workers/actor/fsdp_online_bc_policy_worker.py`，161行 | 复用原生 DAgger 的 FSDP 监督更新，但替换专家数据入口；空池跨rank一致跳过；可选示范混合；冻结检查、checkpoint接点 |
| `rlinf/workers/env/env_worker.py`，+42/−5 | 采集 query 前观察和提交动作，整 episode 成功筛选；排除终止后的未执行 query，保证空消息也按协议发送 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py`，+7/−1 | 原生 FM loss 在最终 mean 前接可选 mask；不传 mask 的原路径不变 |
| `examples/embodiment/train_embodied_agent.py`，+8 | `online_bc` actor 分派入口 |
| `rlinf/workers/rollout/hf/huggingface_worker.py`，+3/−1 | BC 使用原生推理路径，不接 GRPO 中间 SDE 探索噪声；初始 flow 噪声仍随机 |
| 新增 config / tests / `docs/online_bc.md` | 单卡开发配方152行、7项测试所在文件135行、说明83行 |

两项核心新增共302行；原有4文件合计60行新增、7行删除。没有新 critic、teacher、调度器、渲染补丁或依赖升级。**结构较小且职责明确，但不是只改一个 only_success 开关**：RoboTwin 按 chunk 返回观察，DAgger 默认只取专家片段，直接拼接会错位或没有自主数据。

数据语义：监督实际提交的 macro-command，不是测得的关节状态，也不冒称 TOPP 后每个 waypoint 都已物理执行。成功可提前终止物理控制；当前接口没有可逆的原 C50 执行前缀。

训练范围：冻结 PaliGemma VLM，只更新 action expert 与状态/动作/时间投影；不是 LoRA，也不是更新整个 π0。真实可训练参数计数、GPU梯度、同步、模型/优化器恢复仍待 GPU 验收。真实 D0 loader 未验收。

## 3. 正式参数与 smoke 建议：还没有改成启动合同

旧对照是深圳 `grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2`，此次重新读取其服务器 resolved 配置。旧模型同为 `RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`。

| 参数 | 旧 π0 GRPO（已运行） | 在线 BC 正式建议 | 同并发 smoke 建议 |
|---|---|---|---|
| GPU / actor world size | 2卡 | 1卡 | 同正式1卡 |
| train `total_num_envs` | 64，即32/卡 | 32，即32/卡 | **32，不降低** |
| train `rollout_epoch` | 4 | 8 | **1** |
| 每轮训练尝试 | 256 | 256 | 32 |
| 外循环轮数 | 100 | 100候选 | 2，覆盖更新后再采集 |
| C / episode horizon | 50 / 200 | 相同 | 相同 |
| `group_size` | 8 | 1，BC不需要组内优势 | 1 |
| actor micro / global batch | 32 / 1024 | 32 / 1024 | 相同，不偷偷缩训练峰值 |
| 每轮 optimizer steps | 2 | 2 | 2 |
| 每步每卡梯度累积 | 16 | 32 | 32 |
| fixed eval | 共32，16/卡 | 共32，32/卡 | 同正式32 |
| eval / save 间隔 | 5 / 10 | 5 / 10 | 每轮 / 每轮 |
| 推理 flow steps M | **4** | 当前开发配方**10** | 与所选正式一致 |
| LR / weight decay | 5.6e-6 / 0.01 | 2.5e-5 / 1e-10，见下文依据 | 同正式 |
| actor/rollout offload | 开启 | 开启 | 同正式 |
| train/eval env offload | 均关闭 | 均关闭 | 同正式 |

预算换算：有成功池时，BC每轮随机抽1024个query做一次optimizer step，重复2次，共2048次 replay query 呈现，不是遍历累计池两遍。上述 smoke 共64条训练尝试、最多256条新 query、4次更新和4096次 replay query 抽样；评估64条，保存两次。它验收容量与闭环，不判断学习提升。正式100轮对应25,600条训练尝试、最多200次更新。

参数依据分清：

- **任务、SFT、三相机、C50/H200、每卡32环境、每轮256尝试、B1024/U2、fixed32**：便于与本服务器旧 π0 GRPO 对照，都是建议，尚未锁定正式合同。单卡保持256条需串行4→8，吞吐不等于旧两卡。
- **LR2.5e-5、Adam β=(0.9,0.95)、eps1e-8、weight decay1e-10、clip1**：来自锁定官方 `robotwin_adjust_bottle_dagger_openpi.yaml` 的监督 FM 更新配方，不是从 GRPO 推出的最优 BC 超参。官方该配方是多卡、全模型训练，不能照搬容量结论。
- **constant、warmup0**：当前短预算选择。官方 DAgger 的1000 warmup/30000 total 不适合仅200次更新，不能原样搬入；正式效果仍待实验。
- **M10与旧M4不同**：当前来自原生 π0 开发配方。若要严格比较同交互协议，需明确选择是否对齐M4；不能称当前所有行为参数都与旧GRPO一样。BC的 ODE 推理与GRPO的探索方式本身也不同。
- **示范混合**：`demo_weight=0`先测纯在线成功BC；`w>0`传本地 `demo_data_path`，损失 `(online_FM+w*demo_FM)/(1+w)`，不是示范轨迹占比。首版按query均匀抽样，不是episode等权。

昨晚未运行的开发 YAML 是2并行×2串行×2轮，micro1/global4，eval2；它仅适合小闭环，不代表正式并发容量。用户现在提出的“并行不改、串行最小”合理，以上新建议尚未写入该 YAML 或旧启动脚本。

## 4. 单卡、显存与主机内存

**算法上单卡足够；32并行、micro32能否装进80GB卡，目前没有实测答案。** 单卡无参数/优化器跨卡分片，不能从“BC比GRPO简单”推出显存必然更少。

并发与容量有三个不同旋钮：

1. `env.train.total_num_envs / env world size / pipeline_stage_num` 决定每worker并行环境；`rollout_epoch`只增加串行批次、总数据和时长，不增加同时采样数。当前stage=1。
2. `actor.micro_batch_size`决定一次反传的激活量；global batch决定累积与样本量。**原生 DAgger 更新第479–489行先将整个global batch拆成micro列表并全部搬上GPU，然后循环前反传**，因此global batch还影响GPU输入缓存，不仅影响时间。
3. placement/world size、FSDP分片、冻结范围、dtype、offload、checkpointing决定模型/优化器/激活驻留。actor/env/rollout共置一张物理卡不等于只有一个模型进程。

旧 π0 GRPO 历史资源文件2574条采样中，峰值GPU4为78475MiB＝76.64GiB，GPU5为78237MiB＝76.40GiB；这是**旧两卡GRPO实测，不是BC估值**。同为expert-only也仍有VLM推理和环境渲染开销。

32训练环境＋32评估环境且两者不offload，单卡最多保留64个场景；旧两卡每卡为32训练＋16评估＝48个。这也是新smoke应保持正式eval并发的理由，但更不能预先保证装得下。累计成功回放保留图像，且checkpoint再次保存整池，短测不能证明100轮的RAM/磁盘增长可接受；正式前需根据实测每条query字节量、成功率和保存大小估算长程占用。D0开启还可能同时保留online/demo前向图，不能套用w=0的峰值。

**撤回昨晚40–60GiB显存、128GiB RAM作为容量依据**：它们只是未验证的预留估计。现在能确认的是一张H100 80GB作为目标资源，测量加载/采集/更新/评估/保存各阶段显存峰值，以及主机available变化、进程RSS/PSS和回放增长；不把共享RSS相加当独占用量。

10:00现场GPU1/2/3/6/7无compute进程；按用户高编号空卡偏好，可优先考虑GPU7，启动前再查是否空闲。昨晚GPU3选择是当时4–7均在跑的结果，并非必须固定3。本轮不占卡、不启动；也不因为发现Fast退出就擅自重启或移卡。

## 5. 接续边界

下次按选定的正式并发更新并展示完整resolved、精确命令和输出路径；smoke仅缩串行和外循环，覆盖两个完整训练/评估/保存轮次。原2环境合同的30分钟上限不能直接当成32环境测试的足够时长；加载与各阶段耗时需要单独考虑。fatal/OOM/非有限loss立即报告，只处理本任务，不碰shared Ray/其他实验；没有成功样本时明确记录更新未验收，不伪造通过。

本轮只保存本地审计证据、图和交接，没有改服务器源码/依赖/训练，也没有将未执行的测试写成完成。
