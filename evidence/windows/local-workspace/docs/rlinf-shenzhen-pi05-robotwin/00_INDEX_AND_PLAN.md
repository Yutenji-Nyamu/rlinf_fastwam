# 深圳 RLinf × RoboTwin × π0.5：PPO → GRPO → GRPO-DVAC 计划

> 状态：2026-09-02 08:49 CST，用户要求先停止 π0.5 DVAC、继续观察 clean GRPO。matched `GB1024/update2` DVAC 已精确停在完整Step26，`RLinf_1=0`、GPU6/7释放；日志与产物保留。Control仍是原PID/Ray job，`RLinf=15`，完整Step27后继续运行，shared Ray未动。主机available回升到约1.16 TiB。停止证据见 [`evidence/PI05_DVAC_STOP_FASTWAM_HANDOFF_20260902.md`](evidence/PI05_DVAC_STOP_FASTWAM_HANDOFF_20260902.md)；停止前对比图与数据见 [`evidence/pi05-grpo-matched-u2-live-20260902-brief/`](evidence/pi05-grpo-matched-u2-live-20260902-brief/)。
> 本文是 π0.5 专题的当前单一入口；执行流水与动态实验状态以后放在本目录 `evidence/`，不回填历史推测。

## 0. 结论

推荐顺序是：

1. 官方 π0.5 SFT checkpoint + RoboTwin `adjust_bottle` fixed-32 推理。
2. 两卡 π0.5 PPO：保留官方 PPO/模型语义，只按 8→2 卡等比例缩资源。
3. 两卡 π0.5 GRPO：继承深圳已验证的两卡 π0 RoboTwin GRPO 协议，只替换模型身份及 π0.5 固有字段。
4. 两卡 π0.5 GRPO-DVAC Action-Adv `[0.5,1.5]`：从 clean π0.5 GRPO 复制，只改 DVAC 方法叶。

第一步是官方路线；后两步是有明确依据的深圳派生路线。RLinf 没有公开的 RoboTwin π0.5 GRPO recipe，不能把它们称为官方配方。

## 1. 依据

- [RLinf RoboTwin 官方训练文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)：ready-to-run 的 π0.5 路线是 `adjust_bottle + PPO`。
- [官方 RoboTwin π0.5 PPO YAML](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml)：PPO、模型、环境和 8 卡预算母配置。
- [官方 RoboTwin 评估指南](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html)：OpenPI、RoboTwin、ALOHA、assets 和评估合同。
- [官方 RoboTwin 模型集合](https://huggingface.co/collections/RLinf/robotwin)：π0.5 SFT/PPO checkpoint。
- [官方 LIBERO π0.5 GRPO 示例](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml)：只证明 current RLinf 支持 π0.5 + GRPO；其 LIBERO 环境、H、M、batch 和更新次数不能搬到 RoboTwin。
- [深圳扩展实验总入口](../rlinf-shenzhen-experiment-expansion/00_INDEX_AND_PLAN.md)：现有 π0 PPO、两卡 GRPO、Action-Adv 与服务器隔离经验。

官方公布的 `adjust_bottle/demo_clean` 结果是 π0.5 SFT `85.94%`、PPO `96.09%`。它是实现完成后的结果 oracle，不是深圳两卡必然达到的数值。

## 2. 阶段 A：复用环境，先跑官方 π0.5 SFT

### 2.1 可以复用

- 深圳 source-locked current RLinf；实现前只比较一次 π0.5 相关官方 blob，不为此盲目升级整个仓库。
- 已跑通的 OpenPI venv、RoboTwin `RLinf_support`、assets、ALOHA、三相机和 fixed reset seeds。
- current typed rollout 数据链、FSDP、run-scoped 绝对路径、独立 Ray namespace/worktree 和 `local_shard` checkpoint 经验。

### 2.2 必须独立

- π0.5 专用工作树/分支、模型目录和运行目录。
- 完整 `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` snapshot，放 `/data/chenyiteng/...`，而不是 `/home`。
- checkpoint 自带的 π0.5 `assets`、index 和 `physical-intelligence/robotwin/norm_stats.json`；不能复用 π0 的 norm stats。

PPO 训练只需 SFT checkpoint，不需要 demonstration 数据。官方训练好的 PPO checkpoint 暂不下载；以后需要作 oracle 时再取。

### 2.3 最小闭环

先做一次 official-style SFT fixed-8，只覆盖 checkpoint load、环境和动作输出；成功后直接做一次formal-size的真实 PPO outer step并保存checkpoint。这里不增加额外Gate、不做fresh reload，也不预先重装venv；只有真实import/load暴露缺依赖时才复制专用venv。

## 3. 阶段 B：两卡 π0.5 PPO

### 3.1 不能把 π0.5 当作只换 checkpoint

官方 π0.5 PPO还包含这些模型特定字段：

- `model/pi0_5` 与 `pi05_aloha_robotwin`；
- 5 步去噪 `M=5`；
- `H=C=50, D=14`、三图、`noise_level=0.3`；
- GAE + value head，actor/value LR=`5e-6/1e-4`；
- `update_epoch=5`；
- 官方 resolved offload 叶子原样继承，不手工拼出另一套 offload 组合。

因此首条 PPO 必须从官方 π0.5 PPO 母配置缩放，而不是从 π0 两卡 GRPO YAML 换模型名。

这里的`M5`是`num_steps=5`：每次policy query内部，π0.5用5个flow-SDE去噪步把噪声变成一段H50动作。它不是episode步数、rollout次数或update epoch。官方π0是`M4`，π0.5是`M5`；PPO、GRPO和DVAC都应保留π0.5的M5，不能为了表面齐平把它改成M4。

### 3.2 推荐的两卡缩放

| 字段 | 官方 8 卡 | 深圳 2 卡 | 依据 |
|---|---:|---:|---|
| train env | 256 | 64 | 保持 32 env/rank |
| eval env | 128 | 32 | 保持 16 env/rank，沿用已稳定的 fixed-32 |
| rollout epochs | 4 | 4 | 得到 256 trajectories/outer step |
| query records 上限 | 4,096 | 1,024 | 每条 200-step episode、C50，约4次 query |
| global / micro batch | 2,048 / 32 | **512 / 32** | 保持每 rank batch、累积结构和每轮 optimizer-step 数 |
| update epochs | 5 | 5 | 保留官方 π0.5 PPO 更新语义 |
| eval/save | 10/10 | **5/10** | eval5与深圳两卡曲线同轴；save仍10 |
| checkpoint | upstream 默认 | `local_shard` | 复用已经解决并发保存卡住的通用路径 |

这里 `GB512` 不是随意减小。官方：

$$
4096 / 2048 \times 5 = 10\ \text{optimizer steps/outer step}.
$$

两卡等比例缩放：

$$
1024 / 512 \times 5 = 10.
$$

同时每 rank effective batch 都是 256，gradient accumulation 都是 8。若使用此前草案中的 `GB1024/update5`，每轮只有5次 optimizer step、每 rank batch和累积均变成官方的2倍；那是另一种深圳壳，不是官方比例缩放。因此本计划明确修正为 `GB512`。

### 3.3 π0历史缩放与π0.5缩放的关系

| 运行 | 卡 | trajectories / records | GB × update | optimizer steps | batch/rank | M |
|---|---:|---:|---:|---:|---:|---:|
| 官方π0 PPO | 8 | 1,024 / 4,096 | 2,048 × 2 | 4 | 256 | 4 |
| 深圳π0 PPO | 4 | 512 / 2,048 | 2,048 × 2 | 2 | 512 | 4 |
| 深圳π0 PPO | 2 | 256 / 1,024 | 1,024 × 2 | 2 | 512 | 4 |
| 官方π0.5 PPO | 8 | 1,024 / 4,096 | 2,048 × 5 | 10 | 256 | 5 |
| 推荐π0.5 PPO | 2 | 256 / 1,024 | 512 × 5 | 10 | 256 | 5 |

历史π0并不是从官方8卡一步严格等比例缩到两卡：深圳先做四卡时保持GB2048，使optimizer step从4降到2、batch/rank从256升到512；后来四卡缩两卡时再把数据量与GB一起减半，从而保留深圳四卡的2次optimizer step和batch/rank=512。这是一条已经跑通的深圳协议。

首条π0.5 PPO的目标不同：先复现“官方算法/模型语义的两卡版本”，所以数据量、GB和卡数都缩成1/4，保留官方10次optimizer step、batch/rank=256和accumulation=8。`GB1024/update5`并非不能运行，但它是“深圳batch壳+官方update5”的混合口径，不是严格官方缩放。

如果以后要做π0与π0.5的model-only PPO对照，应再单独构造matched配置；不把这个问题混进首条官方π0.5 baseline。

### 3.4 预期差异

π0.5 的模型前缀与 value 路径更重，且 `M5/update5` 不同于现有 π0 的 `M4/update2`。不能根据 π0 两卡单步时间直接给 π0.5 ETA；第一次 outer step负责测墙钟、显存和主存，而不是改算法参数。

首条 formal 自然取 100 outer steps，便于和现有两卡曲线同轴；它属于后续执行决定，本轮不启动。

## 4. 阶段 C：两卡 π0.5 GRPO

### 4.1 配方来源

用户已明确把首要问题改为“同一 π0.5 模型内，clean GRPO 与 DVAC 如何比较”。因此：

- 模型/环境字段取官方 RoboTwin π0.5 PPO；
- PPO、clean GRPO和DVAC共同使用官方π0.5两卡缩放后的资源/更新外壳；
- 官方 LIBERO π0.5 GRPO 只作为代码支持证据，不作为 RoboTwin 参数来源。

### 4.2 推荐 resolved 合同

- 两卡、`64 train / 32 eval / rollout4`，即256 trajectories/step；
- `G8`，32 groups/step，`filter_rewards=true`；
- `adv_type=grpo`、actor-only、无 value head；
- max 1,024 query records，`GB512/MB32/update5`；
- `H=C=50, D=14, M=5`、`pi05_aloha_robotwin`、actor LR `5e-6`；
- fixed32/eval5/save10/local-shard。

这与π0.5 PPO共同保持每rank batch 256、梯度累积8和每轮10次optimizer step。PPO→GRPO只改变算法必须叶：`G1→G8`、`GAE→GRPO`、actor-critic→actor-only、关闭value head和启用group filter；不再为了跨模型对齐而单独换成π0的`GB1024/update2`。

由于 π0.5 SFT 官方成功率已经较高，G8 中全成功 group 可能更多，真实训练时应观察 filtered-group fraction；首版不因此擅自改 group size或采样量。

clean π0.5 GRPO只需一次formal-size的真实outer-step smoke，确认G8→advantage→actor update→checkpoint路径；成功就继续DVAC，不在这里追加重复评估。

## 5. 阶段 D：π0.5 GRPO-DVAC Action-Adv `[0.5,1.5]`

### 5.1 为什么预计是小增量

current π0 和 π0.5 共用 `OpenPi0ForRLActionPrediction`：去噪步数由配置决定，telemetry 统一输出 `[B,M,H,D]` endpoint；rollout、typed `forward_inputs`、actor Action-Adv 和修正后的 H-sum loss 都是模型通用路径。

所以在最新 Action-Adv superset 上，不需要重写 model、worker、schema 或 loss；实现只新增 π0.5 GRPO主配置，DVAC用同一配置的少量方法覆盖启用。真实 M5 smoke若暴露接口差异，再做一个窄 adapter，不预写兜底。

### 5.2 相对 clean π0.5 GRPO 只允许这些差异

- `logprob_type: chunk_level → action_level`；
- DVAC `mode: off → apply`；
- `application: action_advantage`；
- `weight_min/max: 0.5/1.5`；
- `selected_l=3, warmup=1, recent_window=5`；
- 方法命名、DVAC sidecar和run-scoped输出路径。

其余 G8、env、rollout、batch、update、seed、模型、M5、eval/save必须逐叶一致。

`L3` 是为了保持现有 π0 DVAC 信号定义；π0.5 的 M5可以合法计算最后3步。若改成L5，就是另一个信号实验，不能藏在模型迁移里。

这条smoke需要2个formal-size outer steps：Step1建立recent history，Step2才会出现非均匀`[0.5,1.5]`权重。这是方法功能本身的最小覆盖，不是额外防御流程。

## 6. 预计文件与 Git 组织

建议建立一个独立分支/worktree，例如 `codex/sz-pi05-robotwin-rl`，基于深圳 source lock并包含已验证的 Action-Adv H-sum fix与local-shard支持。

预计新增：

- 官方π0.5 PPO配置的两卡resolved运行packet；
- `robotwin_adjust_bottle_grpo_openpi_pi05.yaml`；
- 同一GRPO配置上的 π0.5 GRPO-DVAC `[0.5,1.5]`方法覆盖；
- 本专题实施账本与精确 resolved 对照。

预计不改：

- `openpi_action_model.py`；
- `huggingface_worker.py`；
- `embodied_fsdp_actor_worker.py`；
- PPO/GRPO loss、GAE、schema、Builder和RoboTwin env。

如果实际实现需要改上述生产 Python，必须先指出具体 π0.5 接口差异和依据，不以“迁移方便”为理由整文件覆盖。

## 7. 已收束的三个口径

1. π0.5 PPO：`GB512/MB32/update5`，不是旧草案的GB1024。
2. π0.5 GRPO：`GB512/MB32/update5 + M5`，与同模型PPO共享资源/更新外壳，只切换GRPO算法叶。
3. π0.5 GRPO-DVAC：`[0.5,1.5] + L3/recent5`，逐叶继承clean GRPO，只改方法叶。

已授权的最小执行顺序是：source/checkpoint → PPO one-step → clean GRPO one-step → DVAC two-step。PPO真实一步已经同时覆盖checkpoint加载、π0.5推理、RoboTwin rollout和更新，不另加重复SFT gate；每层成功就直接进入下一层。

## 8. 实施与 smoke 终态

- 分支：`codex/sz-pi05-robotwin-rl@256eeeb4459b4bd5db85bfc6a0eb315771e8c38c`，已推送；只新增一份π0.5 RoboTwin GRPO主配置，production Python零改动。
- 官方π0.5 SFT按精确revision下载到`/data`，复用现有OpenPI/RoboTwin venv；不新建第二套环境。
- PPO Step1：30分12秒，success `83.20%`，GPU峰约`59.1 GiB/卡`，自然`exit 0`并保存两rank local shard。
- clean GRPO Step1：31分13秒，success `80.08%`，GPU峰约`59.3 GiB/卡`，自然`exit 0`并保存两rank local shard。
- GRPO-DVAC两步：51分50秒；Step2 success `83.98%`，weight mean `1.042`、ESS `0.957`、warmup `0`，证明`[0.5,1.5]`非均匀Action-Adv真实生效；GPU峰约`61.9 GiB/卡`，checkpoint另含两份DVAC sidecar。
- 三段均无fatal/OOM；最重并发时主机仍约`605 GiB` available，GPU4--7原任务未被停止，结束后GPU2/3已释放。

### 8.1 与两卡π0的可比观察

历史两卡π0 clean GRPO首步为约21.75分钟、早期GPU峰约49.2 GiB/卡；π0.5 clean首步约31.22分钟、约59.3 GiB/卡。π0.5约慢43%、多约10 GiB/卡，但这不是纯模型差：π0.5同时使用M5、GB512/update5，π0历史是M4、GB1024/update2，且本次每段都在Step1保存大checkpoint。

### 8.2 formal资源壳建议

首条formal不扩大并发或micro batch：共同保留两卡、`64 train/32 eval/rollout4/GB512/MB32/update5/M5/fixed32/eval5/save10`。理由是smoke已经使用约74%--78%的80G显存；主机虽有余量，但长期RoboTwin EnvWorker存在增长史。若之后要调资源，PPO、clean GRPO和DVAC应共同调整，不能只放大其中一臂。

clean GRPO首步KL/clip为`0.366/0.383`，有限但偏高；一条smoke不足以改配方。正式运行时先观察前3--5步；若clean和DVAC持续同时偏高，再共同降低update，而不是破坏两者匹配关系。

完整逐操作证据见[`evidence/IMPLEMENTATION_LEDGER_20260831.md`](evidence/IMPLEMENTATION_LEDGER_20260831.md)。
