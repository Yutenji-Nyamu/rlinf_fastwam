# 深圳 Fast-WAM × current RLinf × GRPO 迁移计划

2026-09-05当前讨论入口：[π0.5接续与Fast线程fatal、学习链审计](evidence/TRAINING_DISCUSSION_PI05_FASTWAM_20260905.md)。Fast scene-fence-v3完整17步后PyGILState_Release退出；完整栈/实际so核对与固定owner线程候选见该文§2，尚未实施新修复。学习链新增BF16全action参数/Adam moments及40更新后局部权重稀疏变化证据，排序与证据边界见§4—5。旧review的“逐sample denoise index”与MB2日志解读以本轮部署源码更正为准；本次仅只读，不自动续训。以下09-04状态均为历史。

更新时间：2026-09-04（23:48只读刷新：scene-fence-v3完整Step10，Sidney完整Step69）

最新[23:48实验、整机与图](evidence/CURRENT_TRAINING_HEALTH_20260904_2348.md)：Fast训练46/256、近5步22.03%；fixed Step5/10均11/32，暂未见提升，Step10双DCP分片及metadata在。
Sidney训练178/256、MA10=64.34%；fixed Step60升至24/32，Step65回19/32，训练上升但评估尚未稳定；Step60双rank/full在。均未恢复测试。
两run继续进入下一步，所查driver fatal/OOM等0；GPU1/2/3空闲，RAM available761.7GiB、memory PSI0，/data余1.07TiB。三树clean、补丁仅Fast双Env加载；无服务器改动或新增监控。以下19:11及更早状态均是历史。

最新[19:11实验、整机与图](evidence/CURRENT_TRAINING_HEALTH_20260904_1911.md)：Fast Step1—3训练65/41/88 /256，已越过旧Step2边界；无fixed/checkpoint，尚不能判断长程修复或学习收益。
Sidney Step58训练158/256，MA10=60.39%；Step55 fixed19/32追平历史最好，Step50双rank/full在。两run所查fatal/OOM等均0。
RAMavailable0.872TiB、memory/io PSI=0；GPU6/7约56GiB/卡，Fast采样峰值63.18GiB。源锁/clean与Env-local映射不变，仅只读。
下一次关注Fast Step5评估、Step10保存及RAM趋势；以下16:56为修复后首次刷新历史。

本轮唯一实施入口：[scene-fence窄修/测试/重启账本](evidence/SCENE_FENCE_FIX_RESTART_LEDGER_20260904.md)。
用户已明确授权本项修复、测试、推送、替换旧run；Fast分支`codex/sz-fastwam-current-rlinf-grpo@62526cc95047c8a4a6e948a76be8eeec8a3926de`已推personal。
修复仅timeline render补reset/submit既有scene-access fence；锁定原svulkan2/headers，采用Env-local原生符号替换，
不升级OIDN/SAPIEN，不改shared wheel，不恢复旧Python cache/reset补丁，不附加GIL/timeout等独立修复。
2场景×64帧×3相机真实短测通过、进程exit0，LD_DEBUG验证原库目标虚函数实际绑定到shim；不代表长程或唯一根因已证实。
旧v1在16:07按owned tree/精确RLinf_1 actors清退，目录保留、无checkpoint。v2在16:21因global preload干扰PyTorch ZIP读取退出。
CPU三组对照定位加载方式后，删除LD_PRELOAD/新增LD_LIBRARY_PATH，只在RoboTwin初始化前RTLD_LOCAL加载；
384图及渲染前后原SFT archive读取复测通过。16:38新v3从原SFT开始：
`fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3`。
完整resolved对旧v1仅6处路径/名称变动，256轨迹、并发32、GB1024/update2/每步4优化器调用、fixed32/eval5/save10等完全不变。
wrapper/PGID1568962、observer1568964、driver1568973、GPU6/7、namespace RLinf_1。正式库为
`/home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so`。
16:56只读刷新：v3首步采样从1/8推进至2/8，完整0步，无所查异常；仅双EnvWorker mapped，Actor/Rollout/driver/Sidney均未加载。
Sidney完整52步、153/256，下一步采样2/4；Step50评估16/32，双rank与full权重在，原进程保留。
Fast/RoboTwin/Sidney三树HEAD及clean状态、原库/补丁hash现场复核未变；RAMavailable约0.98TiB，本轮未改服务器。
[最新回顾/整机与可视化](evidence/SCENE_FENCE_FOLLOWUP_20260904_1656.md)，[此前启动简报](evidence/SCENE_FENCE_RESTART_AND_HEALTH_20260904.md)，[完整逐操作账本](evidence/SCENE_FENCE_FIX_RESTART_LEDGER_20260904.md)。
后续先现场刷新新v3进度、旧Step2失败边界、资源及fatal；新run尚不能判断学习改善或长程修复效果。原50h ETA不沿用。

### 前轮调查与旧v1合同（历史，现由上节v3替换）

最新[noOIDN与scene fence因果审查](evidence/FASTWAM_NOOIDN_SCENE_FENCE_CAUSAL_REVIEW_20260904.md)：
15:17 Fast仍完整Step1、Step2停滞，GPU6/7为0%，无fatal/OOM/exit_code；Sidney完整Step48继续。
15:25实际加载库磁盘反汇编确认：svulkan2 timeline render重载漏接已有scene-access fence，
同场景多相机连续提交可能过早重用场景命令缓冲/TLAS；OIDN同步execute可能掩盖此缺口。
这是已确认的协议缺口和更具体主因候选，尚未得到本次卡死帧的因果闭环。并非最终相机signal被删。
主修候选转为补齐现有scene fence协议（新证据§4–5），不先改线程模型或整库升级；GIL/有限超时只是独立传播问题。
本轮未改生产/安装/编译或启动GPU验证，未attach暂停/停止训练；后续执行需授权，shared Ray/Sidney不动。
前轮[GIL等待链、历史并发与14:50 TB/checkpoint证据](evidence/FASTWAM_STEP2_STALL_INVESTIGATION_20260904.md)保留；
本轮未重查TB/checkpoint，不将旧文件快照当作当前刷新；不能据一步判断学习效果，也不自动套回旧补丁。
此前[14:19图与简报](evidence/CURRENT_TRAINING_HEALTH_20260904_1419.md)保留为历史可视化，不能替代本次现场。
下列12:43/12:49为启动记录，合同仍有效，但原50h ETA因停滞不再适用。

继承的旧执行合同：[clean/noOIDN fresh100账本](evidence/CLEAN_NOOIDN_FRESH100_LEDGER_20260904.md)；本轮仅scene fence执行变更见顶部账本。
用户已授权从头重跑并在启动前追加“并行不变、串行采样翻倍”：32env×rollout8=256轨迹，
G8、GB1024/MB2/update_epoch2、LR5e-6、noise0.3、H32/C24/M10、fixed32/eval5、DCP/save10不变。
实际optimizer calls随样本翻倍为4次/步；这不同于历史256档的GB2048/2次，不混淆。
新RoboTwin `codex/sz-robotwin-clean-oidn-off@f3e30a83365c`从0008ae6仅加BaseTask开关，
vector_env与干净基线一致；旧生命周期补丁不进入本run，但旧分支原样保留。train/eval均none，无升级。
新run=`fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1`，
GPU6/7、wrapper1052625于12:43启动；原始release SFT、resume=null，目标100步。
12:49两rank源路径与原始权重核验通过，GPU6/7约54.4GB/卡，所查OIDN/fatal/OOM/Traceback=0；
已进入首步8轮采样，尚未验证完整更新显存、首次DCP与长程稳定性。新RoboTwin分支已推用户fork。
32x4 packet只准备、从未启动。旧256有OOM记录，本次显存待实测，若OOM停止、不自行调参。
下面OIDN小对照与旧33步run均为历史，不能覆盖当前合同。

前轮[OIDN开关试验账本、参数与图](evidence/OIDN_TOGGLE_TRIAL_LEDGER_20260904.md)：
独立RoboTwin副本只在BaseTask两处接入ray_tracing_denoiser，默认oidn；无须先修C++即可显式选none。
独立分支`codex/sz-robotwin-oidn-toggle@b76d4ed`已推送用户fork，未切换现役训练代码。
原始SFT权重、开/关各3回合，exit0，无本次fatal/OOM；开1/3、关0/3，结构相近但颗粒明显、首块动作小幅变化。
仅小样本观察，不判定成功率/长程稳定性；旧vector_env补丁保留，未恢复Step30或启动GRPO，GPU6已释放。

历史状态：current实现、push与32-env真实strict-resume smoke已完成。旧128-trajectory原v1 Step10续训采用
RoboTwin renderer-life补丁，完成Step33后于09-03 23:51 CST异常退出；Step34先OIDN pthread键分配失败，
后invalid handle/Python autoTSSkey fatal。两rank栈证明补丁实际执行；跨过Step15不等于长程修复完成。
Step30 DCP文件仍在，未做恢复测试。GPU6/7采样时无该run进程；这段为训练终态，未续训/重启shared Ray。
最新调查、健康、补丁边界见[09-04调查](evidence/OIDN_RECURRENCE_INVESTIGATION_20260904.md)。
10:13进一步源码讨论见[补丁去留与干净修复](evidence/OIDN_CLEAN_FIX_DISCUSSION_20260904.md)：
clear_cache不是强制销毁renderer，旧根因定性过强；resolved频率已为1。暂不整体回撤，
原生key/device持有与释放仍需定位，优先审查OIDN上游释放修复；本轮没有实施/升级/运行测试。
最新[实验可视化、π0/π0.5逐叶比较与最小修复复核](evidence/FASTWAM_LEARNING_AND_MINIMAL_FIX_REVIEW_20260904.md)：
Fast主线首/末10步训练均值29.14/28.75%，fixed32末次仍14/32；256档18/32仅单点、随后OOM。
有效官方π0/π0.5对照是异任务adjust_bottle，同任务Sidney短跑为2/32，不能简单排模型名次。
最新[所有权修复与有/无OIDN小对照讨论](evidence/OIDN_OWNER_AND_ONOFF_FOLLOWUP_20260904.md)：
用户选择旧Python补丁先保留、后续审计删去无必要部分；关闭OIDN为备选，允许先做1—2例画面/动作对照来评估代价。
该前轮讨论时尚未运行GPU对照；本轮已明确授权并完成上方小尝试，未整库升级。上层enableDenoiser初始化失败时会静默退无降噪，
因此错误传播须一起覆盖caller；原生最终析构/key释放是否失衡仍待证，不能盲回移上游DeviceGuard补丁。
09-02的256-trajectory扩量启动是历史事件，见[启动账本](evidence/CURRENT_PLAIN_FORMAL100_LAUNCH_LEDGER_20260902.md)，不代表现役运行。

## 0. 一句话结论

不重做 Fast-WAM standalone，也不整搬 AutoDL 旧 adapter。以深圳 current RLinf 为壳，窄接 official
Fast-WAM `7faa711...` 的 tensor-cache 接口，首线做 `move_stapler_pad + critic-free GRPO`。首条128档已经
提供“query-record匹配π0”的结果；09-02曾扩量到`32 train env × rollout8 / G8 / 256 trajectories / 2048 query transitions`，
用于检验outcome样本量假设。此后按用户选择回到原128档修复/续训，最新终态以上方09-04调查为准。

实现提交锁：远端分支`codex/sz-fastwam-current-rlinf-grpo`，implementation/start=`7b2331c55d14397cfb4cb16181470ddc8afae44a`；
09-04现场HEAD=`4faade1d50bf21d1caf1b8a4e5f89282a810208a`，后续包含轻量evidence，working tree clean。
GPU2/3真实smoke完成`fresh Step1 DCP save → 全新进程load → Step2 update/save`；两轮均32 trajectories、
fixed32、有限梯度并`exit0`，两代checkpoint各有`.metadata + 2 distcp shards`。峰值约62.5 GiB/卡，
当时GPU4--7现役训练未受影响。完整命令、问题与证据见
[实施流水账](evidence/CURRENT_IMPLEMENTATION_AND_SMOKE_LEDGER_20260831.md)。

## 1. 已有事实与来源锁

| 来源 | 锁定版本 / 结果 | 本轮用途 |
|---|---|---|
| [深圳 current RLinf](https://github.com/RLinf/RLinf) | official `7d07a421...` 系列已验证两卡 π0 GRPO | typed trajectory、actor、GRPO loss、FSDP、sync、DCP 的目标壳 |
| [深圳 official Fast-WAM](https://github.com/yuantianyuan01/FastWAM) | source `7faa71108368...`；release `8eaceeb24c3c...` | 模型、tensor cache、预处理、checkpoint、H/C/M/D 的唯一 current 真值 |
| 深圳 standalone | adjust/move/turn/pick=`16/11/10/12`，合计49/64；fatal=0 | B=1 action 与视频结果 oracle；不重新安装或下载 |
| AutoDL 旧集成 | RLinf `6d0db56...`；Fast-WAM `45d8e145...`；集成 `768e0243...` | Flow-SDE、action-only、replay、配置预算与故障经验 |

官方 current 模型合同保持：`H=32`、实际执行 `C=24`、去噪 `M=10`、动作 `D=14`，旧 release 推理显式使用
`sigma_shift=5.0`。首版 episode 使用 `192=8×24` actions，避免 200-step 末尾8步产生短尾或静默丢弃。

## 2. 旧实现到底有多大

旧集成 squash commit 是 27 files、`+6348/-2`，但不能把这个数字都理解成算法代码：

- 生产核心：9 files，约 `+2595/-1`；
- 配置：约1562行；
- launcher/resource monitor：约699行；
- tests：约1417行；
- 其余是文档与忽略文件。

生产核心的职责是：

1. `robotwin_adapter`：三相机、14D qpos、state/action stats。
2. `builder`：官方 checkpoint/config、模块去重、action-only trainable set。
3. `fastwam_rl`：shifted schedule、一次随机 Flow-SDE transition、behavior old logprob、actor同 transition replay。
4. `fastwam_policy`：旧 RLinf policy/rollout/replay 桥。
5. `export`：将训练状态导回 official Fast-WAM schema。
6. 三个很小的 RLinf opt-in 接点：registry、rollout train/eval mode、FSDP input cast。

通用 EnvWorker、GRPO advantage/loss、runner 和 checkpoint manager 在旧实现里也没有被重写。这说明方法切口清楚，
但模型 adapter 本身并不是一份 YAML。

## 3. 哪些直接复用，哪些必须适配

### 3.1 直接复用的语义

- official 三相机拼接、14D absolute qpos、normalization；
- `H32/C24/M10/D14` 与 192-action episode；
- 单一 stochastic denoise transition 的 Flow-SDE 密度；
- rollout old logprob 与 actor replay 使用同一个 transition；
- 只训练 action expert；video/proprio/T5/VAE/stats 冻结；
- `move_stapler_pad` 与 clean task 语义；G4 仅是旧 4-env 资源配置，不作为模型语义复用；
- standalone B=1 推理作为 parity oracle。

### 3.2 因 current 变化而必须适配

| 变化 | 旧接口 | current接口 | 迁移动作 |
|---|---|---|---|
| Fast-WAM cache | 每层dict KV cache | `video_cache_k/video_cache_v` tensor cache | 重接 conditioning/velocity，不改Flow-SDE公式 |
| RLinf rollout数据 | tuple/dict与旧 trajectory | `PolicyOutput → ChunkStepResult → Builder → Trajectory` | 所有训练字段放进 typed `forward_inputs` 并随Builder重排 |
| 模型输出 | 旧policy tuple | env `actions[B,24,14]` + result | 明确physical action与normalized model action，只decode一次 |
| actor replay | 旧BasePolicy桥 | current `default_forward(...)` | 返回逐action `logprob[B,24,14]`，通用GRPO loss不变 |
| rollout mode | worker硬编码已知模型 | current仍有同一缺口 | 增加default-off capability，只给Fast-WAM传train/eval |
| FSDP输入 | 旧Fast-WAM保留FP32 chain | current FSDP2默认cast inputs | 增加config开关，默认不变，Fast-WAM关闭cast |
| 模块树 | official对象有多处alias | current optimizer/sync/checkpoint按名字处理 | 规范为唯一canonical `mot.*`树，只开放action expert |
| checkpoint | 旧DCP路径 | current FSDP2官方DCP路径 | 直接复用`get/set_state_dict + DCP`，不新写保存协议 |

### 3.3 明确不搬的内容

- 不搬旧 dict-cache 代码；
- 不覆盖 current schema、Builder、EnvWorker、actor、advantages、loss、runner；
- 不先接 PPO value head；
- 不把旧 standalone 环境改造成 RLinf 环境；
- 不在首版实现 200-step 短尾 mask、proprio训练或 Fast-WAM DVAC；
- 不把没有 fixed eval 的旧 train-rollout 上升写成已验证收益。

## 4. current 最小生产改动面

建议从 current RLinf source lock 建独立 `codex/` branch/worktree：

```text
新增 rlinf/models/embodiment/fastwam/
  __init__.py
  builder.py
  robotwin_adapter.py
  fastwam_rl.py
  fastwam_policy.py

小改
  rlinf/models/__init__.py                 # lazy register
  .../huggingface_worker.py                # default-off rollout-mode capability
  .../fsdp2.py                             # config-driven input cast，默认true

新增
  Fast-WAM model YAML
  move_stapler train/eval YAML
  两卡GRPO主YAML与薄launcher
```

deploy export 可以在首个训练 checkpoint 之后补，不阻塞 GRPO 真实更新。plain GRPO 没有 DVAC history 或算法私有
sidecar，保存/恢复沿用 current DCP。

## 5. 128与256分别匹配π0的什么

Fast-WAM 每条192-action trajectory有 `192/24=8` 个query；π0每条200-action trajectory最多有
`200/50=4` 个query。因此 trajectory 数不能直接横比。

| 协议 | trajectories | query transitions | group | groups | GB/update | presentations |
|---|---:|---:|---:|---:|---:|---:|
| AutoDL旧Fast-WAM GRPO | 64 | 512 | G4 | 16 | 128×1 | 512 |
| 深圳两卡π0 GRPO | 256 | ≤1024 | G8 | 32 | 1024×2 | 2048 |
| **首条Fast-WAM候选** | **128** | **1024** | **G8** | **16** | **1024×2** | **2048** |
| 历史Fast-WAM扩量档（有OIDN） | 256 | 2048 | G8 | 32 | 2048×2 | 4096 |
| **09-04当前noOIDN档** | **256** | **2048** | **G8** | **32** | **1024×2 epochs** | **4096** |

当前档只按最新授权翻倍rollout、不改GB，所以每epoch有2个GB，每步4次optimizer calls；
历史GB2048档是每步2次，二者不能混写为同一个256实验。

128这一档同时做到：

- unique query transitions 与深圳两卡π0相同；
- group size与深圳两卡π0同为G8；
- 每轮样本呈现量相同；
- 保留Fast-WAM模型已验证的micro-batch语义。

它不能同时匹配π0的32个group：128 trajectories在G8下只有16组；若要同时匹配32组，就必须使用256
trajectories。首条优先固定1024条新query数据和G8组内估计器，不为匹配group数量把数据量再翻倍。

这里把global batch更新为深圳两卡π0同款 `GB1024/update2`：1024 records每个epoch形成1次optimizer call，
两轮共2次，累计呈现2048条records。Fast-WAM模型特有的`micro_batch_size=2`与
`model_forward_batch_size=2`保持不变；global batch不会把1024条一次放入显存，而是做更深的梯度累积。

256会把transitions、groups、presentations和顺序采样时间都再翻倍。它回答的是“与π0匹配独立episode与G8
group后，Fast-WAM是否更稳”，不再匹配π0的query-record数。用户已在128档完成14步后选择这条扩量实验。

## 6. 首条运行口径建议

| 项目 | 推荐 | 依据 |
|---|---|---|
| 模型/任务 | official Fast-WAM / `move_stapler_pad` clean | standalone 11/16，有学习余量；adjust已饱和 |
| GPU | 两卡 | 与旧adapter和深圳对照一致 |
| env × rollout | `32 env × rollout4=128 trajectories`，即16 env/卡 | 与深圳两卡π0保持rollout4；因每轨迹query翻倍，把env减半 |
| episode | 192 actions | 8个完整C24 query，无短尾 |
| group | G8，16 groups | 与深圳两卡π0一致；社区Fast-WAM也有G8先例；旧G4只是4-env并发下的整除选择 |
| trainable | action expert only | 旧真实训练语义；避免“配置写proprio但replay未重算”的伪训练 |
| optimizer | LR/noise沿旧Fast-WAM依据；GB1024/MB2/update2 | GB/update匹配深圳两卡π0，micro保留Fast-WAM资源合同 |
| eval | fixed32/eval5 | 继承深圳两卡壳；旧move结果缺fixed eval，是最大证据缺口 |
| save | current `dcp` | Fast-WAM使用FSDP2；DCP保留DTensor与optimizer的分布式state-dict合同 |

256档保持32-env/G8和其他算法叶不变，把`rollout4→8`，并把`GB1024→2048`；这样得到32个G8 group与
2048 query records，同时仍保持“每个update epoch完整看一遍当前数据、每步2次optimizer call”，没有把
数据翻倍与optimizer-call翻倍混在一起。

## 7. 联合环境与主要风险

不污染两套现役模型/source：

- official standalone：Py3.10 / Torch2.7.1+cu128，只读oracle；
- current RLinf/OpenPI：Py3.11 / Torch2.11+cu129，不原地安装Fast-WAM；
- 复用current RLinf venv，通过source-locked `PYTHONPATH`接入official Fast-WAM；只补official runtime缺失的
  四个纯Python包精确版本，没有改变Torch/CUDA/Ray/RoboTwin。

最终Torch固定后再编译CuRobo/Warp等CUDA extension。当前最大工程风险是这套联合runtime与tensor-cache/FSDP边界，
而不是GRPO advantage公式。

其余高信息风险只有：

1. B=1 standalone action与batched adapter输出不一致；
2. old/new logprob在更新前ratio不接近1；
3. canonical module去重失败导致optimizer/sync/checkpoint重复参数；
4. 200-step短尾被静默丢弃；
5. action expert没有真实梯度/权重同步；
6. 与其他RLinf job并发时namespace、worktree、输出路径或Ray actor互踩。

## 8. 实施与最小验收顺序

1. 建独立 current worktree与joint venv，完成official checkpoint真实load。
2. 一次连贯实现五个adapter模块和三个default-off接点。
3. 少量集中检查：shape/stats、B=1 action parity、old/new ratio、action-expert gradient与typed replay对齐。
4. 唯一真实 smoke 直接使用目标并发 `32 env × rollout1 / G8 / 192 actions`：共32条trajectory、
   256条query transition，使用 `GB256/MB2/update1`，覆盖一次GRPO backward/update、sync和fixed32；
   checkpoint另用同预算做`Step1 DCP save → fresh load → Step2 update/save`严格闭环。
5. 这一步验证的是16 env/卡的真实并发、typed replay和完整训练闭环，不把 smoke 本身跑成一轮正式128预算；
   通过后 formal 只把 `rollout1→4` 并切到 `GB1024/update2`。若失败，再按实际OOM/Vulkan/主存位置
   降低env并发、等比增加rollout以保持formal 128，不预建多级fallback。
6. 128 formal有稳定结果后，才决定是否扩到256或接PPO/DVAC。

## 9. formal前的剩余决策

算法与迁移接口已清楚。128档获得与深圳π0相同的1024 query records，但只有一半独立episode/G8 groups；
当前256档改为与π0相同的256 episode/32 groups，同时产生2048 query records。它已在GPU6/7 fresh启动；
尚未完成Step1，因此还没有新效果结论。

192-step、G8、action-only、tensor-cache、current typed path和DCP都有直接依据，不建议作为首轮自由变量。

### 9.1 Action-DVAC-Adv `[0.5,1.5]` 已实现并通过真实恢复 smoke

- 独立分支：`codex/sz-fastwam-action-dvac-adv`；远端 exact HEAD：
  `a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7`。
- 模型侧复用既有 M10 velocity forward，opt-in 记录 FP32 `z=x-tv` 并裁到执行 C24，得到
  `[B,10,24,14]`；L5 生成 `[B,24]` DVAC，不增加模型 forward，也不改变普通 GRPO 动作。
- actor 侧沿用 current 已验证的 recent-5、显式 `[0.5,1.5]` 映射、action-level advantage 与
  H维求和/query均值 loss 聚合；DCP 后附每 rank DVAC JSON sidecar，恢复时不重置 history。
- GPU2/3 两步真实 smoke 已完成：Step1 warm-up 全1并保存完整 DCP；全新进程恢复完成 Step2，
  权重 `min/max/mean/std=0.570/1.500/0.986/0.217`、ESS=`0.954`，梯度/loss有限，第二代
  DCP与sidecar完整，两进程均 `exit0`。峰值约64.0 GiB/卡，退出后GPU2/3释放。
- formal 尚未启动。其候选与本页 plain Control 保持相同的
  `32 env×rollout4/G8=128 trajectories`、1024 query、GB1024/MB2/update2、fixed32/eval5和DCP；
  只改变 DVAC endpoint、Action-Adv 与方法命名/输出路径。

逐操作证据见
[Action-DVAC-Adv实施账本](evidence/CURRENT_ACTION_DVAC_ADV_IMPLEMENTATION_LEDGER_20260831.md)。

### 9.2 pi0-style offload 单步资源 smoke

2026-09-02 在原 GPU6/7 上，只改变 phase offload 与物理并发，保持 256 trajectories、32个G8 group、
2048 query records、GB2048/MB2/update2 和 H32/C24/M10 不变：

- `32 env × rollout8` 完成全部 rollout 后，在 actor update 达到 `81,017/81,011 MiB` 并 OOM；
- 授权 fallback `16 env × rollout16` 完成真实 `rollout -> update -> fixed32 eval -> DCP save`，
  `exit0`，峰值 `75,356 MiB/卡`，两份约14.45 GB shard与`.metadata`完整；
- Step1 train=`75/256`、fixed32=`14/32`，KL/clip/grad=`.0026/.020/6.308`，数值有限；
- 退出后GPU6/7为空、独立namespace为0、shared Ray未重启。

这证明16个常驻train env可以承载一个完整外层step，但还不能直接作为长期formal配置：fixed eval后
EnvWorker增至约`27,266 MiB/卡`，`max_steps=1`没有覆盖下一轮actor update。下一次若继续优化
OIDN生命周期，应优先讨论“train env常驻、eval env仍offload”的窄组合，而不是宣称四项都按pi0设置
已经安全。完整证据见
[pi0-style offload资源smoke账本](evidence/PI0_STYLE_OFFLOAD_RESOURCE_SMOKE_LEDGER_20260902.md)。

## 10. 相关入口

- [深圳扩展实验总计划](../rlinf-shenzhen-experiment-expansion/00_INDEX_AND_PLAN.md)
- [Fast-WAM current interface contracts](02_INTERFACE_CONTRACTS.md)
- [AutoDL旧实现与历史实验过程](05_IMPLEMENTATION_PLAN.md)
- [深圳official standalone计划与结果](09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md)
- [深圳standalone复现摘要](11_SHENZHEN_FROM_ACT_TO_OFFICIAL_FASTWAM_NOTES.md)
