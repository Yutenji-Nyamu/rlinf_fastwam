# click_bell 最小切换 · 2026-10-05

本目录只准备代码与配置。尚未在服务器执行测试、加载奖励权重、启动或切换训练；动态回执由主执行线程记录。Rynn小批成败核验先完成，随后本任务优先，RLT保持精确断点候补。

## 改动与继承

直接复用已跑通的 OpenDW B16、Sidney π0.5、三相机、14D绝对命令、C32、GRPO、资源监控、卸载等待、原生评估和原借还清理逻辑。更换click_bell reset/任务文字/专用RM/原生seed；从原SFT开始，不从摆瓶子CP70开始。

本次另一个明确改动是奖励：摆瓶子旧版使用连续分数差分；这里按用户要求使用0/1。8张未来主图中任一RM分数≥0.9，在整个C32块边界给1并终止，后续该轨迹不再调用WM或重复奖励。原生评估仍使用RoboTwin真实成功判据。0.9沿用旧成功阈值，尚未声称已校准。

| 项目 | 短smoke | 正式 |
|---|---|---|
| 物理GPU | actor/rollout 4、5；WM+小RM 6、7 | 相同 |
| N/G/WM batch | 64/8/16 | 相同 |
| R / 每轨迹动作上限 | 1 / 32 | 8 / 384 |
| 总轨迹 / runner轮 | 64 | 512；G8已包含在N64中 |
| global/micro/U | 64/8/2 | 2048/8/2 |
| runner预算 | 1轮，从原SFT | 200轮，从原SFT；保存/原生评估每10轮 |
| 原生评估 | N32/R1/C32/max32，检查接线与绑卡 | N32/R1/C32/max384 |

短smoke只验证原有正式并行、模型加载、一步生成/评分/训练/评估通路，不把零梯度或32动作低成功率当任务能力结论。正式每轮最多6144动作块槽位；GRPO保留原[0.1,0.9]组平均回报过滤，对0/1回报恰好保留G8中1–7条成功的组。

## 文件与服务器调用

- `build_reset_data.py --task-name click_bell --source <clean50> --output <click-bell-reset.npz>`：完整50条不筛选，逐episode匹配原任务文字，同刻主/左右腕/state。新增参数只改变任务回执。
- `build_assets_rpc.py --output <本机rpc.py> [--native-seed-source <服务器既有click_bell种子JSON>]`：生成可经固定host-key SSH标准输入运行的SZ1脚本；服务器解释器用`/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B -`。只在`/data/chenyiteng/projects/wmrl-click-bell-assets-20261005`写CPU提取代码、50起点NPZ、原指令/种子记录和hash回执，不复制整HDF5、不更改源数据。没有可靠seed记录时明确写缺失，绝不把episode编号当seed；原生eval seed为独立可选输入。主线程按`complete.json`文件清单relay到SZ3即可。
- `patch_click_bell.py --adapter <旧adapter> --env <pre-Rynn旧env> --base-owner <旧multigpu owner> --formal-owner <旧formal owner> --output <新私有source目录>`：精确替换生成四份新文件，校验并记录donor和结果hash；不改旧文件。必须使用生成后的两个owner，因为旧owner硬编码摆瓶子及相对奖励。**不要使用`batch16_formal_owner.py`作为新任务入口，它强制恢复旧摆瓶子checkpoint。**
- `build_config.py --base-config <B16实际formal.json/yaml> --base-config-sha256 <hash> --name <新唯一名> --owner-dir <新owner> --initial-state <新reset.npz> --native-seeds <click-bell-seeds.json> --sft-path <实际原SFT路径> --service-urls <6卡端口> <7卡端口> --output <新配置目录>`：配置为JSON兼容YAML；生成formal/startup_smoke和完整差异。新源代码挂到独立RLinf checkout。
- `probe_reward.py`：接收带真实标签的≤128张主图NPZ，strict加载专用RM，默认B16，报告0.5/0.9混淆计数、逐样本分数、耗时/显存；没有真实标签的中间帧用-1，不能冒充失败。
- `test_click_bell.py`：服务器CPU运行，需把旧`opendw_smoke_20261003`目录与本目录按相邻位置放置；只验证稀疏奖励映射及四处精确改动。

服务本身沿现有`opendw_service_batched.py`，明确`--reward-checkpoint <click_bell/resnet_rm.pth>`、现有`--t5-path`、`--execution-mode batched --wm-batch-size 16`。每个service plan需要新增`reward_checkpoint`和`reward_checkpoint_sha256`，新owner逐项核对。代码没有LPIPS分支，因此不会默默加载官方YAML的LPIPS选项。

新formal owner的`protocol_reference`必须指向本次click_bell配置（同训练协议，允许runner日程不同），不能仍指旧摆瓶子。借还plan必须使用新现场GPU4修复链＋GPU5–7当前身份/断点；不要重放历史owner。命名graphics profile兼容使用本轮已有修复，不能回到旧严格scope。

工程smoke用原formal owner的已授权direct-start路线，`start_mode=direct_start_user_override_20261004`，因此不调用历史长测的非零梯度门槛。C32全0允许通过工程接线测试；仍需正常退出、有限输出及正确卡位。真实RM质量另看本次native32基线里逐条真实成败与同轨迹RM判断的对应，不能用native成功率单个数值代替RM准确率，也不能把短smoke通过称为学习有效。

## 必需资产与剩余证据

1. 专用RM：`WorldArena/WorldArena2.0/reward_model/click_bell/resnet_rm.pth`，公开记录588,338,878 bytes；实际加载/hash仍待主线程。T5继续用原已验资产。
2. OpenDW、原SFT、统计量、原生RoboTwin assets沿当前运行资产。
3. click_bell完整clean50历史路径在SZ1：`/data/chenyiteng/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/click_bell/clean50-20261002/aloha-agilex_clean_50`。当前存在/是否SZ3已有由现场核验；无需新专家筛seed。
4. click_bell原生seed从既有任务列表提取，至少32条唯一整数；不能沿用摆瓶子seed改键名，也不重新根据策略表现挑选。
5. 小RM样本需分初始未成功、真实成功及真实失败；只有专家末帧时，准确标签含义是success_once锁存，抬手图可能给低分，应与首次接触帧区分。OpenDW图像没有物理接触真值，不靠RM自己给自己做真值。

源码与公开资产依据见[任务RM审计](../../docs/world-model/robotwin_pipeline_20261003/click_bell_reward_audit_20261005.md)。目前这是最小代码准备，不宣称奖励准确率、smoke通过或正式已启动。
