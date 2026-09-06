# current RLinf × π0 × global-z DVAC-GRPO 实施与 smoke 流水账

最后更新：2026-08-24  
机器：`SZ-H100`（服务器执行）/ `WIN-LOCAL`（文档与审计）  
状态：实现、focused checks、commit/push 已完成；真实2-step smoke 于 2026-08-24 17:14 CST 启动。

## 1. 本轮授权与冻结边界

用户明确授权：在深圳 current RLinf 上实现、做少量高信息量检查并执行真实 smoke。

冻结选择：

- base：深圳已经运行过的 current GRPO `554c6dc8...`，其 official base 为 `7d07a421...`；
- signal：复用 current π0 endpoint/telemetry `f7cf0f60...`；
- 方法：只实现旧 AutoDL 成功路径的 `global_zscore`、权重范围 `[0,2]`；
- 不实现：R-only、Position/S/I 分解、全量训练期 NPZ/视频、旧 control trace、旧 env/schema/worker 整文件覆盖；
- exact resume：保存和恢复 recent-5 统计状态；
- smoke：真实2个 outer steps，第一步建立历史，第二步验证非均匀权重进入反向传播；
- 参数：以深圳 current GRPO 为主体，只增加 DVAC 字段与 smoke 预算；不借 DVAC 之名改 group、batch、模型、loss 或环境合同。

## 2. 依据链

1. 深圳 current GRPO 决定 current typed trajectory、4-rank actor、RoboTwin/π0、batch 与运行入口。
2. AutoDL old global-z DVAC-GRPO 决定 $V_{L=3}$、recent-5、global mean/std、`[0,2]` 映射与 actor straight-through 梯度语义。
3. current π0 telemetry 决定 RTC、`actions/model_actions` 与 endpoint 提取接口；不回退旧 wrapper。
4. 深圳曾发生 GRPO 中途退出，因此 recent-5 状态进入 checkpoint 是真实恢复合同，不是新增方法。

## 3. 操作记录

### 2026-08-24：启动前本地上下文与范围锁定

- 完整重读 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和两个当前专题入口。
- 核对用户本轮选择：原始 global-z `[0,2]`；exact resume；简洁实现与检查；授权真实 smoke。
- 创建本账本。此时尚未创建服务器 worktree、修改代码、运行测试或启动训练。

### 2026-08-24：live preflight 与本地 source-locked 实施

- 普通账号只读刷新：RLT Stage 2 v4 独占 physical GPU 4--5；GPU 0--3、6--7 空闲；host available
  约 1.9 TiB，`/`、`/home`、`/data` 分别约余 234 GiB、2.2 TiB、2.8 TiB。
- 管理员只读刷新：无 failed systemd unit、OOM、存储或新 GPU error；其他用户有编辑/终端活动但无 GPU
  或大内存任务。Mihomo active/restart0；本轮代理 GitHub/HF 探针出现 SSL EOF，尚未把它解释为流量耗尽。
- 在 partial audit mirror 上从 `554c6dc8...` 建本地分支/worktree
  `codex/local-sz-current-dvac-grpo`；Windows bundled Git 首次 checkout 因 HTTPS helper/Schannel 失败，
  仅为该命令设置 bundled `GIT_EXEC_PATH` 与 `http.sslBackend=openssl` 后成功，没有改持久 Git 配置。
- 用 `git cherry-pick --no-commit f7cf0f60...` 叠加 current π0 telemetry；保留 RTC 与
  `actions/model_actions` 当前合同。
- 新增精简数学模块 `rlinf/algorithms/dvac_train_weighting.py`：只含 endpoint population variance、
  recent completed-step global stats、`w=1+0.5*clip(z,-2,2)`、straight-through 与 state load/save；
  不搬旧 CSV/provenance/control-trace 层。
- current rollout 仅在 train query 请求 endpoint telemetry，立即压成 selected `L=3` 的 `[B,H]`
  variance 并放入 `forward_inputs`；不把 `x_chain` 沿 trajectory 长期保存。
- current actor 在 current shuffle 前计算权重并将其放回同一个 nested `forward_inputs`，因此权重和
  action/query 共同重排；model forward 前移除辅助 tensor，ST 只改变 log-prob backward。
- 每个成功 outer step 每 rank 保存一个小型 `.pt`（`V/weight/z/advantage/reward/mask`）；TensorBoard
  增加 warmup、history/current mean/std、weight mean/ESS 与 clip fraction。
- checkpoint 每 rank 增加 `dvac_state_rankNNNN.json`，恢复时严格核对 mode、`L`、world size 与 recent-5
  配置/历史；旧 AutoDL 只写 state artifact、没有 load hook，本次补全 exact resume。
- 新增继承 current GRPO 的短 child YAML，仅附 global-z `[0,2]` 配置；新增6项 focused tests，覆盖
  endpoint、ST、recent-only/no-self-leak、resume config、current nested shuffle 与 sufficient stats。
- Windows AST compile 通过；bundled Python 不含 torch，因此真实 import/pytest/compose 留到服务器既有
  RLinf venv 执行。

后续每条服务器命令将记录：账号、时间、cwd、command-file、目标文件、结果、问题与唯一窄修复；凭据不入账。

### 2026-08-24：服务器 worktree、实现与 focused checks

- 第一次建服务器 worktree 前的 source probe 发现先前手工抄写的 telemetry full SHA 不可解析；该命令在
  任何写入前退出。重新从已验证引用取得精确 `f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5` 后，创建：
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current`、
  branch `codex/sz-current-pi0-dvac-grpo`。
- 从深圳 current GRPO `554c6dc8d586162d9444c01fa88308ed4f5203d0` 建分支并 cherry-pick current
  π0 telemetry，得到 `bfb99cce722015fe55bb3393bafb6f837e4cfa90`；随后应用精简 DVAC 增量。
- 生产增量仅5个文件、`+667/-2`：一份 default-off config stanza、一个 weighting 模块、current actor
  与 rollout 各一个窄 hook、一个 focused test 文件。没有改 GRPO loss、schema/Builder、RoboTwin env、
  π0 model 或 RTC/action decode。
- 服务器既有 venv 中执行10项 tests（6项新 weighting + 4项现有 telemetry），两轮均 `10 passed`；
  `ruff check`、`git diff --check` 均通过，仅有 PyTorch 既有 deprecation warning。
- 第一次 Hydra compose 使用短 child YAML 继承带 `hydra.searchpath` 的 primary config，被 Hydra 正确拒绝。
  唯一修复是删除重复 child 配置，把 default-off stanza 窄加到 current base YAML，smoke 只 override
  `mode=apply`。第二次 compose 又发现 YAML 1.1 将裸 `off` 解析为布尔 `false`；改成带引号的 `"off"`。
- 最终 baseline 与 DVAC 两份 resolved config 除 `mode: off -> apply` 外完全一致；marker 为
  `BASE_PARITY_AND_DVAC_CONFIG_OK`。这两次均是 compose 期失败，没有启动 actor/env。
- commit `66c863bc5a45e90cb5161b30af54355b1104c810` 已普通 push 到
  `personal/codex/sz-current-pi0-dvac-grpo`；现场 HEAD、remote 与 clean tree 三者一致。
- pretest 产物：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/pretest-current-dvac-global-z-20260824-v3`，约28 KiB。

### 2026-08-24 17:12--17:14 CST：resolved packet 与真实 smoke 启动

- 正确使用 RLT 的 `runtime/wrapper.pid` 与 `runtime/driver.log` 刷新：RLT v4 存活，完整 Step176/250，
  `global_step_175` 已完成；physical GPU4--5 使用约18--22 GiB。GPU0--3无 compute process；host
  `MemAvailable` 约1.86 TiB；shared Ray `172.17.0.1:6389` active、无 pending/failure。
- packet 生成命令：
  `local_scripts/remote_commands/shenzhen_prepare_current_dvac_grpo_smoke2_packet_20260824.sh`。
  远端完成 source/remote/clean、GPU、Ray、Hydra resolve、预算断言和生成 launcher 的 `bash -n`；成功 marker
  `SZ_PREPARE_CURRENT_DVAC_GRPO_SMOKE2_PACKET_OK`。
- packet：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-smoke2-4gpu128train64eval-v1`；
  `resolved.yaml` SHA-256 为
  `7e687fd076f5da5d6187c0ee4efa169526e17a78af739f504915a98b2b68ec2c`。
- resolved contract：physical0--3；2 outer steps；每步 `128 env x 4 rollout epochs = 512 trajectories`、
  G8/64 groups、最多2,048 chunk records；actor GB2048/MB32/update2；global-z `L3/recent5/warmup1/[0,2]`；
  Step2后 fixed64 与 `global_step_2`；train/eval video均关闭；hard timeout7,200秒。
- 用户已在本轮明确授权实现、简洁检查与真实 smoke；聊天中展示上述完整 packet 后执行：
  `bash /data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-smoke2-4gpu128train64eval-v1/launch.sh`。
- 17:14 CST 启动成功：run root
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-smoke2-4gpu128train64eval-v1`；
  wrapper PID `2120313`、observer PID `2120317`；4个 rank 精确放置到 physical0/1/2/3。
- 17:15 CST 首次只读刷新：wrapper存活、exit pending、无 fatal；模型初始化期约1.7 GiB/card，host
  `MemAvailable` 约1.78 TiB。RLT在GPU4--5同时满负载更新，未发生卡位混用。

### 2026-08-24 17:16--17:42 CST：Step 1 完整

- 四个 rollout epoch 分别约6分钟，Step1完整 wall `1553.412 s`；其中 rollout `1489.4 s`、actor training
  `25.513 s`、sync weights `38.350 s`。512 trajectories，train success `0.7285156`。
- DVAC合同精确满足：`warmup=1`、history count为0，weight mean/square mean/ESS均`1.000`，上下clip fraction
  均0；当前 `log(V_L3)` mean/std为`-4.425/0.681`，只在本步完成后进入recent history。
- 原GRPO优化指标有限：KL `0.088`、clip fraction `0.123`、grad norm `29.715`、ratio `1.004`；无
  traceback/OOM/worker crash/nonfinite。Step2随后自然进入4个rollout epoch。
- Step1期间 physical0--3峰值截至该时点约51--52 GiB/card；host available最低约1.58 TiB，远离Ray
  memory阈值。RLT physical4--5保持独立运行。

### 2026-08-24 17:42--18:37 CST：Step 2、评估、保存与收尾

- Step2四个 rollout epoch 用时约21分11秒；随后非均匀 global-z 权重完成 actor update，没有
  traceback/OOM/worker crash/nonfinite。
- Step2汇总：train success `0.71875`，weight mean/std=`1.002/0.483`，ESS fraction=`0.811`；
  P05/P50/P95=`0.249/0.966/1.931`，上下z裁剪比例=`0.98%/4.10%`。KL/clip/grad norm=
  `0.030/0.099/29.729`，均为有限值。
- fixed64=`49/64=0.765625`；`global_step_2` 保存了4个DCP shard、metadata、full weights及4份
  `dvac_state_rankNNNN.json`。wrapper于18:11:07 CST自然退出，`exit_code=0`，GPU0--3释放。
- driver存活期单卡峰值61.8--66.5 GiB，host available最低约1,307 GiB；退出后恢复约1.86 TiB。
  RLT在GPU4--5持续正常运行，18:36 CST已到完整Step220/250，无fatal。
- finalizer只读取8个rank-step tensor、TensorBoard和resource CSV，生成CSV/JSON/图与16,343-byte轻量
  tar；没有复制checkpoint正文。结果入口为
  `evidence/current-dvac-grpo-smoke2-20260824/`，结论见专题25号文档。

### 2026-08-24 18:59--20:15 CST：formal启动与迁到GPU4--7

- formal v2按已审阅合同在physical `2,3,6,7`启动；参数为`32 env×8 rollout epochs/G8/B512/MB32/update2`、
  global-z `L3/recent5/[0,2]`、fixed64/save10、100 outer steps。
- v2完整到Step4；用户在RLT自然完成、GPU4/5释放后明确要求停止并迁到physical `4,5,6,7`。按其授权仅TERM
  目标owned PGID，旧run原样保留，并写`runtime/stopped_by_user.txt`；未清理checkpoint、日志或其他Ray job。
- v2尚未到Step10，因此没有可resume checkpoint。v3保持所有算法/模型/全局预算不变，fresh启动，不拼接
  Step1--4指标。
- 第一次packet派生只做文本替换，但JSON中的`physical_gpus`跨行，验证发现仍为旧值后在启动前退出。唯一窄修为
  结构化更新JSON和带转义的command，再核对resolved/command/launch三者一致。
- 20:11 CST v3启动成功：packet/run分别为
  `.../packets/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3`与
  `.../runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3`；wrapper PID `2743592`。
- 20:15 CST一次简短健康确认：已进入Step1第3/8个rollout epoch，wrapper alive、exit pending、无fatal；
  GPU4/5约21.6/21.7 GiB，GPU6/7约43.8/42.7 GiB。按用户要求不持续盯守。

### 2026-08-24 21:11 CST：迁卡后只读刷新与残留纠正

- v3 wrapper仍alive、exit pending；完整Step4，Step5 rollout到`4/8`。Step4 train success=`83.59375%`、
  KL=`0.039`、clip fraction=`0.099`、grad norm=`43.827`；DVAC weight mean/ESS=`1.015/0.839`，fatal=0。
- save10前没有checkpoint，符合配置。host available约1.4 TiB，无OOM。
- 进程级核对发现两组chenyiteng Ray actors：旧组PID前缀`246...`已运行约2h13m，新组PID前缀`274...`
  已运行约1h02m。旧wrapper停止并没有让shared raylet托管的actor自动退出；旧组仍占GPU2/3/6/7，
  新组占GPU4/5/6/7，两组在GPU6/7重叠。
- 因此此前“旧run已释放”的判断更正为“旧owned PGID已停，但Ray actor残留未清”。本轮用户只要求查看，
  未执行actor kill、namespace cleanup或shared Ray重启。

### 2026-08-24：formal与深圳成功GRPO v2 resolved精确对比

- 只读加载旧`128×4/B2048`与当前`32×8/B512`两份resolved YAML并递归展开；leaf数为227/237，差异24项。
- 10项是新增DVAC合同；其余主要为run路径/命名、worktree seed路径、显式GPU表达、train/eval绝对输出与
  关闭视频。两个worktree的train/eval seed JSON均以`cmp`确认逐字相同。
- 模型、SFT、H=C50、Flow-SDE、GRPO/G8、reward/logprob、LR、PPO clip、MB32、update2、fixed64、
  save10与100 outer steps无差异。
- 真正额外变化为`128×4→32×8`、trajectory `512→256`、global batch `2048→512`；由此最大records
  `2048→1024`、optimizer calls/step `2→4`。因此当前不是旧深圳run的严格单变量DVAC对照。
- 本轮只做参数审计，不生成ZIP、不修改训练配置或进程。

### 2026-08-24 21:42--22:07 CST：残留清理、预算事故纠正与严格匹配重启

- 只读进程/显存/PSS核对把迁卡残留唯一锁定为old job `1b000000`、namespace `RLinf_1`；当前错误配置为
  job `1c000000`、namespace `RLinf`。旧组12个GPU进程合计159,014 MiB显存、398.393 GiB PSS；当前组
  当时为83,374 MiB显存、249.462 GiB PSS。
- 按旧namespace杀21个named actors，`no_restart=True`；当前namespace的21个actors前后集合完全相同，
  shared Ray未重启。GPU2/3归零，GPU6/7从约63 GiB降到约22--23 GiB，host available从约1.3升至1.7 TiB。
- 管理员账号本次密码认证失败，依固定认证规则未更换管理员路线；用chenyiteng只读`ps/nvidia-smi`已足够
  确认liwenbo在GPU0/1短暂运行两条`starwam-r002` dual-stream screen任务，未触碰。两条任务随后自然退出。
- 用户指出`32×8/B512`不是目标深圳GRPO对照。错误v3在完整Step7后停止；精确清理job `1c/RLinf`的
  21个named actors后，8卡均回到约6--10 MiB、host available约1.9 TiB。
- 根因是formal启动沿用了config-only commit `554c6dc8...`的默认`32×8/B512`，而不是目标GRPO v2
  resolved的`128×4/B2048`；主存风险不能替代对照设计授权。
- v4 compose后与目标resolved逐叶比较：227/237 leaves，仅19项允许差异，unexpected=0；10项为DVAC，
  其余为等价卡号、逐字相同seed的路径、run隔离路径和命名。video flags也与baseline同为true。
- v4 fresh启动到GPU4--7，run为
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4`；
  resolved预算为`128×4/512 trajectories/64 groups/2048 records/B2048/MB32/update2`。22:12 CST已完成首个
  rollout epoch `1/4`（约5分55秒），GPU4--7约27--29 GiB/card、host available约1.8 TiB，
  wrapper/observer存活、fatal=0；按用户要求停止持续观察。

### 2026-08-25 09:39--09:56 CST：Step29与整机只读刷新

- 先按固定host-key、普通账号Paramiko路线运行v4状态脚本；一次调用因把`SHA256:`前缀重复传入而在
  认证前被helper拒绝，改回helper要求的裸digest后成功。没有更改服务器认证、进程或文件。
- 管理员`toom`沿同一固定host-key/password路线认证失败；依既有规则停止管理员尝试，不切换key、sudo或
  其他认证路线。后续公共资源、GPU PID归属与用户进程均由`chenyiteng`只读完成；因此本轮没有内核journal
  级OOM/Xid结论。
- 09:54 CST最新训练现场：完整Step29/100，Step30的4/4 rollout已经完成并进入后续update/eval/save；
  wrapper `3268047`存活、exit pending、fatal=0。Step29 train success=`0.92578125`，KL/clip/grad=
  `0.013/0.062/14.235`。
- TensorBoard通过服务器既有venv只读提取，Step29 DVAC weight mean/std/ESS=
  `1.0288/0.4436/0.8432`，upper/lower z clip=`3.860%/0.292%`；Step10/20 fixed64分别为
  `57/64`与`59/64`。`global_step_10/20`完整目录存在。
- 资源现场：GPU4--7全部属于该run，当前约62--68 GiB/card，observer历史单卡峰值`74.90 GiB`；GPU0--3
  无compute app，其他用户未占GPU。四个EnvWorker RSS合计约`1.47 TiB`；host available从启动约
  `1.984 TiB`降至09:41约`450 GiB`，09:54瞬时约`392--405 GiB`。最近3小时平滑变化约`-103 GiB`；
  memory/io PSI为0，vmstat无持续swap-in/out，但6 GiB swap已基本占满。当前没有OOM/抖动，继续运行的
  主要风险仍是与旧matched GRPO v2相同的常驻EnvWorker主存增长。
- 服务器公共健康：failed systemd units=0，Mihomo与Docker active；`/`、`/home`、`/data`分别余
  `233 GiB/2.1 TiB/2.7 TiB`，inode使用`2%/1%/1%`。其他用户只见开发会话与轻量下载：liwenbo
  VSCode/Codex/watchdog，zhangwei Codex及HF下载，guorenjie新Cursor会话；均无GPU与大内存任务，
  zhuanghuiping/qiufuwen无活跃进程。本轮未查看私人文件内容、未干预任何进程。
- 远端run约35 GiB；仅下载driver/resource/resolved/TensorBoard四个小文件，合计约0.38 MiB到
  `evidence/dvac-grpo-v4-live-step29-20260825/`。本地生成三张PNG、逐步CSV与summary JSON；没有下载
  checkpoint、视频或逐步tensor，没有生成ZIP。
- 10:02--10:05 CST收尾刷新：Step30已完整，train=`0.90234375`，fixed64=`63/64`；
  `global_step_30`为18 GiB、10个文件并含4份DVAC sidecar，随后Step31已进入rollout。瞬时available RAM
  由约381 GiB降到293 GiB；以完成Step29口径对比，v4约450 GiB、旧matched v2约464 GiB，内存轨迹高度
  接近。因此数值/保存正常，但重现旧v2在约Step50--53接近Ray阈值的风险很高；本轮只读，不自动停止或改参。

### 2026-08-25 10:26--11:10 CST：Step31重绘、跨机器审计与用户公开进程刷新

- 通过固定host-key、`chenyiteng` Paramiko只读刷新：10:26完整Step31、Step32 rollout；11:10完整Step33并进入Step34 rollout `1/4`。两次均wrapper alive、exit pending、fatal=0，未控制进程。
- 先核远端小文件总量：driver/resource/resolved/TensorBoard合计约0.42 MiB，本地C盘余37.67 GiB；只下载这些小文件到`evidence/dvac-grpo-v4-live-step31-20260825/`，未复制checkpoint、视频或逐step tensor。
- 新图修正旧快照时序与绘图截断：matched baseline完整保留到Step52，Step32--52明确为baseline-only；DVAC紫色、GRPO蓝色，并为fixed64使用真实方块/三角图例和逐点计数。Step10/20/30累计两边均`179/192`。
- Step1--31 train mean与最新5步差为`-0.50/-1.25 pp`；Step31 weight std/ESS=`0.4382/0.8425`，upper/lower clip非零。当前尚无收益证据，但实现并非no-op。
- 只读核旧AutoDL代码、resolved、逐步CSV和Git历史；再以`git ls-remote --heads`现场确认公共GitHub branch HEAD：old `afdaa2e2...`、current `66c863bc...`均与本地authority一致。公式、L3、global统计域、recent5、warm-up、`[0,2]`、ST挂点与动态rank reduce等价；current新增typed-shuffle适配和严格resume sidecar，未发现实现错误。
- AutoDL g1--49的训练rollout累计/末5/末10优势为`+2.081/+2.734/+2.930 pp`，但旧run没有fixed held-out eval、同seed同代码mode-off对照或重复seed。跨机器预算为256/B512/2 ranks与512/B2048/4 ranks，不能把同一step机械等同。
- 公共进程只读观察：liwenbo的VSCode server、extensionHost、SSH notty、Codex和watchdog仍活跃，说明Remote会话通道存在，但不能证明桌面窗口正处于前台；其Wan模型只出现在tmux server原始命令行，没有活动`hf`子进程。zhangwei在10:26活动下载为`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`，11:10已无`hf`进程，无法仅凭公开进程判断成功或失败；两人均未用GPU。未读取其私人文件或日志。
- 11:10 GPU4--7约59--61 GiB/card，GPU0--3空闲；host available约318 GiB，swap 6 GiB已满，memory PSI avg300约0.16但无fatal。`/`、`/home`、`/data`分别余233 GiB、2.1 TiB、2.7 TiB。

### 2026-08-25 12:23--12:46 CST：误杀后从Step30严格续跑

- 用户明确说明旧GRPO-DVAC进程被其他用户误杀，并授权保持physical GPU `4,5,6,7`、全部参数不变，继续到总Step100。
- 固定host-key、`chenyiteng` Paramiko只读刷新确认：旧v4 wrapper已退出、`exit_code=255`；driver明确写出actor被`ray.kill`，随后Gloo peer断开。最后完整训练记录为Step33，Step34只完成`2/4` rollout；最后完整checkpoint仍为`global_step_30`。
- Step30 checkpoint共10个文件：4个DCP shard、`.metadata`、full weights与4份`actor/dvac_state_rankNNNN.json`；sidecar合同为rank0--3、world size4、`mode=apply`、`selected_l=3`、`runner_step=30`。current runner源码确认会把resume起点设为30，并执行Step31--100；DVAC load hook在载入DCP前严格校验并恢复recent-5状态。
- 现场8卡均空闲，shared Ray为0 CPU/0 GPU usage且无alive actor；旧Ray worker pool中的`ray::IDLE`进程不属于活跃实验，无需清理。没有重启shared Ray、没有kill其他用户进程。
- 新续跑run为`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30`。resolved与v4逐叶比较只差7个允许项：`runner.resume_dir`及6个run-scoped输出路径；所有模型、采样、batch、更新、评估、保存及DVAC字段完全相同。
- 两次preflight在创建训练进程前被本地parity脚本拦截：第一次漏列随run变化的`algorithm.dvac_gradient_weighting.output_dir`，第二次误把flatten后的dict当嵌套dict。两份仅含7,912-byte resolved YAML的partial packet均移到带`failed-preflight-*`后缀保留；未创建run、actor或GPU占用。修复仅涉及启动器校验代码，不涉及算法或resolved参数。
- 12:42 CST正式启动，wrapper PID=`4182258`。12:46现场明确出现`Resuming training from .../global_step_30`并进入Step31 rollout `0/4`；4组actor/rollout/env均落在physical 4--7，四卡约20.8--21.6 GiB、host available约1.9 TiB、fatal=0、exit pending。按用户要求不持续盯守。

### 2026-08-25 13:07 CST：续跑参数复核与管理员身份探针

- 现场重新加载v4/v5两份runtime resolved并递归展开：均为237个leaf，仅7项差异、unexpected=0。差异精确为`runner.resume_dir`、logger路径、train/eval视频与RoboTwin数据路径、DVAC telemetry输出路径；不存在模型、采样、batch、update、DVAC数学、评估或保存节奏差异。
- 关键值现场仍为physical `4,5,6,7`、target100、`128 env × 4 rollout epochs`、G8、GB2048/MB32/update2、fixed64、eval/save10、global-z apply、L3/recent5/warm-up1/`z_clip=2`/`strength=.5`，train/eval video均开启。
- wrapper仍存活、exit pending、fatal=0；已从Step30恢复，Step31 rollout进行到`3/4`。GPU4--7约51.9--53.3 GiB/card，host available约1.7 TiB。
- 按用户新提供的凭据，仅以固定host-key Paramiko对`toom`做`hostname/id/whoami/pwd/groups`身份探针；认证成功。现场为hostname `admin`、UID1000，组含`sudo/adm/lxd/labdata`。未执行sudo或任何管理员写操作；凭据未写入文件或文档。

### 2026-08-25 16:05--16:56 CST：v5 Step40简要可视化

- 固定host-key Paramiko只读刷新v5；16:54完整Step40并进入Step41，wrapper PID `4182258`存活、exit pending、fatal匹配0；`global_step_40`已存在。
- 先从远端既有venv提取小型TensorBoard scalar JSON，再仅SFTP下载driver/resource/resolved/scalar JSON；未下载checkpoint、视频或Ray大日志。
- 以真实恢复分支`v4 Step1--30 + v5 Step31--40`拼接；被误杀前但未被checkpoint承接的v4 Step31--33明确排除。与matched原GRPO Step1--52同轴，Step41--52仅作baseline-only背景。
- Step40 train/fixed64=`99.02%/63-of-64`；fixed四点累计DVAC与baseline均`242/256`。配对Step1--40均值/最近5步差=`-0.42/-0.98 pp`。
- Step40 KL/clip/grad=`0.01397/0.03385/6.406`；weight mean/std/ESS=`1.0049/0.4258/0.8478`；全部有限。v5资源CSV最新available RAM约`971 GiB`，单卡显存最新最大约`65.1 GiB`。
- 生成3张PNG、自包含HTML、拼接CSV、summary与README到`evidence/dvac-grpo-v5-live-step40-20260825/`；可重建脚本为`local_scripts/render_shenzhen_dvac_grpo_v5_resume_live_20260825.py`。没有生成ZIP。

### 2026-08-25 16:36--16:54 CST：创建普通账号yanchuhan

- 用户明确授权创建`yanchuhan`、口令由用户指定、无sudo。先确认账号不存在、`labdata`和`/data/shared`存在。
- 通过已验证的`chenyiteng` sudo创建账号与私有home/data，加入`labdata`，建立`~/data`与`~/shared`；后检`id`仅显示自身组和`labdata`，`sudo -l -U`明确拒绝。
- 使用新账号与用户指定口令完成固定host-key SSH实登，`pwd=/home/yanchuhan`，两条链接解析正确；口令未写入脚本或文档。

### 2026-08-25 17:23--18:01 CST：`[0,2]` 收尾与 `[0,5]` 正式启动

1. 固定host-key Paramiko只读锁定v5 wrapper PGID、Ray job `20000000`、namespace `RLinf`的21个actors及12个GPU进程；现场完整Step41、Step42 rollout `2/4`，fatal=0。
2. 按用户授权TERM owned PGID并精确`ray.kill(no_restart=True)`上述named actors；target actors/job进程均归零，shared Ray未重启，写入run-scoped stop marker。完整Step41作为终态。
3. 从server现有TensorBoard event提取小型scalar JSON，只下载driver/resource/resolved/manifest/parity/contract/stop marker/scalars；本地重绘3张PNG、HTML、拼接CSV和summary，不下载checkpoint或视频。
4. 审计current映射确认不能用单一strength得到非负连续`[0,5]`。在独立branch中加入显式`weight_min/max`分段映射，保持null时旧`[0,2]`公式和sidecar schema兼容；actor只透传两个字段，YAML默认均为null。
5. server运行`tests/unit_tests/test_dvac_train_weighting.py`，`7 passed in 5.34s`；commit=`0e28ac6f...`并普通push到`personal/codex/sz-current-pi0-dvac-grpo-w0to5`。
6. compose fresh formal packet并同时对照原GRPO resolved和`[0,2]` resolved：除DVAC端点、批准的eval `10->5`与run路径/命名外，unexpected differences=0。save仍为10，全部训练预算不变。
7. 启动run `dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1`；wrapper PID=`1509464`。18:01 CST已进入Step1 rollout，21 actors、fatal=0，GPU4--7均已建立模型上下文，host available约1.9 TiB；停止持续观察。

### 2026-08-25 18:23--18:27 CST：图表换色与整机只读刷新

- 按用户反馈，仅把`[0,2]`主成功率图的DVAC/GRPO/baseline-only改为橙`#E4572E`、深青`#00796B`、浅青`#80CBC4`；图例、marker和resume虚线同步，数据与布局未变，PNG已解码和目视核验。
- 原轻量ZIP用PowerShell更新时首次误用`-LiteralPath`配合通配符而失败；立即改用`-Path`重建同名ZIP，最终15成员、405,737 bytes，未丢失原始证据。
- 管理员固定host-key Paramiko只读刷新：`[0,5]`完整Step1并进入Step2，fatal=0；GPU4--7只属于该run，GPU0--3无compute。RAM available约1.6 TiB、PSI=0；`/`、`/home`、`/data`分别余230 GiB、2.0 TiB、2.6 TiB。
- Mihomo active、代理GitHub/HF均HTTP200；500 GiB配额已用128.087 GiB、剩371.913 GiB、2026-11-23到期。其他用户均未用GPU，只见公开命令层面的VSCode/Codex、下载或环境安装；未读私人文件/日志、未干预任何进程。

### 2026-08-25 22:46--23:21 CST：`[0,5]` Step12与整机只读刷新

- 先后以固定host-key Paramiko按权限分工只读刷新：`chenyiteng`读取私有run，`toom`读取整机公开资源；没有修改、停止或启动任何进程。管理员直接访问私有run为空并非训练停止，owner路线确认wrapper PID `1509464`一直存活。
- 23:18完整Step12并进入Step13 rollout `1/4`；exit pending、fatal匹配0。Step12 train/KL/clip/grad=`0.7871094/.023/.065/44.108`，DVAC weight mean/ESS=`1.685/.619`，均为有限值。Step5/10 fixed64=`54/64,57/64`；`global_step_10`目录存在，run当前约18 GiB。
- 与strict-matched original GRPO配对到Step12：训练成功率均值差`+0.13 pp`、最近5步差`-2.42 pp`；Step10 fixed64双方均`57/64`。只有12步和两个评估点，不作方法收益结论。
- 一分钟资源CSV共323点、约5.38小时：GPU4--7单卡峰值`74.21 GiB`、最新约65--66 GiB；host available最新约`729 GiB`、最低约`728.7 GiB`，近1小时继续下降约`133 GiB`。memory PSI仍为0、swap仍有约4 GiB free，故当前未故障，但主存未平台化，是最重要风险。
- 23:21整机8卡均在使用：chenyiteng的GRPO-DVAC占GPU4--7；guorenjie的四条LeRobot π0/LIBERO训练占GPU0--3，约44--48 GiB/card。任务分卡，无GPU冲突；guorenjie进程合计约52 GiB RSS，不解释GRPO四个EnvWorker约1.2 TiB RSS。未干预其任务。
- `/`、`/home`、`/data`分别余约`230 GiB/2.0 TiB/2.6 TiB`，inode使用`2%/1%/1%`；failed units为空。Mihomo active，代理GitHub/HF均HTTP200，direct GitHub 200、direct HF超时。liwenbo仍是VSCode/Codex/watchdog与Wan下载tmux，zhangwei仍是VSCode/Codex，无GPU；未读取私人日志。
- 只下载driver/resource/TensorBoard三个小文件并生成两张橙/深青高对比PNG与summary JSON，未生成ZIP、未下载checkpoint或视频。入口：`evidence/dvac-grpo-w0to5-live-step12-20260825/`；可重建脚本：`local_scripts/render_shenzhen_dvac_grpo_w0to5_live_20260825.py`。
