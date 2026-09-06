# AutoDL 运行、Git worktree 与视觉时间对齐计划

最后更新：2026-08-20  
状态：**branch/worktree、默认关闭实现和服务器前测已完成；source HEAD为`61996e15…`，严格停在真实smoke批准边界。**  
当前对象：只含 `adjust_bottle` 原始 π0 SFT；深圳 H100、RLT、PPO与训练均不在本文范围。

## 1. 哪些是方法语义，哪些可按工程现场适配

首轮不能变化的高层合同：

- task-matched原始π0 SFT、`adjust_bottle`、clean、三路policy输入图；
- π0原推理得到的`H=50/M=4`，环境仍执行官方`C=50/D_active=14`；
- 每个query保存四个clean endpoint previews及denoise times；
- telemetry默认关闭，打开也不改变模型RNG、action或环境轨迹；
- 每条trace能回连reset ID、query、action-slot时间、episode结果和视觉状态；
- variance、`L=2/3/4`、threshold和图全部在推理后计算。

可根据source/runtime窄适配的工程细节：

- Hydra字段层级、writer放在rollout worker还是独立collector；
- NPZ或等价dense二进制、每rank文件名、flush批次；
- query图像用PNG/JPEG、压缩array或稳定索引；
- 两卡上每次并发多少env、需要多少互斥seed shards；
- `x_chain`是否随endpoint一并导出。endpoint和时间坐标必需；`x_chain`是推荐审计量。

## 2. Git：不是改原目录，而是从原commit开一个新工作区

### 2.1 当前已知关系

AutoDL上不是四五份互不相干的clone，而是同一Git对象库下的多个worktree：

```text
/root/autodl-tmp/RLinf                     共同π0/RoboTwin基线 6d0db56b
├─ RLinf_fastwam_rlinf                     DSRL 48a775db
│  └─ RLinf_rlt_pi0_robotwin               RLT 2b8199d8
├─ RLinf_qam_pi0_robotwin                  QAM ff8e28ef
└─ RLinf_ogpo_pi0_robotwin                 OGPO 5d5c84e3
```

worktree共享commit objects/refs，但每个目录有独立checkout和index。因此在新Idea2 worktree改文件，不会
改动原RLinf、RLT或其他算法目录；commit会进入共享Git对象库，这是正常Git行为。

2026-08-20 12:16只读快照显示：共同基线tracked clean但保留5个用户既有untracked文件；四个算法
worktree当时均clean。动态事实见
[AutoDL现场账](/C:/Users/86136/Documents/rl/docs/autodl-live-audit/evidence/OPERATION_LEDGER_20260820.md:60)，
真正开发前仍需刷新一次。

### 2.2 本次source选择

既然RLT已退出当前scope，Idea2不应从RLT `2b8199d8`继承RLT/DSRL代码；最清楚的父提交是之前真正跑过
π0/RoboTwin的共同基线：

```text
parent commit: 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
branch:        codex/idea2-dvac-pi0-robotwin
worktree:      /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
```

概念上这就是“从原RLinf继续往前改”，但物理上不直接在`/root/autodl-tmp/RLinf`目录改，也不在里面
`git switch`。实际建立命令是：

```bash
git -C /root/autodl-tmp/RLinf worktree add \
  -b codex/idea2-dvac-pi0-robotwin \
  /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
```

该路线已经执行并完成：当前worktree为
`/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin@61996e15cc7f5a32bd6012b61b20893d94636c82`，
branch和personal remote均为`codex/idea2-dvac-pi0-robotwin`且tree clean。执行前后的parent、其他worktree、
checkpoint/norm stats和共享venv检查见实施流水账。Windows下`.dsrl-impl-worktree/.rlt-impl-worktree`仍只作
历史patch/source证据，不是服务器权威branch起点。

## 3. 实现复杂度：主体简单，关键是数据出口和join

source-locked基线已经存在：

```text
x0_pred = x_t - v_t * t_input
chains   = [B,M+1,H,D_model]
```

所以不增加forward。主体只有三处：

1. **model sampler**：telemetry开时把每步现成endpoint append到result；旧采样语句不动。
2. **rollout worker/collector**：当前standalone eval丢弃附加result；改为按rank暂存CPU数据并flush
   shard，环境仍只收到actions。
3. **最窄metadata join**：让reset ID、env slot、query index、action-slot起点与query图像/MP4位置进入
   同一索引。若rollout侧拿不到reset ID，从env侧传一个小key，不传整套simulator对象。

这不是大算法改造。真正容易错的不是公式，而是把某条endpoint错接到另一个env/reset/video tile；因此
join是首版正式功能，不是额外防御性工程。

## 4. 两张A800怎样跑第一批数据

### 4.1 历史工程锚点

旧AutoDL纯SFT eval曾使用：

```text
2×A800 placement 0-1
16 eval env
10 rollout epochs
enable_offload=true
约20 GB GPU peak/card
约15 min
```

证据见历史材料
[Openpi + PPO AutoDL A800.md](</C:/Users/86136/Documents/rl/exports/dsrl_pi0_robotwin_formal_v1_work_materials_20260729/historical_source_materials/Openpi + PPO AutoDL A800.md:900>)。
这只能证明“两卡16并发曾跑通”；该旧run缺RLinf/RoboTwin SHA、权重revision、seed manifest、resolved hash
和RAM峰值，不能当当前性能基线或正式复现。

当前standalone eval没有ActorGroup，正确两卡placement候选是：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env, rollout: 0-1
```

两个rollout worker各在一张卡加载一份完整π0，不把一个π0跨两卡切分。待批smoke每个worker处理1个
env；后续16-env collection候选才是每个worker 8个env。

### 4.2 首轮80/20规模

smoke通过后推荐的第一批collection不是128条，而是：

```text
official-128目标集合中的第一个16-ID shard × 1 epoch
每卡8 env
每episode最多4 queries
总计最多64 query-state
```

理由：它已经覆盖多条成功/失败trajectory和最多64条`V(h)`曲线，足够第一次检查信号、视觉阶段和数据
join；并发规模有历史工程证据。它必须命名为`telemetry collection shard`，只报告这16条的success事实，
不写成official-128 success rate。

raw `x+z` 约1.6 MB；真正的额外体积来自query三路图像和视频，但16条episode仍是小规模。正式packet
再根据图像编码方式给出预计大小。

不做2/4/8/16并发sweep。真实链路先做一项高信息量smoke：

1. 2 env × 完整200 action slots：预期2 episodes/8 queries，闭合真实CUDA sampler、双rank writer、
   episode/finalize和图像/视频join。

16 env、1完整episode是随后仍需单独批准的首轮telemetry collection，不另跑一个重复的并发sweep；
在该次采集中同步刷新GPU peak/util、host/cgroup RAM、CPU、query latency、video encode时间与总wall。
如果2-env pre-test已经暴露明显资源异常，先回到packet调整；否则不预设32/64 env。

### 4.3 如果以后需要完整official-128

不能简单用`16 env × rollout_epoch=8 × fixed=true`：当前fixed逻辑已有reset IDs后不会更新，会重复同一
批16条。

若真实数据表明需要完整覆盖，后续才：

1. 按official 8-rank seed partition算法离线解析原本会使用的128个唯一IDs；
2. 生成8个互斥的16-ID seed shards；
3. 两卡串行运行8次`16 env × 1 epoch`；
4. 按reset ID合并telemetry和episode结果。

这保持同一个128-ID集合，但seed到rank/batch以及flow-noise流与八卡run不保证bitwise一致；应称
“相同环境seed集合的两卡sharded采集”，不是八卡进程级复刻。

## 5. 现有视频实际保存了什么

official YAML的`save_video=true`启用的是RLinf `RecordVideo`：

```text
<log_path>/video/eval/seed_<env-worker-seed>/<video_cnt>.mp4
```

这里目录中的`seed_*`是env-worker seed，不是RoboTwin reset ID。每个worker把本地多个env tile到同一
复合帧；视频只使用head/main camera，不含两只wrist camera。

C50/200-slot时，通常是：

```text
frame 0  reset初始状态 / query 0输入
frame 1  执行slots 0..49后 / query 1输入
frame 2  执行slots 50..99后 / query 2输入
frame 3  执行slots 100..149后 / query 3输入
frame 4  执行slots 150..199后的终态
frame 5  auto-reset spill frame（可能存在）
```

30 FPS只是MP4播放参数，所以历史上看起来“视频很快”；它不是robot control/physics频率。正式前用一个
小probe核对实际frame count、tile顺序和spill frame。

RoboTwin native `eval_video_log`和`render_freq`在RLinf VectorEnv初始化时被强制关掉，因此默认不会再
额外保存一套RoboTwin视频/PNG。现有路径通常只有RLinf这一套query-level MP4。

## 6. 首轮怎样把DVAC与任务阶段联系起来

每个query索引至少能恢复：

```text
eval_epoch, env_worker_rank, stage_id, env_slot, reset_id
query_idx, action_slot_start, query_uid
video_relpath, video_tile_index, video_frame_pre, video_frame_post
success_before, success_after, episode_done_after
```

并保存/引用该query真正送入π0的：

```text
head image, left-wrist image, right-wrist image, 14D robot state
```

于是首轮可以做：

```text
query q 的三路输入状态  <->  V_q(h), h=0..49
执行该C50 chunk前/后    <->  MP4 pre/post frame
episode结果             <->  reset ID与query序列
```

这已经能观察：free-space或接触附近的**当前query state**是否出现更高/更早的variance，以及一次chunk
执行前后任务发生了什么。

## 7. 为什么暂时不能“每一/两个action抽一张图”后直接对h

π0的`h`是50个未来qpos waypoints。RoboTwin不会把每个waypoint当成一帧：它先压缩重复点，再用TOPP
把整条chunk插值成可变数量的controller/physics steps，最后只返回chunk终态图。RLinf计数的
`action_slot_start=q×50`不是SAPIEN physics-step。

因此对现有MP4每一/两帧抽图，或把连续simulator帧平均切成50份，都会制造看似精确但语义错误的
`h↔frame`映射。

若query-level真实数据表明信号值得深挖，第二阶段再专门实现：

- 暴露waypoint到TOPP/control-step的映射；
- 在不改变原control loop的前提下采样head frame；
- 记录`query_uid,h,control_step,physics_step,frame_idx`；
- 实测渲染/拷贝开销后决定每h还是每2h。

这会触碰RoboTwin simulator执行层，方法和性能风险都高于首版旁路telemetry，所以不提前混进最小实现。

## 8. 开发前停点

用户确认本轮文档后才进入：live refresh → 新Idea2 worktree → 一个连贯代码批次 → 两项服务器pre-test。
完整16-ID采集仍需另给resolved config、精确命令、输出目录、reset-ID manifest、预计资源/数据量、监控
和停止条件，并等待明确批准。
