# 深圳 current RLinf：RLT / DSRL formal 与并发决策

日期：2026-08-23  
状态：历史启动 packet 已执行；当前正式实验与并发问题终态由
[08号并发解决文档](08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md) 接管。

## 1. 结论

- RLT 先启动：Stage 1 `2×H100 / MB16 / GB32 / 2k`，成功后由同一 wrapper 自动进入 Stage 2
  `2×H100 / 8 env / 250 cycles / GB512 / MB128`。
- DSRL 保持 AutoDL 成功主体：`2×H100 / 4 env / H50/N20 / GB256/MB64 / UTD20 / ring25k`；
  推荐 `200 cycles` 复刻旧成功 endpoint 规模，而不是把旧 launcher 的 `650` ceiling 冒充已完成规模。
- 两个训练可以共享一个显式 `ray start --head` 建立的持久 Ray cluster；current RLinf 用不同 namespace
  隔离 manager/named actor，GPU 仍由 placement 硬锁 `RLT=4,5`、`DSRL=6,7`。Ray namespace 不提供
  RAM/GPU quota，因此第二个训练只在第一个已稳定后启动。
- 不能直接让第一份训练隐式 `ray.init()` 后再并发：Ray 官方说明 owner driver 退出时这种 local runtime
  会终止。持久 head 是本次唯一额外基础设施变化。

官方依据：[Ray 启动与退出语义](https://docs.ray.io/en/latest/ray-core/starting-ray.html)、
[Ray namespace](https://docs.ray.io/en/latest/ray-core/namespaces.html)、
[ray.init / RAY_ADDRESS](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)。

## 2. RLT resolved formal packet

### Stage 1

| 项 | 值 |
|---|---|
| GPU / world | physical 4,5 / 2 ranks |
| model | exact pi0；current causal AR；frozen VLA、token-only；RTC off |
| data | canonical clean-50；50 episodes / 7,188 frames / 3 cameras / 14D |
| batch | MB16/rank、GB32；64,000 sample presentations |
| optimizer | AdamW 2.5e-5；warmup100 + cosine；clip1 |
| budget | 2,000 optimizer steps；无 val；只保存 step2000 |
| hard timeout | 10,800 s；算法正常终止仍由 step2000 决定 |

### Stage 2

| 项 | 值 |
|---|---|
| GPU / world | physical 4,5 / 2 ranks |
| rollout | 8 train env 真并发；250 cycles；每 episode 最多200 primitive；C10 |
| budget | 2,000 train episodes；最多400k primitive / 40k macro |
| replay/schedule | 50k/rank；warm-up10k/rank；post-collect30k；cap1600/cycle |
| update | macro UTD5；critic:actor 2:1；GB512/MB128 |
| actor schedule | warm-up20k + ramp50k；BC/Q 7/.05 -> 2.5/.45 |
| eval/save | 每25 cycles；4 env × 5 waves = fixed20；共200 eval episodes / 10 checkpoints |
| hard timeout | 72,000 s；算法正常终止仍由 cycle250 决定 |

统一输出：

```text
/data/chenyiteng/results/rlinf-rlt/
  formal-current-ar-stage1-2k-stage2-8env250-20260823-v1/
    runtime/
    stage1/
    stage2/
```

精确 launcher：
`local_scripts/remote_commands/shenzhen_rlt_current_formal_chain_gpu4_5_20260823.sh`。

## 3. 代码与 provenance

- branch：`codex/sz-rlt-pi0-robotwin-ar`
- formal protocol commit：`f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4`
- 新增内容仅两项：current Stage 2 formal overlay、旧 AutoDL fixed-20 seed bank；无 Python/算法改动。
- overlay SHA-256：`92e4a212b78148fd5af65fdb82fc49359b9c5321c3531e0d667bde4bed613bf6`
- seed bank SHA-256：`fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7`

## 4. 启动与停止合同

RLT chain 只在以下条件继续 Stage 2：Stage 1 exit0，且 `full_weights.pt` 与 DCP metadata 存在；随后生成
带 current-base、current-AR、数据、norm 和模型合同的 manifest。任一 stage 非零、artifact 不完整或 hard
timeout 时 chain 停止；不按中间成功率早停，不自动重启。

共享 Ray 在两个 formal 都结束前不得执行 `ray stop`/`pkill`。若 DSRL 第二 namespace 不能正常建立，保留
已运行的 RLT，DSRL 不反复重试。
