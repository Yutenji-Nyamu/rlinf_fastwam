# GRPO positive τ1：本地配置与生产接点审计

2026-09-15。本轮只读本地材料；未 SSH、未运行项目测试、未修改生产源码或配置。现场 HEAD、当前 top20 停止身份及待启动实配由根任务核验。

结论：**复用已跑通 exp 源即可，算法无需新增代码。** 相对 exp 原配置，活跃方法参数只改 `scope: both → positive`、`temperature_local: 0.5 → 1.0`、`temperature_chunk: 0.5 → 1.0`。但正式预算须从 clean128 的真实 resolved 配置继承，不能从当前 top20 实配或旧 256 预算启动器推导。`positive` 指 `A>0`；本组二元成功奖励、混合结果的 GRPO group 中对应成功轨迹，不能推广为所有奖励设计下的 success-only。

**精确输入与复用路径**

| 用途 | 路径 / 已留存身份 |
|---|---|
| clean128 实配，远端权威输入 | `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-seed42-formal200-phys45-20260911-v1/runtime/resolved.yaml` |
| clean128 本地实配副本 | `docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence/dvac-new-half128-seed42-20260912/clean-closeout/package/raw/runtime/resolved.yaml` |
| clean 源 root | `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-seed42-20260911`；已有留存 HEAD `b51f185295e06bf22fbf51e94de05a15e8fef4eb`，源 parent `76c2ac4c40dd211d6e6eaab84022e75a61ea5ed7`；这些是历史身份，启动前现场核验 |
| 已跑通 exp 源 root | `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-dvac-exp-20260913` |
| exp 原 run | `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-exp-half128-seed42-t05-formal200-phys45-20260913-v1` |
| exp recipe | 源 root 下 `examples/embodiment/config/dvac_grpo/adv_exp.yaml`；本地副本 `local_scripts/method_cutover_20260913/grpo-src/examples/embodiment/config/dvac_grpo/adv_exp.yaml` |
| exp 源文件清单与 SHA | `local_scripts/method_cutover_20260913/grpo-source-manifest.json`；纯权重函数 SHA256 `0a2114a341d73cbdf922aaddf4447c2304c6fe857f2377724efdd75bcdb0cc3f` |
| exp 已跑通 driver / 基础 ops | `local_scripts/method_cutover_20260913/ops.py`；远端旧副本 `/data/chenyiteng/results/server-maintenance-20260913/method-cutover/ops.py` |
| 显式继承 clean 实配的最近模板 | `local_scripts/grpo_top20_cutover_20260914/ops.py:38` 与同目录 `specs.json` 的 `baseline_root/baseline_run`；只借 clean 继承、路径替换、leaf diff、seed 文件哈希、唯一输出/回执等流程 |

**新 resolved 的方法块**

```yaml
algorithm:
  dvac_gradient_weighting:
    mode: apply
    application: chunk_clipped_action_advantage
    normalization: two_level_group
    scope: positive
    selected_l: 3
    mapping: exp_mean
    alpha_local: 1.0
    alpha_chunk: 1.0
    temperature_local: 1.0
    temperature_chunk: 1.0
    log_eps: 1.0e-12
    minmax_eps: 1.0e-6
    save_step_tensors: true
    output_dir: <new_run>/<new_experiment_name>/dvac_train
```

最小做法是在 clean 的 `dvac_gradient_weighting` 字典中更新上述活跃字段，保留原有非活跃历史字段可减少无关 diff。clean 原 `mode=off, application=logprob_st` 必须相应覆盖；不能只加 scope 与 τ。`strength=0.5/warmup_steps/window_steps/std_floor/z_clip/weight_min/weight_max` 不进入 `two_level_group` 活跃路径，不是本轮幅度参数。`algorithm.dvac_top20` 应不存在或明确禁用；新分支从 exp 源出发无需迁入 top20 helper、Adam 旁路或独立抽样状态。

**scope 与温度全链支持证据**

行号以下均对应本地保留的准确源码：

1. rollout：`local_scripts/grpo_seed42_half_20260911/src/rlinf/workers/rollout/hf/huggingface_worker.py:162` 从同一 `algorithm.dvac_gradient_weighting` 读取模式；`mode=apply` 启用采集。`:683` 在训练查询请求 telemetry，`:731` 计算并写 `forward_inputs[dvac_v_l3]`。这里没有需要额外打开的 openpi 配置开关。
2. actor：`local_scripts/method_cutover_20260913/grpo-src/rlinf/workers/actor/embodied_fsdp_actor_worker.py:143` 接受 positive；`:152` 校验两层温度；`:636` 将 scope、alpha、mapping、两个 τ 纳入实际合同。`:697` 汇集全 rank 的 V、mask、A、贡献质量和原生 group IDs，`:708` 将整个合同传入权重函数；`:717` 才切回各 rank，`:897` 在 minibatch/更新前冻结权重。
3. 权重：同 packet `rlinf/algorithms/dvac_two_level.py:208` 先构造 actor-valid、有贡献的 eligible 域，再用 `A>0` 限定；`:235` 域外 local 因子置 1；`:239` 至 `:255` 外层统计只读 eligible 域，chunk 因子从全 1 开始；`:257` 两层相乘。因此失败域 W 精确为 1，成功域的归一化不混入失败 V。不能先 scope=both 算完再把失败置 1。
4. loss：actor `:1092` 传入 `dvac_chunk_advantage_weights`。继承的 `local_scripts/grpo_new_switch_20260912/src/rlinf/algorithms/utils.py:412` 将已算好的 A 乘 detached W；同目录 `losses.py:170` 的 action surrogate 保留原 chunk SUM-logprob ratio、PPO clip、dual clip 和原 reducer；没有乘权后再进行全局优势白化。失败域 W=1 的直接样本梯度等于该时刻 Control 公式，但成功梯度变化会经共享参数和 Adam 状态影响后续所有样本，不能声称整个失败学习过程永远不变。
5. 温度：`dvac_two_level.py:68` 的两层 exp 因子均为 `(1-alpha)+alpha*exp(z/τ)/mass_mean(exp(z/τ))`；本轮 alpha 均 1。两个 τ=1 均通过相同映射合同传入，不需要新分支。τ 提高使同一批数据上的权重分配更平缓，不改 LR、PPO clip 或 rollout noise。

**必须继承的 clean128 合同**

| 项 | clean resolved 已核值 |
|---|---|
| 任务 / 初始模型 | `move_pillbottle_pad`；actor、rollout 均 `/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab` |
| 训练采集 | 64 env × 2 rollout epochs = 128 episodes/轮；group=8；max episode=200 |
| 动作与采样 | H=50，action_dim=14；4 queries/完整 200 步轨迹；num_steps=10；openpi noise_level=0.5，flow_sde |
| 优化预算 | global batch=512，micro=32，update_epoch=2；LR=5e-6；clip_grad=1；其余 Adam 参数原样 |
| GRPO / loss | adv_type=grpo；reward/logprob_type=chunk_level；token-mean；normalize_advantages=true；filter_rewards=true，0.1/0.9；PPO clip low/high=0.2、dual clip=3；KL beta=0、entropy bonus=0 |
| 随机数 | rollout.seed=42，加 logical rank 得 42/43；**actor.seed=1234、env train/eval seed=0，均保持**；train/eval seed JSON 内容哈希与 clean 一致 |
| 评估 | fixed reset IDs；32 eval env × 1；每 5 轮评估；原 seed42 patch 在 evaluate 中保存/恢复训练 RNG |
| 训练终点 / checkpoint | fresh max_steps=200、max_epochs=1000；save_interval=10；resume_dir=null、ckpt_path=null；local_shard |
| 资源 | actor/env/rollout 在物理 GPU4/5；原 FSDP、precision、offload、weight_syncer 设置完整继承 |

身份类改动包括新 root/run/namespace、日志和输出路径、引用新 root 的 seed 文件路径；seed 文件内容不变。新 resolved 须对 clean 全部叶子做 diff，仅允许上述 DVAC 字段与实际需要的身份路径变化。13:49 准备结果的独立全叶复算现已通过，见文末补审。

**现有测试范围足够覆盖配置复用，不扩算法测试**

- `tests/unit_tests/test_dvac_exp_mean.py`：`:73` 参数化 both/positive/negative，验证域外 W=1、域外 V 极大变化不影响 eligible 权重；`:104` 验证 τ 与 alpha；`:47` 验证两层质量归一；另有空域、常数、极小 τ 与无效参数检查。源在 `local_scripts/method_cutover_20260913/grpo-src/`。
- `tests/unit_tests/test_dvac_adv_new_actor.py:230`：两个 rank × 三个 scope × 两种 mapping，验证真实 actor 函数全量 gather、冻结、分片与合同；已有实际两进程 Gloo 测试和 sidecar 合同变更拒绝。源同上。positive actor 参数化案例用 τ0.5，不冒充此前已经 GPU 跑过 positive τ1。
- `tests/unit_tests/test_dvac_adv_new_loss.py:82`：H=2/50、mask、clamp、dual-clip 下 all-one 与 Control 的 loss/metrics/gradient 一致；`:117` 检查正负两侧梯度与 chunk 强度。源在 `local_scripts/grpo_new_switch_20260912/src/`。
- 原 `tests/unit_tests/test_dvac_two_level.py` 保留；原 `tests/unit_tests/test_grpo_rollout_seed.py` 覆盖 rank 种子、eval 不消耗训练 RNG、异常恢复及真实 GRPO 噪声路径。seed 源在 `local_scripts/grpo_seed42_half_20260911/src/`。
- 今日已取回的只读 exp R98–100 六份冻结张量反事实，直接运行了 positive + 两层 τ1，证明该批数据域/质量恒等式正确；这属于系数核验，不能替代新训练收益证据。此前实机跑通的是 exp/both 路径。

若根任务按既有服务器 CPU 检查流程复用，精确文件集即旧 exp specs 的五项：`test_dvac_two_level.py test_dvac_exp_mean.py test_dvac_adv_new_actor.py test_dvac_adv_new_loss.py test_grpo_rollout_seed.py`，保持既有 CPU/CUDA selector，不额外扩测试或引入 top20 的 coin/sidecar smoke 条件。

**启动模板的具体陷阱**

旧 `method_cutover_20260913/ops.py:70` 从 `old_run` 读基线，`:93` 只替换 mapping 和两个 τ；其旧基线本来已经是 two_level/both。当前 old_run 是 top20，原样复用会错误继承方法或预算，而且不会自动设置 positive。应改用最近 top20 模板 `prepare:38` 的显式 clean baseline 输入，再覆盖本审计方法块；新鲜当前 top20 身份仅用于根任务授权范围内的精确停止，不充当新配置基线。旧脚本中历史 driver PID、namespace、source receipt、ST 目录和 STOP receipt 也不能直接重放。

没有发现必须修改 exp 算法源码的缺口；审计任务自身仅写本文档。实际准备新增一份固定参数 recipe，未改模型路径、训练预算或恢复流程。

**13:49 准备结果独立补审：通过**

本次直接读取 `E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/` 下本轮 `preflight.json` 的 `runs.clean.runtime[resolved.yaml]`、`runs.exp.runtime[resolved.yaml]`，以及 `prepared/prepared-formal/resolved.yaml`、`contract.json`；未用旧快照代替基线。独立按 YAML 缩进和序列解析所有叶子，不调用项目代码；clean 283 个叶子、formal 291 个叶子。

- 对 clean 的差异独立复算为 **19 处：10 个方法字段、9 个身份/输出路径**，与 contract 的 `baseline_diff` 逐项完全一致，`unexpected_nonmethod_diff={}`。10 个方法字段为 alpha_chunk、alpha_local、application、mapping、minmax_eps、mode、normalization、scope、temperature_chunk、temperature_local；selected_l、log_eps 等原值保持。新配置无任何 top20 叶子。
- 将 exp 的 root/run、实验名及 DVAC 输出路径替换为新身份后，独立 diff 恰为 **scope、两个 temperature，共 3 个方法字段**，另有 actor/env/rollout 的 3 个 group_name 恢复为 clean 名称；与 contract 的 `exp_method_diff` 完全一致。没有发现其他方法外叶子差异。
- clean 原始 YAML 的 SHA256 独立复算为 `3049922cd93199ae5a9e8848efd7a0da482c38d924c146a2b420f927bce8ecfd`；formal 文件为 `6b94592e12352a40eccfdbe2d7314f5a5b79b6cfe1defd93d20b1ee768671e5a`，均匹配合同。128 episodes、G8、B512/micro32、U2、LR5e-6、seed42 与原 actor/env seed、fixed32/每5、save10、fresh200 均保持。
- 源码回执 HEAD 为 `1732f1ef3e4a993657ac54dc052e6f0d90d2afc3`，基于本次 preflight 确认的 exp HEAD `04341d74d0bbefa097f795bc752deabc7e5e4dfd`；只新增 `examples/embodiment/config/dvac_grpo/adv_exp_positive_t1.yaml`，SHA256 `eb34561cc24f3f24a98316522ca2239491362b36dbf002bb46fd5cde86c5dee8`。preflight 单独采集的 3 个核心源码（权重函数、actor、hf rollout）与新合同 SHA 逐项一致；源码回执另记 `algorithm_source_unchanged=true`。
- 合同锁定新 root `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-dvac-positive-t1-20260915`，新 run `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-positive-t1-half128-seed42-formal200-phys45-20260915-v1`，namespace `RLinf_dvac_positive_t1_half128_seed42_formal45_20260915`，GPU `[4,5]`。已读 command 指向本轮 `/data/chenyiteng/results/server-maintenance-20260915/grpo-positive-t1/ops.py` 与新 run/runtime；环境中的代码工作目录/PYTHONPATH 指向新 root。
- 本轮实读 `prepared/cpu-tests.log` 与 `code-receipt.json`：**41 passed、1 deselected，12.07 秒**。实际范围为完整 `test_dvac_exp_mean.py`、actor 的 `test_actual_prepare_gathers_then_freezes_native_chunk_weights` 参数化案例、`test_grpo_rollout_seed.py`，排除 CUDA sentinel `test_real_cuda_coverage_required`。上文五项是已有覆盖清单，本轮没有重跑全部五文件，也未新增测试。

本轮沿用旧真实 exp smoke 回执 `/data/chenyiteng/results/server-maintenance-20260913/method-cutover/grpo/smoke-result.json`，不新增 GPU smoke。此补审确认已准备配置与合同一致；不提前声称新训练已启动健康。根任务按已定流程切换后核实际配置、actor 存活及采集即可，无新增阻塞项。
