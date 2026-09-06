# 深圳：从 RoboTwin/ACT 成功到 official Fast-WAM 推理

更新时间：2026-08-22

状态：**完成。Fast-WAM official `adjust_bottle` 单集推理 `1/1 success`。**

范围：只总结 standalone 部署中真正必要的路径、深圳适配和实际问题；不讨论 Fast-WAM 训练或
Fast-WAM × RLinf 集成。

## 1. 起点与最终路线

native RoboTwin/ACT 已成功，只能说明深圳 H100 上的 SAPIEN、RoboTwin 资产和任务本身可以工作；
Fast-WAM 仍须使用它自己锁定的 vendored RoboTwin、模型预处理和 evaluator。我们因此没有把 ACT env、
CuRobo binary 或兼容补丁搬进 Fast-WAM，只把 native official assets 当作同机只读内容来源。

最终闭环是：

```text
official Fast-WAM current source + 独立 Python env
  -> Fast-WAM vendored RoboTwin + 独立 official assets copy
  -> official HF checkpoint/stats
  -> current loader 所需 ModelScope T5/VAE/tokenizer
  -> MPLib/NumPy 唯一兼容修复
  -> official eval_robotwin_single.py
  -> adjust_bottle 单集 1/1 success + video
```

ACT 完成态见
[`../rlinf-shenzhen-pi0-ppo-rlt/05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md`](../rlinf-shenzhen-pi0-ppo-rlt/05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md)；
Fast-WAM 完整计划和逐操作证据分别见
[`09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md`](09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md) 与
[`evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md`](evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md)。

## 2. 哪些严格沿 official 路径

- source 使用 official Fast-WAM `7faa711...`，vendored RoboTwin 使用其锁定的 `bf44be51...`；没有用
  native RoboTwin 或 RLinf compatibility tree 覆盖 vendor 代码。
- base env 沿 README：Python 3.10、Torch 2.7.1+cu128、Torchvision 0.22.1+cu128、
  `pip install -e .`。
- policy 使用 official HF release 的 RoboTwin checkpoint 和 dataset stats；Wan 推理组件沿 current
  official loader 的 ModelScope 路由获取 T5、VAE、tokenizer。
- 推理使用 official `experiments/robotwin/eval_robotwin_single.py`；它就是 official manager 调用的
  单任务入口。任务仍是 `adjust_bottle / demo_clean / unseen`，模型、观测、动作和去噪数学没有修改。
- 旧 release checkpoint 按 current README 的兼容语义显式使用 `sigma_shift=5.0` 和
  `replan_steps=24`；这不是深圳自定超参。current source 的默认 shift 已变成 1，因此不能省略覆盖。
- 该 release 的 RoboTwin 配置不需要 pretrained 5B DiT，所以只下载真实会加载的组件，没有下载 5B DiT。

## 3. 哪些是深圳隔离、复用或规模选择

| 处理 | 性质与理由 |
|---|---|
| source、env、cache、models、results 各自独立 | 深圳隔离策略；防止污染 native ACT 和 latest RLinf PPO |
| native official assets 向 Fast-WAM vendor 做普通独立 copy | 只复用同一份 official 内容，vendor 可自行更新路径；不共享可写树、env 或 binary |
| CuRobo 在 Fast-WAM 最终 Torch/CUDA env 内为 H100 sm90 重编 | 机器适配；没有复用 ACT/RLinf `.so`，也没有预搬 ACT 的 fused-LBFGS workaround |
| 只补 evaluator 实际需要的 simulation packages | current Fast-WAM 与旧 vendor full requirements 存在版本冲突，机械整装反而不成立 |
| physical GPU 3 | 资源隔离；同期 latest RLinf PPO 固定使用 physical GPU 4–7 |
| official single evaluator 缩为 1 episode | 只缩运行规模以证明推理链；不改模型语义，`1/1` 也不外推总体成功率 |
| 先做 render/expert gate | 本轮额外检查，不是最终最短推理路线；render 通过后 expert 暴露兼容问题，用户纠偏后直接回到 evaluator |

因此，从当前完成态复现不需要再跑 ACT、采集 expert 或下载训练数据；使用现成独立 env、模型和 vendor
inputs，直接运行 official evaluator 即可。

## 4. 实际问题：自身额外复杂度与真实兼容问题分开看

### 4.1 我们自身操作造成或放大的问题

| 现象 | 实际原因 | 处理 | 性质 |
|---|---|---|---|
| 首次 env 安装被取消 | 为正式 PPO 让路，主动 SIGINT | 保留 prefix/cache，从同一 env 续装 | 调度中断，不是依赖失败 |
| `packaging>=26.2` 冲突 | 启动脚本额外安装了 official runtime 不消费的 build-only `setuptools-scm -> vcs-versioning`，而 Fast-WAM pin `packaging==25.0` | 证明无人消费后只移除这两个额外包 | 自身多装依赖造成 |
| 清理脚本功能通过却 exit 1 | 脚本误以为 SSH stdout 同时写了远端日志，并对不存在路径做 hash | 不重复安装，只跑一次无该假设的最终验收 | 自身证据脚本假设错误 |
| CuRobo distribution 一度显示 `0.0.0` | 上一步移除 SCM tooling 后，CuRobo editable build 只得到 fallback metadata | 为 CuRobo build 窄装兼容的 `setuptools-scm==9.2.2` 并重装同一 source | 自身清理带来的 build metadata 问题；binary/source 没变 |
| 一次 eager-import acceptance 过早失败 | vendor assets/task config/policy 尚未准备好，却提前导入顶层 `envs` | 先完成 vendor inputs，再复跑 acceptance | 自身验收顺序问题，不是 runtime 缺陷 |

这些问题都不应包装成部署必经复杂度。新的最短复现只保留最终依赖状态和正确顺序，不重复这些探针、
清理和日志假设。

### 4.2 真正的上游组合兼容问题

真实 simulator 调用中，official expert 在 MPLib 的
`Box(side=shape.half_size*2)` 处 segfault。失败组合是 Fast-WAM pin 的 NumPy `2.2.6` 与 MPLib
`0.2.1`；后者的官方依赖合同明确为 `numpy<2`。此前 `--no-deps` 安装后 import 能通过，只能证明模块
可载入，不能证明 planner ABI/转换路径可用。

唯一修复是：**仅在 Fast-WAM 独立 env 中把 NumPy 窄降为 `1.26.4`，MPLib 保持 `0.2.1`。** 没有改
MPLib/RoboTwin/Fast-WAM 代码，没有换 planner，也没有移植 ACT 的 CuRobo workaround。按用户纠偏没有
再跑独立 expert；修复后直接运行 official evaluator，完整 simulator + policy 路径 exit 0 且单集成功，
即为复测证据。

另外两点不是本轮故障：SAPIEN 的 Vulkan warning 最终由 official `test_render.py` 的 `Render Well`
证明不阻断渲染；ACT 曾遇到的 CuRobo fused-LBFGS illegal instruction 在这套 cu128 Fast-WAM env 中没有
出现。

## 5. 最终路径、资源隔离与产物

| 角色 | 最终位置 / 结果 |
|---|---|
| Fast-WAM source | `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711` |
| standalone env | `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128` |
| cache/build | `/home/chenyiteng/cache/fastwam-7faa` |
| release checkpoint/stats | `/data/chenyiteng/models/fastwam/release-8eaceeb` |
| T5/VAE/tokenizer | `/data/chenyiteng/models/fastwam/diffsynth`；5B DiT 未下载 |
| vendor assets | Fast-WAM source 内 `third_party/RoboTwin/assets` 的普通独立 copy |
| evaluator run | `fw-sz-500-20260822_045105`，physical GPU 3，peak `30,274 MiB` |
| result | `adjust_bottle` accepted seed `4300001`，`1/1 success` |
| remote video | run 下 `adjust_bottle/episode0_randomized-false_success-true.mp4` |
| local light evidence | [`evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4`](evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4) |

checkpoint 约 12.04 GB、Wan 组件约 12.79 GB、vendor assets 约 16.66 GB，连同完整 env/cache 都留在
深圳服务器的 `/data` 或 `/home`；Windows 文档仓只保留约 88 KB 的成功视频和 Markdown 证据。

official evaluator 的精确 resolved 参数、调用细节和验收见
[`10_SHENZHEN_S4_S5_EXECUTION_PACKET.md`](10_SHENZHEN_S4_S5_EXECUTION_PACKET.md)。
