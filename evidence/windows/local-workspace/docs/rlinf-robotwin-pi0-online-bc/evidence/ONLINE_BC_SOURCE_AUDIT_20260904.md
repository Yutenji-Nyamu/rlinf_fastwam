# 在线 BC 调研：固定源码与可复核边界

日期：2026-09-04，GitHub API 锁定约20:19–20:26 CST。
本轮只检索／下载公开源码文本到本地证据目录，不执行下载的工程代码，不改远端训练。
当前专题唯一规划：[00_RESEARCH_AND_PLAN.md](../00_RESEARCH_AND_PLAN.md)。

## 1. 来源锁

| 仓库 | 固定 SHA | 备注 |
|---|---|---|
| RLinf/RLinf | dc9b87cc49334c7516487ead68ebeb060fd7c090 | 上游main，不是深圳部署HEAD |
| huggingface/lerobot | 3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e | 09-03提交；RA-BC在rewards/sarm目录 |
| hiors-project/hiors | 5fa4b23f80f420dcb340875daec72a051e161fae | MIT；2025-11提交 |
| Jasper-aaa/SEIL | dab0eb4bdd16e19f6e98d95ef0bfbd857f315644 | 2025-09提交；API未给出许可证 |
| alibaba-damo-academy/RynnValue | 10e0d333f5f3811d0d130587e50f1faf48da49e5 | 2026-08提交；API未给出许可证 |
| Lei-Kun/RL-100 | 64264d952c1fda9d5096c090ddaa7177757a77ad | 09-01提交 |
| DaojiePENG/AttenA-Plus | 58bea853373dfd98b24ac995bad3e203761d4629 | 2026-05提交；README许可证badge指向的LICENSE未在根树看到，不作已核实许可证判断 |

锁及tree在 `source-audit-20260904/<owner>__<repo>/lock.json` / `tree.json`。
选中源文件在各自的 `source/` 下。只抓相关文件，没有克隆submodule或下载模型、数据集、视频。
读取作者代码不等于已验证其完整可运行；实际移植还需检查许可证和目标依赖。

## 2. RLinf：对应符号与发现

下表相对文件路径均以固定仓库为根；[官方固定树](https://github.com/RLinf/RLinf/tree/dc9b87cc49334c7516487ead68ebeb060fd7c090)。

| 文件／位置 | 实际证据 | 对首版的影响 |
|---|---|---|
| `workers/rollout/hf/huggingface_worker.py:494`（均在rlinf/下） | DAgger使用mode=eval | 采样模式不能无意继承 |
| 同文件510–550 | 只有非空expert才会专家执行／重标注，替换forward_inputs.action/model_action | 无expert能自采，但还需正确接收 |
| `workers/actor/fsdp_dagger_policy_worker.py:300–307` | 普通replay只extract_intervene_traj(all) | 删expert不是完整self-BC开关 |
| 同文件219–241 | 有放回采样，每epoch总样本数是global_batch_size | epoch不是全池遍历 |
| 同文件441–459、469–565、630–677 | prepare→ForwardType.SFT→梯度累积→一次optimizer；update_epoch循环；跳过adv | supervised更新设施可以复用 |
| `models/embodiment/openpi/openpi_action_model.py:377–414` | 基础forward得到逐元素FM MSE；可裁动作chunk/环境维，再mean | 在mean前加入有效mask和DVAC |
| 同文件615–715 | model_action直接用；物理action经input_transform；LeRobot组三路图像 | 保留labelspace，避免重复norm |
| 同文件911–940 | 同时记录transform前model_action、transform后action和clone的输入obs | query/chunk数据路径已有合适源头 |
| `envs/robotwin/robotwin_env.py:322–390` | 单个chunk后obs；term/trunc只在C-1 | 不是primitive逐帧流 |
| `data/schema/embodied_trajectory_builder.py:737–903` | only_success需成功且termination；循环len(obs_list)，按step_idx取action、done | C>1 RoboTwin不能直接拼该collector |
| `data/datasets/dagger/in_memory_store.py:251–299` | 逐帧chunk采样，会生成episode-boundary padding mask | 不等于actor已把该mask用于FM；需要沿链确认 |

最小可复核推导：C=50，RoboTwin `len(obs_list)=1`，builder只执行step_idx=0；动作取a0，图像是chunk后图像；term[0]=false而真实标记在term[49]。
这是固定源码的接口审计；没有在本轮部署运行该组合，也未提交上游issue。

## 3. 其他公开实现

- Hi-ORS `config/algo/reject_sampling_pi0.yaml:8–30`：只SFT、成功池阈值10、每50 learner step发布网络。
  `examples/train_rlpd.py:348–405`：动作存入episode列表，末尾判定成功，再放成功池；还过滤too-short/no-op。
  `:540–550`看的是`len(successful_buffer)`，不能说“先攒10个episode”。
  `:752–814`成功池采样并反传；`:840–844`是参数发布。
  `agents/continuous/sac_pi0.py:464–493`调π0 FM且hard-code前7维；padding处理被注释，不自动照搬。
  [官方代码](https://github.com/hiors-project/hiors/tree/5fa4b23f80f420dcb340875daec72a051e161fae)。
- SEIL `SEIL/baku/record.py:131–221`：每任务按配置rollout，成功存pkl；`train.py:299–306`支持load_bc。
  论文说公平设置从scratch训练，但公开config `load_bc:true`：不同stage的重启语义不能只凭一句话概括，应按具体实验配置核验。
  `select/config_inf.yaml`样例num_to_save=25，而论文主设置提15，不能混为一张已复现合同。
  [官方代码](https://github.com/Jasper-aaa/SEIL/tree/dab0eb4bdd16e19f6e98d95ef0bfbd857f315644)。
- LeRobot `utils/sample_weighting.py`：独立weighter接口；`rewards/sarm/rabc.py:144–173`统计episode内delta，`:215–235`计算并归一化weights，`:262–295`分段软／硬阈值。
  `scripts/lerobot_train.py:198–208`：policy先给per_sample_loss，再加权归约。
  原实现的missing-index／invalid-delta fallback不作为本项目设计承诺；若我们请求外部权重却缺失，不应默默宣称加权方法运行成功。
  [权重源码](https://github.com/huggingface/lerobot/blob/3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e/src/lerobot/rewards/sarm/rabc.py)。
- RynnValue `pi-rl/scripts/train_iql.py` 有实际Q/V和weightedFM桥接；`episode_filter.py`是静态episode白名单，不是在线成功判定。
  `examples/franka/train_online_async_lora.sh`启动EXPO-FT；同目录`train_franka.py`可见PixelSACLearner／噪声动作空间；不能将这些脚本统称onlineIQL。
  [官方代码](https://github.com/alibaba-damo-academy/RynnValue/tree/10e0d333f5f3811d0d130587e50f1faf48da49e5)。
- AttenA+ `attena/velocity_attention.py:106–131`：选定action维度取norm、inverse映射、clip和固定倍数缩放；`:133–190`支持FM独立target及padding。
  `docs/openpi_integration.md`给JAX OpenPI接点；本轮未递归检查submodule具体修改，不能说已验证RLinf PyTorch直接可用。
  [作者实现](https://github.com/DaojiePENG/AttenA-Plus/blob/58bea853373dfd98b24ac995bad3e203761d4629/attena/velocity_attention.py)。
- RL-100 `tools/teleop_off2off_data/DATA_PREPARE.md`与配置：支持轮次数据源、增量写新zarr、source_manifest去重；不需要把其PG/IDQL/one-step蒸馏全带入BC。
  [数据工具](https://github.com/Lei-Kun/RL-100/blob/64264d952c1fda9d5096c090ddaa7177757a77ad/tools/teleop_off2off_data/DATA_PREPARE.md)。

## 4. 论文与状态证据

- [Hi-ORS原文](https://arxiv.org/html/2510.26406v1)：§III-B为成功/阈值加权FM，§III-D异步UTD约1，§IV-E解释人工纠正及过滤；不是固定批次N轮实验。
- [SEIL原文](https://arxiv.org/html/2509.19460v1)：§V-A3各模型25rollouts、选择15；TableIII扫描10/20/50/100；不把一次小论文的最优预算当通用结论。
- [Batch Online RL原文](https://arxiv.org/html/2505.08078v1)：§4/AppendixB的200×10–20与batch256；[ICLR2026版](https://openreview.net/pdf/ec3cce7293b2ec2b09590c196e27c316a636a87d.pdf)搜索摘要确认正式发表，但直接打开被浏览器验证阻挡。预算本轮引用的是明确可读v1，不假称终版全文已读。
- [SARM作者项目页](https://qianzhong-chen.github.io/sarm.github.io/)明确Accepted to ICLR2026；[LeRobot文档](https://huggingface.co/docs/lerobot/en/sarm)明确π0/π0.5/SmolVLA与40000-step样例。
- [RL-100作者新闻](https://lei-kun.github.io/)与[项目页](https://lei-kun.github.io/RL-100/)确认Science Robotics2026；publisher直接访问本轮失败，不以未读publisher内容作证。
- [AttenA+论文](https://arxiv.org/abs/2605.13548)作者Daojie Peng与GitHub用户对应；README的会议信息含占位说明，未验证NeurIPS录用。

## 5. 本轮操作与未执行事项

- 阅读根规则／交接／指定近期窗口；当前问题路由到实验扩展SSOT，再新建在线BC专题，不扫全部实验历史。
- 使用 `local_scripts/research_online_bc_sources_20260904.cjs` 获取公开API锁／选中源文件；无认证、无依赖安装。
- 手工沿collector→replay→SFT检查，发现并记录RoboTwin/LeRobot粒度不匹配。
- 新建研究规划与本证据；更新根交接和近期窗口路由。
- 没有修改算法、创建服务器分支、推送、运行smoke/训练、刷新或干预两项现役实验。
- 未解决：部署source-lock差异；实际执行prefix长度来源；BC更新预算；DVAC定义与权重映射。它们属于下一次讨论／实施合同，不在本轮偷偷作决定。

## 6. 独立上下文整理补充核验（2026-09-04）

范围：用户要求从其他窗口杂乱材料中整理干净 RoboTwin／RLinf／π0 在线 BC 的参考代码优先级；本次只阅读、核验公开源码和维护文档。

### 6.1 新增来源锁与判断

- RLinf 后续参考提交：`a3795cc17b17e6f452edb03debc82a4b2068120d`。本次重新请求 [GitHub compare API](https://api.github.com/repos/RLinf/RLinf/compare/dc9b87cc49334c7516487ead68ebeb060fd7c090...a3795cc17b17e6f452edb03debc82a4b2068120d)，返回 `status=ahead, ahead_by=1, total_commits=1`，文件只有 `README.md`、`README.zh-CN.md`。因此本证据§2所读源码在这两锁之间未变；未断言后续 main 或服务器部署状态。
- SIME：`831824101dee6900bdee6f65a925523676920326`。完整读取作者仓库的 [`simulation/run_full_multi_round.py`](https://github.com/EricJin2002/SIME/blob/831824101dee6900bdee6f65a925523676920326/simulation/run_full_multi_round.py)，并再次带行号提取关键调用核对。只执行读取文本的工具，没有执行该训练脚本。

| SIME 行号 | 证据 | 参考边界 |
|---|---|---|
| 20–36 | `generate_training_cfg` 设置 `resume_ckpt=None`、`resume_epoch=-1` | 不将其编排误当成“每轮自动接续上一轮模型和 optimizer”的实现 |
| 79–80 | rollout 参数含 `n_rollouts=500`、`try_times=5` | 仅属该脚本设置，不迁为本项目采集预算 |
| 115–188 | 提取、合并数据，包含 `success/train_success` 与多个 `sr_lss/train` 分支 | 可以参考结果筛选和累计数据组织；不是所有分支都等于纯成功筛选 |
| 343、394、446 | 显式 `enable_exploration` 和 noise scale | 探索机制不随编排一起引入 |
| 361、385、413、437 | 主训练调用使用 `train_0.5` | 不能把默认训练配置描述为直接使用 `train_success` |
| 454 | `for i in range(2, 6)` 调用后续轮次 | 有直接多轮脚本；只借编排，不照搬模型、任务、轮数和训练重置语义 |

### 6.2 材料取舍与操作记录

1. 完整读取用户附件及指定根文档／窗口路由，沿当前专题读取规划和源码证据；按附件线索读取 seek 的主交接、实现证据与在线 BC 综述，没有遍历全部历史实验。
2. 网络读取初次遇到浏览工具 cache／URL 问题以及 PowerShell TLS 错误；随后用现有 Node 的标准 `fetch` 正常读取公开 compare API 和固定 SIME raw 文件，没有关闭证书校验、安装依赖或切换 SSH 认证。
3. 修改前查询精确目标 Git 状态：相关文档为 untracked。保留已有内容，将旧广搜规划移到 `history/RESEARCH_DISCUSSION_20260904.md` 并标记历史；当前入口仍为原路径 `00_RESEARCH_AND_PLAN.md`，没有建立并行 SSOT。
4. 当前入口按 RLinf 主干 → Hi-ORS 成功池/FM → SIME 编排 → LeRobot 局部存储排序；两处 DAgger/RoboTwin 接口问题保留为必要适配。
5. 不采纳为既定合同：引用数排名作为工程优先级、混 D0 必须／禁止、`50×5`／`256×100` 等预算、SIME 默认探索和重置训练。
6. 更新根交接与近期窗口路由说明；文档检查仅核对链接、结构和目标文件，不进行项目 import／测试。

结果：独立当前上下文已收敛；本次没有实现、提交／推送、服务器操作或训练状态刷新。下一步仍是确定数据配方和预算后，再授权文件级实施。
