# AttenA+ 作者源码审计

2026-09-09，只读联网研究；不执行作者代码，不安装依赖，不改服务器。主规划见 [BC_ATTENA_PLUS_PLAN_20260909.md](../../BC_ATTENA_PLUS_PLAN_20260909.md)。

## 锁定来源

| 来源 | 版本 | 本地证据 |
|---|---|---|
| arXiv论文 | 2605.13548v3，2026-06-01修订 | [全文](https://arxiv.org/html/2605.13548v3)，本轮浏览§3、Table3/4、AppendixC/E |
| 作者主仓库 | 当前main `58bea853373dfd98b24ac995bad3e203761d4629` | [实时锁](upstream-main-lock.json)；main与09-04锁相同 |
| 作者OpenPI | 主仓库gitlink `fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8` | `source/openpi/`，真实JAX模型/配置/PyTorch/loader/transforms |
| 作者OFT | 主仓库gitlink `f6b63b95f3623dae4d14fd8322d399bd462f3cbf` | `source/oft/train_utils.py` |
| 作者FastWAM | 主仓库gitlink `1510b6c32af8336025a05bbb205940916249a3b4` | `source/wam/action_loss.py` |

GitHub网页部分文件返回cache miss，Windows HTTPS请求返回TLS认证错误；没有关闭TLS校验，随后使用已连接的GitHub只读工具成功获取全部列出的文本。源码内容未修改，文本文件末尾换行可能经本地保存规范化；本地快照SHA256见 [source-manifest.json](source-manifest.json)，每个文件的精确上游URL见 [source-urls.json](source-urls.json)。子模块按main固定gitlink取数，不假设子模块分支HEAD与其一致。第三方源码快照保留本地审计用途，Git可只发布本规划、审计和来源锁。

## 文件与关键行

| 文件 | 行 | 已核对事实 |
|---|---|---|
| `source/root/core.py` | 106–131 | norm(first6)→eps→mapping→上下clip→可选`/clip*2` |
| 同文件 | 138–198 | ground_truth用于权重，target用于监督；L1/MSE与pad mask可分离 |
| `source/root/pi05_libero.yaml` | 12–18 | inverse_squared、clip2、eps1e-3、alpha5、normalizeTrue |
| `source/openpi/pi0_config.py` | 37–46 | 实际模型默认inverse_squared；不用旧integration.md的inverse |
| `source/openpi/training_config.py` | 767–797 | 真实attena_pi05_libero注册值；LIBERO H10、batch256、30k步，不是我们的实验预算 |
| 同文件 | 326–342、781–785 | LIBERO已是delta，π0.5关闭额外DeltaActions |
| `source/openpi/data_loader.py` | 185–190 | data transforms→Normalize→model transforms的顺序 |
| `source/openpi/pi0.py` | 204–227 | noise/time抽样、FMtarget=noise-actions、位置MSE×w(actions) |
| 同文件 | 247–273 | 信号来自actions；没有沿H差分，没有物理dt，没有batch/episode归一化 |
| `source/openpi/pi0_pytorch.py` | 317–374 | forward仍返回F.mse_loss(...,reduction=none)，无AttenA hook |
| `source/oft/train_utils.py` | 58–117 | 函数默认inverse/clip10/alpha2/normalizeFalse，动作L1；不等于实验命令 |
| `source/wam/action_loss.py` | 12–102 | 函数默认inverse/clip2/alpha2/normalizeTrue/use_l1True；独立ground_truth/target，pad后按有效时间步均值 |

## 五个影响迁移的差异

1. **动作信号与FM速度不是一个东西。** 直接用`noise-actions`生成权重会把随机噪声注入credit，偏离作者实际代码。
2. **没有严格均值1。** 作者注释/论文写“约1”，实际算式是常量缩放。c=2时毫无归一化效果；c=1且normalizeTrue时所有权重变2。
3. **默认不统一。** main集成文档的inverse/alpha2文字与真实OpenPI注册的inverse_squared/alpha5不一致；OFT/WAM helper默认也不同。主规划以OpenPI注册为准。
4. **PyTorch不是已经集成。** 作者guide给出PyTorch运行指令，但同锁版本模型实际没有读取AttenA设置；我们需要显式接入RLinf的FM loss，不单靠flag。
5. **双臂不是改joint_dims=12即可。** 本项目6/13是夹爪；first12包含6且漏12，first6只看左臂。WAM helper也仍first6，没有在该函数中提供双臂专用规则。

## 数学推论与需要另做的适配

- `clip_max=2`、inverse_squared：norm≤1/√2时w=2，norm≥√2时w=0.5。这由公式直接推出，未声称已在实际BC数据验证分布。
- 作者动作范数在本项目normalized绝对qpos上，是相对训练均值的姿态幅值代理；与运动速度的关系未知。动作变化差分、更换范数尺度或严格均值归一都必须显式标为本地适配。
- 本轮未读出作者实际全量训练启动回执，也未复现实验；不把helper默认当实验最终参数，不把文章97.95%宣称为我们可复现的指标。

搜索覆盖论文入口、作者main/所有相关子模块、online/robotwin/normalization关键词；未找到可据以宣称其已有“在线成功BC训练器”的官方实现。本次仍采用原有在线BC，把作者权重作为监督阶段的适配。
