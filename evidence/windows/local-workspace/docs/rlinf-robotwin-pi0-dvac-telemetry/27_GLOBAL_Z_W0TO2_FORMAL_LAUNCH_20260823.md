# global-z DVAC `[0,2]` 100-step formal launch

日期：2026-08-23

## 1. 实验问题

本次只组合两个已经分别跑过的因素：

- **信号**沿用v1：把最近5个runner step中所有进入trajectory的`query × h`之`log(V_L3+eps)`放在一起，用全局均值/
  标准差得到`z`；它保留固定future-h位置趋势。
- **强度**沿用v3：把`z∈[-2,2]`线性映射为action梯度权重`w∈[0,2]`。

公式为：

\[
w(q,h)=1+0.5\,\operatorname{clip}(z(q,h),-2,2).
\]

因此`z=-2/-1/0/1/2`对应`w=0/0.5/1/1.5/2`。相对v1只把`strength`从`0.1`改为
`0.5`；相对v3只把R-only per-h residual换回v1 global-z。GRPO advantage仍决定强化/抑制方向，
权重只改变50个future action的反向贡献。

## 2. 冻结配置与命令

```text
source  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
branch  codex/idea2-dvac-residual-downweight
commit  afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8
config  robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal
source SHA256    48ae7a3965fa43844377862df4ec294b051b7b93cccde461d6c119cd62374149
resolved SHA256  8e7e807678be41d41d2d20644ea9105771a6c73f42f6986633370c0e0b1e6951

run      /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
runtime  /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
```

训练driver的精确命令：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal
```

后台启动仍复用v3已验证的wrapper/observer结构；observer每2秒只读记录GPU、cgroup、memory events、
磁盘和owned process，不设阈值、不发signal、不改变训练。

## 3. 与成功GRPO对齐的主体参数

| 类别 | 冻结值 |
|---|---|
| task/model | `adjust_bottle`；task-matched原始π0 SFT；fresh start |
| flow/action | train `flow_sde`；`H=C=50`；active `D=14`；`M=4` |
| rollout | 2×A800；16 env；16 rollout epochs；200 action slots/episode |
| GRPO | group size 8；chunk reward；chunk log-prob；GRPO advantage |
| update | global batch 512；mini-batch 32；update epoch 2；lr `5.6e-6` |
| clipping | PPO ratio clip `±0.2`；global grad clip `1.0` |
| run | 100 global steps；checkpoint interval 10；control trace保持既有抽样设置 |

预算上界/估计：25,600条rollout trajectory、最多102,400次policy query、约400次optimizer update、
10个checkpoint；按前几次同配置运行约41--42小时、总产物约100--105 GiB、每卡显存峰值约30.4 GiB。
这些是启动估计，不是完成事实。

## 4. 启动前最小检查

- 新source YAML相对v1只有`log_path`、`experiment_name`和`strength`三处变化。
- synthetic probe确认forward值不变、backward倍率精确为`[0,.5,1,1.5,2]`。
- Hydra `--cfg job --resolve`通过；关键字段为100 steps、16 env、16 rollout epochs、G8、B512/mb32、
  update2、global-z、`z_clip=2`、`strength=.5`。
- 启动前目标run目录不存在；GPU0/1约4 MiB；cgroup约68.1 GiB；`oom=oom_kill=0`；数据盘余669 GiB。
- 用户已明确授权直接启动本次100-step实验，不再等待新的smoke或二次确认。

## 5. 完成边界

本次目标是自然运行到Global Step100。启动确认只要求driver进入真实rollout、核心worker alive、GPU/RAM
开始增长且无立即fatal/OOM；之后退出当前交互，不持续盯守。动态step、资源、checkpoint与结果以后仍以
服务器现场刷新为准。

## 6. 实际启动与首轮rollout确认

正式训练已于`2026-08-23T19:27:49+08:00`启动：

```text
wrapper / driver / observer = 820640 / 820644 / 820645
process group               = 820638
```

`19:33:04 CST`只读刷新时：

- wrapper、driver、observer和2 actor/2 rollout/2 env worker全部alive；
- driver已经进入真实`Generating Rollout Epochs`并完成`1/16`；
- GPU0/1分别约`22.1/25.7 GiB`，cgroup约`111.83 GiB`；
- `oom=oom_kill=0`，没有driver/observer exit marker；
- run中TensorBoard config/event已创建，observer资源CSV已持续写入287行。

日志中的可选`curobo.types.math`导入traceback与v2/v3成功启动路径相同；当前核心worker仍存活且rollout继续，
因此不构成主训练失败。该结论只确认正式训练已成功开始，不表示Global Step1已经完成。

新配置提交`afdaa2e2...`已普通非force push到
`personal/codex/idea2-dvac-residual-downweight`，远端branch HEAD复核一致。

`19:40:32 CST`二次刷新时，三根控制进程和六个核心worker继续存活，首轮rollout已到`5/16`；GPU0/1约
`25.9/25.5 GiB`，cgroup约119.7 GiB，`oom=oom_kill=0`，仍无exit marker。
