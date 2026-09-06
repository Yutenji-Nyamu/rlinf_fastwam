# RLT checkpoint 与 RLinf 双 job 调查流水账

日期：2026-08-24  
范围：服务器现场、源码/官方资料、RLT 精确停止、checkpoint 最小 A/B 与并发方法论。

## 1. 操作与结果

| 序号 | 操作 | 结果 |
|---:|---|---|
| 01 | 读取专题 SSOT、并发结论、既有 live 指标和 operation ledger | 锁定 RLT v3、DSRL v2、shared Ray 和授权边界；不复述历史状态为当前 |
| 02 | 10:49--10:52 CST 通过既有只读现场入口刷新两个 run | DSRL 完整 154/200、Step155训练中；RLT仍卡 Step25 save；host/GPU/Ray 健康 |
| 03 | 核 RLT partial checkpoint tree 与 actor/driver 末尾 | 仅 99-byte incomplete manifest；无 DCP shard/metadata；两 actor 均停在 distributed optimizer-state 路径 |
| 04 | 用 DSRL 的并发进展和 checkpoint 作反事实核对 | RLT hang 后 DSRL 继续训练并成功保存 Step65/130；排除共享 Ray/GCS、磁盘、主存或通用 DCP 全局失效作为主因 |
| 05 | 对照 RLT smoke/formal 的 optimizer 状态 | smoke 保存前已有8次真实update；formal首次保存为update0。此时一度按单optimizer路径推断已有synthetic state；序号14的逐调用链复核已纠正该判断 |
| 06 | 核 PyTorch DCP/FSDP 官方源码和 issue | DCP在写shard前提取Stateful/optimizer state；fresh/partial optimizer与collective hang均有官方已知边界 |
| 07 | 核并发启动器、current RLinf cluster源码与resolved config | 并发只改运行层；算法0 LOC。稳态新增约82行控制，事故探针/恢复约412行，不进入算法热路径 |
| 08 | 形成最小A/B方案 | shared Ray下比较0-update save与1-update save；只有二者不能解释时才做isolated-Ray对照 |

## 2. 当前停止边界

- 没有停止 DSRL：它尚未完整到200，资源正常，最新可恢复点是Step130；建议自然结束。
- 没有停止 RLT：虽然该run已不可恢复且不再推进，但精确停止owned PGID及namespace属于进程控制，等待用户明确
  批准最小复现方案。
- 没有执行全局 `ray stop`、没有修改PyTorch、没有删除partial checkpoint或旧失败目录。

## 3. 下一条有状态操作

若用户批准：

1. 保存当前RLT日志和partial tree清单；
2. 只停止RLT v3 owned PGID并清它的exact namespace；
3. 保持DSRL和shared Ray不动，运行一次0-update save复现；
4. 运行一次至少1次真实update后的save对照；
5. 依据A/B结果决定“延后首次full checkpoint”或进一步查process group；不预先patch barrier。

## 4. 用户授权后的实施续账

用户于2026-08-24明确授权：精确停止不可恢复的RLT v3，做0-real-update与post-update最小A/B；若能有效
定因、窄修并验证，可自行重新启动RLT Stage2正式训练。DSRL与shared Ray必须保持不动。

| 序号 | 操作 | 结果 |
|---:|---|---|
| 09 | 03:12:46 UTC重新执行既有RLT/DSRL只读live入口 | RLT wrapper `390905`仍卡Step25 save；GPU4/5约17.4 GiB且util0。DSRL wrapper `370980`已完整到Step160/200并继续训练；GPU6/7约34.4/34.1 GiB；host约1.9 TiB available；旧actor0 |
| 10 | 运行只读`shenzhen_rlt_dsrl_namespace_pid_inventory_20260824.sh` | 精确确认RLT=`job 0b000000 / namespace RLinf / actor PIDs 391536等 / GPU4--5`；DSRL=`job 08000000 / namespace RLinf_2 / actor PIDs 371579等 / GPU6--7` |
| 11 | 新增并远端`bash -n`检查`shenzhen_stop_rlt_v3_checkpoint_hang_keep_dsrl_20260824.sh` | 脚本只允许目标namespace恰15个RLT actor且DSRL namespace恰15个actor时继续；语法通过 |
| 12 | 执行上述精确停止脚本 | TERM RLT owned PGID `390905`，清空`RLinf` 15个named actor；RLT GPU4/5降至5 MiB。DSRL wrapper `370980`、`RLinf_2` 15 actor、shared Ray全部保留，GPU6/7约34.1 GiB；partial Step25未删除 |
| 13 | RLT停止后按DSRL基线签名复核 | DSRL PID/job/namespace/6主actor/GPU映射全部不变，并由完整Step160自然推进到161；指标finite、错误0、OOM/PSI0；host available随RLT释放升至约1.93 TiB。证明本次精确清理未扰动DSRL |
| 14 | 沿实际 RLT/SAC 调用链复核 `FSDPModelManager.build_optimizers()` 与 checkpoint 合同 | RLT 使用 plural builder；它构造 actor/critic 两个 Adam 后直接返回，漏掉 singular builder 已有的 `warmup_optimizer_state()`。因此 `update_step=0` 时两个 state 预期为全空；保存时才由 PyTorch DCP 内部临时初始化。最窄候选修复是构造期逐 optimizer 调用既有 helper，不改算法目标或更新预算 |
| 15 | 建立独立诊断 worktree `codex/sz-rlt-checkpoint-diagnosis`，只加入 env-gated optimizer-state 日志 | 源基线仍为 `f3ea5f6...`；日志只在 `RLINF_CKPT_DIAG=1` 时输出每rank、每optimizer的params/state/missing/empty/step计数及 `get_state_dict` 前后标记；server compile与`git diff --check`通过 |
| 16 | 新增、上传并 `bash -n` 检查参数化 A/B 启动器 | 使用同一shared Ray、GPU4--5、4 env×1 cycle、fixed Stage1-2000、GB512/MB128；eval关闭、Step1保存、结果绝对隔离。A0保持formal warm-up所以0真实update；B只把warm-up降到能产生1次update；A′仅加入构造期prewarm |
| 17 | 启动 A0 `a-zero-update-pre-fix` | wrapper `1363933`，observer `1363934`；run root `/data/chenyiteng/results/rlinf-rlt/diagnostics/rlt-ckpt-ab-20260824/a-zero-update-pre-fix`；DSRL/shared Ray保持运行，等待进入Step1 save |
| 18 | A0 终态 | exit0、完整DCP shards/metadata/full weights/RLT sidecar。保存前两rank×两optimizer均为`state_entries=0, missing=1`；`get_state_dict`返回后均变为`state_entries=1, step=1`。证明empty state真实存在，也证明它不是每次都必挂；同时证明DCP的lazy init会把尚未训练的Adam推进到step1 |
| 19 | B `b-one-update-pre-fix` | 将warm-up降到2、只调度1次真实训练；exit0。保存前后两optimizer均已有完整state且`step=1`，checkpoint完整。与既有8-update smoke一致，post-update路径稳定 |
| 20 | A′ `a-zero-update-post-fix` | 只在plural builder返回前对两个optimizer调用既有`warmup_optimizer_state()`；exit0。保存前后两rank×两optimizer均为完整state且`step=0`，checkpoint完整。由此验证窄修满足RLinf checkpoint前置合同，并保持fresh Adam首次真实update语义 |
| 21 | 秒级对齐 RLT Step25 与 DSRL | RLT于01:39:19 CST进入save；DSRL当时在Step19普通UTD20更新，第一份Step65 checkpoint晚3:32:55。RLT进入save后GPU4/5归0，host仍约1.87 TiB available、PSI/OOM为0、首个shard未写出；排除同时checkpoint、磁盘大写入、主存压力和shared-Ray整体故障 |
| 22 | 将最窄两行修复放入正式RLT worktree并运行聚焦回归 | 只改`FSDPModelManager.build_optimizers()`：逐optimizer调用既有helper。回归确认2个optimizer均initial step0、zero moments、权重不变，第一次真实step后均为1；compile与diff check通过 |
| 23 | commit并push正式RLT分支 | `8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1 fix(fsdp): initialize all optimizer states` 已推到`Yutenji-Nyamu/rlinf_fastwam`的`codex/sz-rlt-pi0-robotwin-ar`；正式worktree clean |
| 24 | 启动RLT Stage2 formal v4 | run root `/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix`；wrapper `1389988`、observer `1389989`；GPU4--5、8env、250 cycles、UTD5、GB512/MB128、fixed20/save每25步均与v3相同；DSRL/shared Ray不动 |
| 25 | 对照PyTorch 2.11与RLinf official源码/issue收束证据边界 | iterable optimizer为正式接口；fresh Adam由DCP内部lazy fake-step且不会自行复位0，官方`#164929`确认会改变首次真实更新；RLinf singular builder已有warmup合同而plural遗漏。没有官方证据支持“多optimizer必挂”“empty state必挂”或通用加barrier |
| 26 | v4早期运行与DSRL并发复核 | v4完整推进到Step12，`update_step=0`符合replay warm-up，fatal/error/exit marker为0；DSRL在v4启动后由Step167自然推进到169，PID/job/namespace/GPU映射不变，fixed169=`10/12`，资源与错误正常 |
| 27 | v4正式跨越原Step25卡点 | fixed-20=`0/20`且`update_step=0`，与v3事故条件一致；12:25:52 CST进入save。DCP `.metadata`与两rank shard完整，full weights、两份target model、3,639个replay文件及RLT sidecar均落盘；sidecar=`complete=true / saved_runner_step=25 / update_step=0`。driver随后完整打印Step26，exit marker无、fatal/error=0 |
| 28 | 12:31 CST最终并发与系统轻量刷新 | RLT v4完整到Step28/250、DSRL完整到Step180/200，两者exit无、错误0；GPU4--5约21.6/21.9 GiB，GPU6--7各约34.1 GiB。host约1.8 TiB available；`/`、`/home`、`/data`分别余234 GiB、2.2 TiB、2.8 TiB；两项训练继续共享persistent Ray；正式RLT worktree clean且HEAD=personal remote=`8bbd0216...` |

截至序号28，A0/B/A′、正式分支回归与formal v4 Step25→26终验均完成。结论边界：plural builder漏warmup是
已证实的checkpoint合同与Adam-step语义缺口；A0成功意味着不能声称“empty state单独、确定性导致v3 hang”。
两行修复已在正式规模、同样pre-update保存条件和DSRL并发现场绕开lazy-init分支并通过；v3那一次间歇stall的
更细内部时序没有被唯一复现，不再为此增加猜测性barrier或更多小实验。
