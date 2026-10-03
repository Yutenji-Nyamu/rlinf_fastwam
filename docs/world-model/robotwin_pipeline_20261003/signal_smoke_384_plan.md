# L384 信号 smoke：最小合同与停止条件

> 历史方案，2026-10-03 23:15已被用户要求的近正式并行替代，未启动，不可执行下文旧N16/R1计划。当前采用四卡N64/G8/R8，L32/GB512后连续L384/GB2048，micro8/U2；详见[当前实施记录](execution_20261003.md)。以下仅保留预算推导历史。

2026-10-03。只读固定 RLinf `2151a08ee1bd75df1bef0d8190e594bd5c7f7977` 和当前 OpenDW 适配；没有 SSH、启动或正式训练。适用触发：N8/N16 的 L32 工程与资源 smoke 跑完，但全部 G8 被原生过滤，或者没有可证实的非零学习信号。

**下一步是单次 `signal_smoke`，L384、C32、R1、U2、GB=12N、micro1。** 保持同一个 N，不同时扩大并行；N 根据已完成资源档选择。384 是 12 个完整动作块，不是精确 400 动作，也不是原生成功率评估。

## 为什么 GB 必须是 12N

固定源 `EnvWorker` 与 HF rollout 都用 `max_steps_per_rollout_epoch // num_action_chunks` 得到动作块数。actor 将 `[chunk_steps, N, ...]` 展平为 `chunk_steps × N` 个训练样本，再按 global batch 切分；每个 global batch 结束才调用一次 `optimizer_step()`。因此单 runner iteration 的调度数为：

$$
\text{optimizer dispatches}=U\times\frac{N(L/C)}{GB}=2\times\frac{12N}{12N}=2.
$$

`gradient_accumulation=GB/(micro×actor_world_size)`，这里单 rank、micro1，所以分别为 96/192。过滤和提前结束只改变 mask，**不把填充样本从 batch 物理删除**，不会降低这两次预定调度。`runner.max_epochs=max_steps=1` 才是单次完整 GRPO iteration；`update_epoch=2` 不是两个 runner iteration。

| 合同/上限 | N8 | N16 |
|---|---:|---:|
| G8 组数 | 1 | 2 |
| 轨迹长度 / 动作块数 | 384 / 12 | 384 / 12 |
| 展平 chunk 样本 / GB | 96 / 96 | 192 / 192 |
| 每次优化累积 microbatch 数 | 96 | 192 |
| U2 合计 micro forward/backward 数 | 192 | 384 |
| optimizer 调度数 | 2 | 2 |
| WM 行推理上限 | 96 | 192 |
| 环境动作上限 | 3,072 | 6,144 |
| 未来图像上限（每行 8 张） | 768 | 1,536 |

服务仍逐行 B1 推理；这里 N 是逻辑环境数，没有增加 WM 同时处理的 batch。提前成功后，该环境不再调用 WM，实际 WM 行数可能较少；训练端仍保留 12N 格式与结束 mask。固定源默认 `rollout.collect_final_values=true`，因此还有一次原生末观察 policy query：最多 13 次 policy batch query，只有前 12 次动作进入 WM。草案保留这个行为。

模型/优化器体积、micro1 前反向大小不变；rollout 历史与缓存增加到最多 12 倍，CPU RSS、GPU 常驻缓存仍可能增加，不能声称显存不变。墙钟也不是严格 12 倍：初始化/保存只一次、actor 前反向与 WM 行数主要增加，优化器仍两次。

来源：[actor 训练](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/workers/actor/embodied_fsdp_actor_worker.py)、[runner](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/runners/embodied_runner.py)、[EnvWorker](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/workers/env/env_worker.py)、[HF rollout](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/workers/rollout/hf/huggingface_worker.py)。

## 最小配置与 owner 改动

相对于相同 N 的 L32 配置，训练预算只改三个值：`env.train.max_episode_steps=384`、`env.train.max_steps_per_rollout_epoch=384`、`actor.global_batch_size=12N`。另换新 run/namespace/output 身份。保持 C32/H50/M10、U2、micro1、G8/R1、奖励系数 1、相对奖励、0.9 成功阈值、原生组过滤 `[0.1,0.9]`、原模型起点、`resume_dir=None`、seed/reset、单卡 GPU4 与卸载屏障。保持保存第 1 轮、关闭原生评估。

两个长度字段要一起改：原生 loss 用 `loss_mask_sum/max_episode_steps` 修正提前结束轨迹的权重。L384 的轨迹若在第 k 块结束，比例为 `k/12`；按完整 12N batch 累积后保留原来的按轨迹归一方式。不应只改 rollout 长度而留下 `max_episode_steps=32`，也不额外手动缩放奖励/梯度。

已备独立草稿 [signal_smoke_mode.patch](../../../local_patches/opendw_smoke_20261003/drafts/signal_smoke_mode.patch)，**尚未应用**：

- generator 增加明确 `--mode signal_smoke`，继承固定 Control，写出上述合同与差异；默认 smoke 仍 L32。
- owner 增加第三模式 `signal_smoke`，只允许一个 N8/N16 trial、一个 runner iteration、L384/GB12N/micro1/U2。不得用 `mode=formal, runner_iterations=1` 冒充信号测试。
- 输入必须有相同 N 的 L32 成功退出与归还回执；新的 `signal_smoke_budget` 绑定该回执 SHA、trial key 与实测秒数。timeout 必须显式填写，草稿允许上限 `max(2700, 15×同N的L32实测秒数)`，同时受 owner 原 7 天绝对上限约束。15×只是 12 倍主要工作量加余量的保守上界，不是自动采用的默认时长；实际时长由资源读数决定。
- owner 结束仍走原来的精确 cleanup→release→RLT resume→首轮确认。最终回执写 `mode=signal_smoke`、`signal_audit_required=true`；该回执只证明运行/归还完成。
- 后续所有 formal plan 都要求 `learning_evidence`：其中 `status=effective_update_verified`，并且 `owner_final_sha256` 必须绑定所引用的 final.json。旧 L32 回执也不能凭 exit 0 绕过此要求。这个独立审计结论要依据下方原始读数填写，不能把退出码改名成学习通过。

草稿基于 generator SHA `0cd40d25233849f5fe77e10d6d512e2f3874d60eeb18a8fa3364160d20bcbf45`、owner SHA `36ddcb97a06f8c5aedbf6d2f9ba4594a7f5789996ef5ce547cb3125b294b50af`；patch SHA `9d234a3d8f3edd6974d748bac7b41b39bef8042726d4caec7ff94bc6dfd09f28`。后续若 owner 加入 graphics scope，应在新副本上按这几处语义重新合并，不覆盖正在使用的冻结 owner。

预算字段示意（下面为结构，数值须取实际回执）：

```json
{
  "mode": "signal_smoke",
  "smoke_evidence": "/data/chenyiteng/.../previous-owner/final.json",
  "signal_smoke_budget": {
    "source_key": "n8",
    "source_seconds": "替换为 result.json 的实测数值",
    "source_final_sha256": "替换为前述 final.json 的 SHA256",
    "timeout_seconds": "替换为根据实测选择的数值，且等于 trial timeout",
    "reason": "记录初始化、WM、actor、保存耗时与选定余量"
  }
}
```

## 有效学习与无信号停止

相对奖励在每个动作块求和后望远镜相消。累计回报仍是**每个环境最后一个实际执行块的末帧分数**；提前成功后取成功块末帧，未成功取第 12 块末帧。不是把十二个绝对成功分数相加，也不是对十二个 chunk 独立做 G8 过滤。

完整 rollout 后，对每个原始 G8 的上述回报求均值；组均值位于 `[0.1,0.9]` 且组内非零差异，才可能产生有效优势。有效 chunk 数必须从 `done` 与组过滤共同计算；提前成功时不再只能是 0/96/192，不能照搬 L32 的计数公式。

本轮已按授权对 service 做纯日志补充：`row_started`、`row_completed` 和样本 metadata 添加可选 `env_index/reset_id`。payload 原来已有这些字段；日志缺失或格式不匹配时记 null，推理与 reward 不变。这样在提前成功导致 alive batch 收缩后，仍能把后续行连回原始环境和 G8，不能把压缩后的 `row` 当全局环境编号。该修改单独部署/CPU 检查，不在上述 owner 草稿里。

通过“存在有效学习”的最低证据是：

1. 至少一个保留 G8、至少一个有效 chunk；记录组均值、组内标准差、有效样本数。
2. 有限的非零正/负优势；`advantage_mean≈0` 正常，`success_once=0` 不自动失败。
3. 有限、非零的 actor 梯度，配合 `policy_loss_abs`、ratio/clip/KL；不能只看有符号 policy_loss。
4. 完整 `global_step_1/actor` 检查点、可训权重有限；Adam 真实 step 应为 2，动量有限且有非零值。参数变化/step=2 单独不足以证明奖励学习，因为 AdamW weight decay 也能改参数。
5. 资源峰值、WM/actor 阶段切换、GPU 绑定与归还回执均正常。

若仍然全滤，记 `no_valid_group`；若组保留但同分，记 `zero_advantage`；若有优势却没有有效梯度，记 `no_effective_update`；数值/协议/资源异常单独记失败。**只完成这一轮并归还，不自动进入正式训练，不自动关闭过滤、改阈值或加训练轮数。** 草稿没有在 actor 收包处新增 zero-mask 跳过逻辑，仍保留原生两次优化器调度；无信号的停止点是这次受限测试结束。

当前已部署 formal owner 的 `smoke_evidence` 检查只核退出码和归还，不能充当学习判定；新草稿对所有正式运行统一要求独立学习回执。下一份正式 plan 必须先具备可追溯到本次配置/日志/checkpoint 的真实学习证据；不得仅凭本次 exit 0 发起正式任务。原生 RoboTwin 评估继续单独配置、单独预算、单独报告，WM 代理分数不替代真实成功率。
