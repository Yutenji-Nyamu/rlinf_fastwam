# guorenjie LeRobot / π0 / LIBERO / CLP 只读审计（2026-08-26）

## 1. 结论先行

- 现场是正常的机器人学习研究任务，没有发现恶意程序、越权、跨用户写入或系统配置修改迹象。
- `guorenjie` 是普通用户：仅自身组与 `labdata`，无 `sudo`；训练进程 `CapEff=0`、`TracerPid=0`，home 权限为 `700`。
- 2026-08-26 09:42 CST 的四个 GPU worker 实际属于 **两条两卡 DDP 实验**：
  - GPU 0/1：未剪枝 π0 baseline；
  - GPU 2/3：π0 `3M-v2-equal` 六层剪枝。
- 项目研究的是 CLP 风格的 π0 固定层剪枝：先加载完整 `lerobot/pi0_base`，再对 VLM 与 Action Expert 同时删除同一组六层，使两边均由 18 层变为 12 层，然后在 LIBERO 上全量微调与评估。
- 当前 full-finetune 对照尚未评估，不能得出新结论；已有 VLM-LoRA 400-episode 结果显示 `3M-pure=65.25%`、baseline=`63.75%`，但 paper mask 两次为 `56.50%/60.25%`，说明单次差异仍含明显运行/seed波动。
- 主要运维风险不是安全，而是可复现性与空间：LeRobot 源码处于 detached HEAD 且有未提交改动；项目已约 `341 GiB`，每条完整 30k run 约 `104 GiB`。`/home` 仍有约 `1.7 TiB` 可用，暂不逼近容量上限。

## 2. 当前实验身份

### 2.1 GPU 0/1：baseline

- `CUDA_VISIBLE_DEVICES=0,1`，`WORLD_SIZE=2`，rank 0/1 各占一张卡。
- 模型：`lerobot/pi0_base`。
- 数据：`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。
- 无 `clp_remove_indices`，即未剪枝对照。
- full fine-tune：`train_expert_only=false`、`freeze_vision_encoder=false`。
- relative action；`chunk_size=50`，`n_action_steps=10`。
- 每 rank batch 16，梯度累积 1，因此全局 batch 32。
- BF16、gradient checkpointing、compile；LR `2.5e-5`，warmup 1,000，30,000 steps。
- seed 1000；每 5,000 steps 保存；W&B 与 Hub push 均关闭。
- 已有完整 `global step 5000` checkpoint；09:42 CST 两个 rank 仍约 100% CPU、GPU 0/1 约 52.2 GiB/卡且高利用率。

### 2.2 GPU 2/3：3M-v2-equal

- `CUDA_VISIBLE_DEVICES=2,3`，`WORLD_SIZE=2`，rank 0/1 各占一张卡。
- 与 baseline 的训练预算相同；唯一算法性差异是：
  - `clp_remove_indices=[1,2,3,6,8,9]`。
- 09:42 CST 尚未出现首个 5,000-step checkpoint；进程仍存活，GPU 2/3 约 43.9/44.1 GiB且利用率约 95--98%。

### 2.3 已完成的三条 30k run

| run | 删除层 | 并行 | 已确认终点 | 大小 |
|---|---:|---:|---:|---:|
| `3m_pure_s30k` | `[1,2,3,6,8,10]` | 2×GPU，global batch 32 | 30,000 | 104 GiB |
| `3m_v2_default_s30k` | `[1,2,4,6,8,10]` | 1×GPU，batch 32 | 30,000 | 104 GiB |
| `paper_s30k` | `[1,2,4,6,8,9]` | 1×GPU，batch 32 | 30,000 | 104 GiB |

三条目录均含 5k、10k、15k、20k、25k、30k 的完整训练步状态。当前 baseline/equal 是在补严格的未剪枝/另一 mask 对照，而不是四条互不相关任务。

## 3. 项目、代码与官方关系

- 工作目录：`/home/guorenjie/research/smolvla-libero-clp`。
- LeRobot 源码根：`.../lerobot@c903b114a90e703b3f7d0c46cb38727c328c55ff`，detached HEAD。
- tracked diff：8 files，`+331/-9`；另有4个未跟踪源码/测试文件。
- 生产实现的核心很窄：`clp_prune_pi0.py` 在完整 checkpoint 加载后，对 π0 的 PaliGemma VLM 与 Gemma action expert 使用相同 keep list，18→12；随后重编号 `layer_idx`，避免 KV cache 出现洞。
- 代码注释明确写了“不是 official CLP reproduction”。它借鉴的是官方 CLP 的 CKA-guided structural pruning 思路，但加入了本项目的 3M/ΔH/因果 ablation 诊断与多组候选 mask。
- 官方参考仓：`jibby2803/CLP_VLA@53b9da7244ce4f8d23285fc57a1b2c8dbea7d5af`；官方论文为 arXiv `2606.20246`。

## 4. 数据是什么

精确服务器 cache 元数据：

- LeRobotDataset v3.0，robot=`panda`；
- 1,693 episodes，273,465 frames，40 tasks，10 Hz；
- 单一 train split `0:1693`；
- 两路图像 `image/image2`、机器人 state、7D action、task index等；
- 本地训练用 rename map 把两路图像映射到 π0 所需的 base/wrist camera key。

它对应 LIBERO 的 Spatial/Object/Goal/Long 四个标准10任务套件。已有评估均为每任务10次、总400 episodes。

## 5. 已有什么结果，能说什么

此前 VLM-LoRA 40k checkpoint 的400-episode结果：

| 方案 | success |
|---|---:|
| `3M-pure` | 65.25% |
| baseline | 63.75% |
| `3M-v2-default` | 63.75% |
| `3M-v2-equal` | 63.50% |
| paper mask，第一次 | 56.50% |
| paper mask，rerun | 60.25% |

因此当前最克制的结论是：

1. 这些固定剪枝模型确实能训练、加载和完成完整400回合评估；
2. `3M-pure` 在该单次旧协议上比 baseline 高1.5个百分点；
3. 不能据此宣称某个 mask 稳定更优，尤其 paper mask 两次相差3.75个百分点；
4. 当前正在跑的 full-finetune 30k 系列尚未做评估，必须等相同 checkpoint、相同 fixed episodes/seed 后再比较。

## 6. 安全与资源审计

### 6.1 未发现的风险

- 无 sudo、无 effective Linux capabilities、无进程 tracing。
- 当前训练命令不含删除、kill、系统目录写入、Hub push或W&B外发。
- socket 主要是两组 DDP rank 在本机公网地址上的自连接；唯一外部 HTTPS 地址解析为 `hf-mirror.com`，且只是 `CLOSE-WAIT`，符合模型/数据缓存访问。
- 定向扫描自定义 launch/source 只找到两处 `rm -rf "${out}"`，均在未运行的 VRAM smoke 脚本里；`out` 被限定在项目内的 `outputs/pi0_vram_gate/<tag>` 或 `outputs/pi0_vlm_lora_smoke/<tag>`。

### 6.2 真实需要注意的点

- Git dirty + detached HEAD：不影响服务器安全，但若不及时 commit/记录 precise diff，后续复现实验会困难。
- checkpoint 密度：每条30k run保存6次完整模型+optimizer，约104 GiB。当前两条补充实验若都按相同策略完成，预计还会增加约180--210 GiB。
- 09:39 CST 整机 `/home`=`674 GiB/2.3 TiB`、`/data`=`892 GiB/3.5 TiB`、根分区=`56 GiB/296 GiB`；短期无爆盘风险。
- 当前八卡全部被两组任务占用：guorenjie 使用0--3，chenyiteng GRPO-DVAC使用4--7，卡号无重叠。

## 7. 本轮只读操作账

- 使用固定 host-key Paramiko 登录 `toom`，密码只进入当前进程；通过 `sudo` 只读访问私有 home。
- 读取：身份/组/sudo、进程命令与环境中的 rank/GPU映射、GPU/RAM/磁盘、Git HEAD/status/diff stat、launch参数、dataset metadata、checkpoint step、既有 eval summary、CLP核心代码与定向安全关键字。
- 未执行：信号、kill/stop、终端attach、文件写入/改名/删除、安装、下载、Git checkout/commit/push、读取SSH key/token等凭据。
- 精确命令文件均在 `local_scripts/_readonly_guorenjie_*_20260826.sh` 与 `_readonly_sz_training_and_guorenjie_audit_20260826.sh`；本轮没有服务器写操作。

