# RoboTwin native ACT 数据、训练、评估流水账

本账只记录 standalone RoboTwin/XPolicyLab ACT pipeline 的真实服务器操作。凭据不落盘；未执行的
启动包不冒充服务器结果。用户已在 2026-08-21 明确批准 render、单条 collect、1-epoch training
smoke、official checkpoint load 与单 episode eval；本账后半部记录了该闭环的真实结果。

## ACT-PIP-001 — official clean-50 ACT preprocess

- 时间：2026-08-21 18:42–18:43 CST。
- 账号：`chenyiteng`。
- 初始目录：`/data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/ACT`。
- command file：`local_scripts/remote_commands/shenzhen_preprocess_adjust_bottle_act_20260821.sh`。
- command file SHA256：`032C8EBBD97C20398B96DEC7EACE09D14D3B345751A28C357C81EC7EEE35E769`。
- 关键 preflight：XPolicyLab HEAD 为 `c07a09614dd44cc4a67483bcb9a82e7439d99926`；source 恰好 50
  episodes；总帧数 7,188；三相机未压缩 payload 下界 18.509 GiB；目标 output/key 均不存在；
  `/data` 约 3.2 TiB 可用。
- 完整 official entrypoint：

```bash
cd /data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/ACT
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act
timeout --signal=INT --kill-after=60s 7200s \
  bash process_data.sh demo_clean adjust_bottle aloha_agilex joint
```

- 退出码：0；50/50 episodes 均完成，没有 retry 或窄修复。
- output：
  `/data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/ACT/processed_data/demo_clean/adjust_bottle/aloha_agilex-joint`；
  50 个 processed HDF5，约 19 GiB。
- `TASK_CONFIGS.json` 新增精确 key `demo_clean-adjust_bottle-aloha_agilex-joint`，映射到上述 dataset
  root，`num_episodes=50`、`episode_len=5000`、camera order 为
  `cam_head, cam_right_wrist, cam_left_wrist`。
- 验收：episode 0 与 49 存在；output file count、key/value、disk usage 均通过；`/data` 仍约 3.2 TiB
  可用。该阶段无 GPU，不加载模型，不启动 simulator。

## ACT-PIP-002 — 三阶段功能闭环启动包收敛

- 时间：2026-08-21 18:44–18:55 CST。
- 服务器动作：无；本条只冻结下一次真实执行前的 resolved packet 和授权边界。
- 用户建议被收敛为：1 episode 自采并取证后精确删除；clean-50 上 1 epoch/3-update training smoke；
  锁定 official HF ACT checkpoint 的 offline debug + 1 episode sim eval/video。
- 不采用“启动 official 6000 epochs 后强杀”：current `train.sh` 仅自然结束才保存 `policy_last`，提前
  kill 不能稳定闭合 checkpoint contract。
- training command file：
  `local_scripts/remote_commands/shenzhen_act_train_smoke_1epoch_20260821.sh`；SHA256
  `056CF085237B2F160DC93C5358EF063CCF667028228720341CE690A4589492EB`；这是当时未执行的 hook draft，
  只通过固定 host-key Paramiko 把正文流入远端 `bash -n` stdin，退出 0；没有写远端文件，也没有执行
  脚本正文。用户随后要求减少防御工程，该 draft 被 official direct CLI 版取代；真实执行见
  ACT-PIP-007，不能把此 SHA 当最终运行命令。
- packet：[`../03_NATIVE_ACT_EXECUTION_PACKET.md`](../03_NATIVE_ACT_EXECUTION_PACKET.md)。它明确列出
  source/revision、参数、exact paths、GPU/CPU/RAM、timeouts、20 分钟无进展监督语义、success evidence、
  official pickle trust boundary 和唯一删除目标。
- 当前停点：等待用户明确集中批准。未启动 render、collect、training smoke、checkpoint load 或 eval；
  未创建/删除临时 collect config/data root，也未停止任何服务器进程。

### ACT-PIP-002A — config/CLI source 只读闭合

- command file：`local_scripts/remote_commands/shenzhen_inspect_act_smoke_config_sources_20260821.sh`。
- 首版 SHA256 `16C7F91E415F01332A8791CC8D68C0FD5342EE82B34658D4260B5F66C69888D2`：成功打印
  `demo_clean.yml` 全文与 SHA，但在 `eval_policy.sh multitask --help` 处退出 1；原因是 Paramiko
  non-login shell 没有 Python，而非 scheduler 源码失败。没有远端写入。
- 窄修复：同一 command file 在 help 前 source conda 并激活 `RoboTwin`；SHA256
  `C1B63274A38A62D2BC6C177E38DAEC781E6388A3AD9245F668414399BB4752A2`，退出 0，确认 CLI 接受
  `--expert-check/--no-expert-check`、absolute `--ckpt-name`、`--output-dir`、`--stream-output`、`--dry-run`。
- 最终扩展只读 probe 同时打印 `all_tasks.yml`；SHA256
  `EEBFC2D3B80DB81C599C2703F1E32BEA9A35083F6468E11BB6636017FD9D12AA`，退出 0。
- official source hashes：`demo_clean.yml = 3acfcd508078a51e3a48fd7e2f6e61c1b9a6ced0cc44dedd149b55114248e2ab`；
  `all_tasks.yml = 948931440aef34ba856c67a957aabfd75b798363bc6577af92353b602922507f`。
- Windows 已用 `apply_patch` 生成待批准后精确上传的两个 config；当前尚未传到服务器：
  - `local_scripts/remote_configs/sz_collect_smoke_1ep_20260821.yml`，SHA256
    `A00159E4FDC7D1FA2927A76B5EE9662A09D71F4DB938E6077D167F8A21F9BFFB`；与 official collection
    config 只有 `episode_num: 50 -> 1` 的语义变化。
  - `local_scripts/remote_configs/sz_adjust_bottle_act_1ep.yml`，SHA256
    `783A69199E287419C084EE004BD077665823DBD6FFA3A7B30EFD630D0F6E6BC9`；GPU0、1 job/GPU、1 worker、
    local server、唯一 task `adjust_bottle`。

## ACT-PIP-003 — 授权后 live preflight

- 时间：2026-08-21 19:32 CST；账号：`chenyiteng`；只读。
- command file：`local_scripts/remote_commands/shenzhen_act_authorized_live_preflight_20260821.sh`；
  SHA256 `67BC3936A29F5D982B5BEE4C36DB762F766B4D580C14695B95DFA46A6DBE4F1E`；退出 0。
- 8×H100 全空闲，目标相关 owned process 为空。`/` 296 GiB、约 237 GiB 可用；`/home` 2.3 TiB、
  约 2.3 TiB 可用；`/data` 3.5 TiB、约 3.2 TiB 可用。
- source/assets/raw data/processed data/model/results 均位于 `/data/chenyiteng`；Conda 与 cache 位于
  `/home/chenyiteng`。没有把大资产或 checkpoint 写入根分区。
- Mihomo active/enabled，监听 `127.0.0.1:7890`；新 login shell 的 proxy env 生效。GitHub 与 HF
  小请求分别约 0.35 s、0.17 s；10 MB proxy 测试约 0.82 s，即约 12.2 MB/s。实际 HF checkpoint
  下载此前约 2.9 MB/s，说明端到端速度随上游而变，不能把 10 MB 测试当持续下载保证。

## ACT-PIP-004 — Gate A render 与首次 collect 失败

### Render

- command file：`local_scripts/remote_commands/shenzhen_act_gate_a_render_20260821.sh`；SHA256
  `68A18FA7724193DF07E93CD556624E06564654BD6EE898788788F82058C784EB`；退出 0。
- official `scripts/test_render.py` 在 GPU0 输出 `Render Well`；日志：
  `/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/01_test_render.log`。

### 首次 collect

- 上传 config：`env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml`，SHA256
  `A00159E4FDC7D1FA2927A76B5EE9662A09D71F4DB938E6077D167F8A21F9BFFB`；相对 official
  `demo_clean.yml` 只把 `episode_num` 从 50 改为 1。
- command file：`local_scripts/remote_commands/shenzhen_act_gate_a_collect_20260821.sh`；SHA256
  `FB2CCF36F87283984AA1A1D2FB286EAC3FA15097BB8B84D3C83A0EC77F13453F`。完整入口：

```bash
cd /data/chenyiteng/projects/robotwin-native/RoboTwin
CUDA_VISIBLE_DEVICES=0 bash collect_data.sh \
  adjust_bottle sz_collect_smoke_1ep_20260821 0
```

- 内部 render 通过，但所有候选 seed 在 `[Start Seed and Pre Motion Data Collection]` 同点报
  `CUDA error: an illegal instruction was encountered`。collector 捕获异常后继续试 seed，故不是 shell
  退出码即可表达的失败。
- 精确 owned timeout PID 经
  `shenzhen_stop_owned_collection_after_cuda_error_20260821.sh`（SHA256
  `05A9C61216BD0EDEAF6FAFEFBB77D8F91643B9B9F11C5E600F18C7AEE7E05C05`）TERM 停止；没有误停其他
  用户进程。失败日志保留为 `$RUN/02_collect_1ep.log`，空的 partial root 与 config 暂时保留取证。

## ACT-PIP-005 — CuRobo/H100 根因与单一窄修复

- `shenzhen_collect_cuda_illegal_instruction_probe_20260821.sh`（SHA256
  `027DD6F16E9E7B9FD9605F2B94B443BED97BB75F26747316132CEAF30242B351`）确认运行栈为 H100 sm90、
  torch 2.4.1+cu121、CuRobo v0.7.8、Warp 1.12.0；render 仍正常。
- 最小复现 `shenzhen_curobo_lbfgs_sm90_repro_20260821.sh`（SHA256
  `A969C21FE52788B5BB53D4245EEF0E0B4019C2A1592206A5E9AE532394971328`）给出完整栈：
  `LBFGSOpt._get_step_direction -> LBFGScu.apply -> lbfgs_step_cu.forward -> CUDA 715`。这排除了
  SAPIEN render、数据写盘与 ACT。
- config object probe（SHA256
  `1A0ADB2B5B91CB0517D82061002DB015A8962BF23C9B33D0149460BF1057381D`）找到 5 个已构造的
  IK/trajopt/finetune LBFGS optimizer，均为 `use_cuda_kernel=True`。
- 不改 source 的最小验证脚本 `shenzhen_curobo_unfused_lbfgs_warmup_20260821.sh`，SHA256
  `BFFE43C1202498EE828036BB6CE487080A0D3794F1F2F281CE0BB96DDD8CD748`：只把上述 optimizer 的
  `use_cuda_kernel` 置 False；同一 MotionGen warmup 在 40.33 s 内退出 0 并输出
  `UNFUSED_LBFGS_WARMUP_OK`。该 fallback 仍是 CuRobo 的 LBFGS GPU 路径，只旁路已失败的 fused
  step-direction kernel。
- 采用单一 server-local compatibility patch：
  `local_scripts/remote_patches/robotwin_h100_cuda121_unfused_lbfgs.patch`，SHA256
  `655D520E7A2208A9910279EE04A84C8B719836B32EE00992BC1A274377645091`。它只修改
  `envs/robot/planner.py`：当 GPU capability >=9 且 PyTorch CUDA runtime <12.6 时，在两个
  `MotionGen` 构造前关闭上述 fused LBFGS；其他 GPU/runtime 不变。
- apply command SHA256
  `2B0FF452F41150CC45F35F7D6A7B9C30AFC52914D99F6720EAF1371A3C1CED84`；先确认目标 clean、patch
  hash、`git apply --check`，再 apply；`py_compile` 与 `git diff --check` 均通过。第一次 patch 文本
  因 hunk line count 写错在 `git apply --check` 阶段退出，未修改 source；修正后才真实 apply。
- 长期若要大量 native collection，可另建 torch2.6/cu126 sibling env 并重编 CuRobo；本轮不原地
  替换 official RoboTwin pin，避免扩大环境破坏面。

## ACT-PIP-006 — Gate A collect 成功、验收与精确清理

- retry command：`shenzhen_act_gate_a_collect_retry_h100_fix_20260821.sh`；SHA256
  `6E4EB4D2DEF6F36955F0CB4C2485F1148869E9156C8B1AC05872D91C81D80999`；复用首次失败留下的空目录，
  未覆盖任何既有 episode。
- official collect 结果：seed 0，1/1 pre-motion simulation success、0 failed tries；保存 141 个
  action/state/image rows，生成 142-frame MP4；exit 0。
- verify command：`shenzhen_act_gate_a_verify_collection_20260821.sh`；SHA256
  `06FF6FE37696CCEF633720F20E1FBC916255F8DD73EDCAB5DF28160537246C35`；退出 0。核心产物：
  - HDF5 9,187,836 B，SHA256 `5e6158b6f9c628f1683406391d0bc4faf74c54175c929f6840b8119b8981be80`；
  - instruction JSON 15,000 B，SHA256 `628326d9fc3554ad677b75158fda76ba730b2d5ec34695e9e47380b1571bb223`；
  - MP4 104,719 B，SHA256 `97cf2caef4d6fb2962b8e869a55b951b8dd29e437b49f220687451d034cae6fa`，
    4.734 s；
  - seed file 内容对应 seed 0；HDF5 action/state 均 141 rows，三腕/头相机 payload 可读。
- manifest 保留为 `$RUN/03_collect_manifest.txt`。按用户要求，验收后执行 exact cleanup command
  `shenzhen_act_gate_a_cleanup_exact_smoke_20260821.sh`，SHA256
  `354C02A267073C1519B6A580C7049ED8B4D59FA750E93007FCEA89126A815BCA`：realpath 精确等值后只删除
  `/data/.../RoboTwin/data/sz_collect_smoke_1ep_20260821` 与对应临时 config；两者现均不存在，日志、
  patch 和 manifest 保留。

## ACT-PIP-007 — Gate B official ACT 1-epoch training smoke

- command file：`local_scripts/remote_commands/shenzhen_act_train_smoke_1epoch_20260821.sh`；最终
  SHA256 `69BBA215D7C34B81561649C44B15F9E3527211316DE6CF9219CDF3F2117CC401`。它已取代 ACT-PIP-002
  中未执行的 hook 版：直接调用 official `imitate_episodes.py`，不 monkeypatch、不增加自造 sentinel。
- 与 official `train.sh` 保持相同模型/objective 参数：ACT、KL 10、chunk 50、hidden 512、FFN 3200、
  batch 16、lr 1e-5、seed 0；只把预算设为 `num_epochs=1, save_freq=1`，输出隔离到
  `$RUN/06_act_train_smoke_1epoch_not_for_eval`。
- source clean-50 为 40 train / 10 val；由 batch16 得 1 个 val batch 与 3 个 train batches。首次按
  official torchvision 路径下载 ResNet18 44.7 MiB 到 `/home/chenyiteng/.cache/torch`，实际约 6.5 MB/s。
- official tqdm 1/1 约 2.60 s；命令退出 0。产物：`dataset_stats.pkl` 10,609 B、
  `policy_epoch_1_seed_0.ckpt` 335,910,986 B、`policy_last.ckpt` 335,907,442 B。该 checkpoint 明确只作
  training mechanism smoke，不用于成功率或 simulator eval。

## ACT-PIP-008 — Gate C official HF checkpoint inference 与 simulator eval

### Offline model/action loop

- command file：`shenzhen_act_hf_offline_debug_20260821.sh`；SHA256
  `1584054AF3FFE7E2FA958C5DB687D217126C6D93FEA59C8FC34C0D2D905F999C`。
- absolute checkpoint leaf：
  `/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50`；仅加载此前锁定、
  校验的 `policy_last.ckpt` 与 `dataset_stats.pkl`。
- official `eval.sh` 在 `EVAL_ENV_TYPE=debug` 下完成 10 episodes × 20 action steps，日志出现
  `[MAIN] eval finished`，exit 0；证明权重、stats、14D action、server/client transport 与 inference 闭合。

### Real single-episode simulator eval

- eval config 上传到 `$RUN/configs/sz_adjust_bottle_act_1ep.yml`；SHA256
  `783A69199E287419C084EE004BD077665823DBD6FFA3A7B30EFD630D0F6E6BC9`；GPU0、local server、1 worker、
  `adjust_bottle`、1 episode。
- command file：`shenzhen_act_hf_real_eval_1ep_20260821.sh`；SHA256
  `8367236B9308BCB5D2B79FF74FA4D7ECA2C54B7F62AD96E2B474B4E9535F03B3`。先用同参数 `--dry-run`
  解析为唯一 job，再运行 official `scripts/eval_policy.sh multitask`；保留 bare `--expert-check`。
- expert check 判定 candidate seed 100000 不稳定并跳过；实际策略 seed 为 100001。ACT 在 step 147/400
  成功完成任务；job return code 0，duration 133.498 s，scheduler summary 为 success=1/failed=0。
- official result：`1/1 = 100%`。这只说明一条工程闭环且这条碰巧成功，不作为稳定成功率估计。
- verify command `shenzhen_act_hf_eval_verify_20260821.sh`；SHA256
  `109F8947D5B1A2C86C21283B76C755AF71F84FD326C68DDF28F7F18A50424FD9`。服务器视频：H.264、
  320×240、10 fps、148 frames、14.8 s、172,523 B，SHA256
  `b9ee562adfbf9eff3c23af8b7ca6dff72bcfbc3fb0edadef5f333fa27c229340`；`_result.txt` 为 1.0。
- 视频已通过 pinned-host SFTP 下载到本地
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4`；本地 size/hash
  与服务器精确相同。下载前 C: 约 41.6 GB 可用。

## ACT-PIP-009 — postflight：资源、网络、配额与 source 状态

- 时间：2026-08-21 20:20–20:27 CST；账号：`chenyiteng`。
- quota probe 只读 `/etc/mihomo/config.yaml` 中 root-only provider URL；输出严格限制为 HTTP status 与
  `upload/download/total/expire` 数字，不输出 URL、token、query 或节点。probe SHA256
  `27A9F0B6D91F7B193D555853248012E8CDB2D9FF2202379AA1D81262C6B4FDB5`；sudo command SHA256
  `9A0053C17275B32B1D1CE570CC883328A9C3BDB7A1E665214145D34BFADF71C2`。sudo 密码与 SSH 密码仅由
  `verified_password_ssh_sudo.py`（SHA256
  `1CB6BDC68B8916985FE6405CB5CE05EA891D4E347D4178233813C299664C1DEE`）在当前进程内持有并写入远端
  stdin；没有命令行、文件或环境残留，结束时 `sudo -k`。
- live subscription：HTTP 200；upload 2,017,440,489 B、download 46,570,978,431 B、used
  48,588,418,920 B=`45.251 GiB`；total 100 GiB、remaining 58,785,763,480 B=`54.749 GiB`；expiry
  `2026-08-23 14:01:48 Asia/Shanghai`。代理没有用完。
- 第一版 postflight 没有 source login proxy profile，因 Paramiko non-login shell 而让 HF 走 direct，
  得到 TLS reset；这不是配额耗尽。direct/proxy 对照脚本 SHA256
  `D4A7D1EB7B7D151A8434BCB28A0651CD01A327E8629C9FD0AC5197AFB88D63ED`：同一 pinned model API direct
  exit35，显式 `127.0.0.1:7890` 为 HTTP 200。最终 postflight 显式 source
  `/etc/profile.d/mihomo-proxy.sh` 后 GitHub/HF 均 HTTP 200。
- final postflight command：`shenzhen_act_postflight_20260821.sh`；SHA256
  `E31DF2DF28E00E35AB22DD896E4CB0965BC8F2844A78B286E4F1025DD3CB3415`；退出 0。
- 20:27 CST：`/` 296 GiB、47 GiB used、237 GiB available；`/home` 2.3 TiB、51 GiB used；`/data`
  3.5 TiB、104 GiB used、3.2 TiB available。`/home/chenyiteng` 20 GiB，`/data/chenyiteng` 37 GiB；
  本轮 run root 641 MiB，其中 training output 641 MiB，eval leaf 180 KiB。大文件仍全在 `/data` 或
  `/home`，根分区未被挤占。
- 8 张 H100 均 0 MiB/0%；无 owned collect/train/eval/policy server 进程。
- source 预期状态：top-level 记录 XPolicyLab gitlink 从 parent pin `c371...` 到 official installer选择的
  `c07a...`；XPolicyLab 自身 clean；`envs/robot/planner.py` 保留本轮20行 H100/CUDA12.1 compatibility
  diff。临时 collect config/data 已删，未留下额外 untracked source 文件。
