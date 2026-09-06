# current RLinf π0 DVAC telemetry 实现流水（2026-08-22）

目标：在exact `7d07a421...`独立worktree语义重放default-off观测旁路；只采模型已有denoising chain，
不改action/RNG、trajectory schema、actor/critic、advantage或loss。真实episode P0/P1另行提交resolved packet。

## PI0-DVAC-IMP-001 — source/worktree preflight

状态：`COMPLETED`（2026-08-22 09:37:02 UTC，exit 0）。

- 目标：在任何源码写入前核对指定 worktree、branch、exact base、clean tree、个人仓 remote 和旧 telemetry source object。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_preflight_20260822.sh`
- SHA-256：`c4f0b5771960d3efecd145065bb1ed4a09e953389726f2b8d4c9141329b6245e`
- 远程执行：固定 host-key 的 `verified_password_ssh.py run --command-file ...`；密码仅注入当前交互进程。
- 预期：`HEAD=7d07a421...`、branch=`codex/sz-current-pi0-dvac-observe`、tracked/untracked 均为空，且 `61996e15...` 可解析。
- 结果：指定 worktree/branch/exact base 全部一致；`git status --short` 为空；`origin` 为 official，`personal` 为 `Yutenji-Nyamu/rlinf_fastwam`；旧 telemetry commit `61996e15...` 已在对象库；三个 current 接缝文件存在。
- 判断：可以开始 semantic replay；不直接 cherry-pick 旧提交，因为三个现有文件与 current API 均已演进。

## PI0-DVAC-IMP-002 — 旧 6 文件增量的三方可应用性检查

状态：`COMPLETED`（exit 0）。

- 目标：把 exact old base `6d0db56...`→telemetry `61996e15...` 的限定 6 文件 patch 对 current base 做 `git apply --3way --check`；只判断可合并性，不写源码。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_threeway_check_20260822.sh`
- SHA-256：`b92ca035b7d740ac865328e5b8cfbc6c3bffc8636168bfe1e57e4ac594e7431e`
- 预期：新增文件 clean addition；三个 existing files 若自动三方合并，也仍须逐接缝审查 current 行为，不能把“可应用”当作 runtime 已兼容。
- 结果：`huggingface_worker.py` 可 clean 三方应用；`openpi_action_model.py` 与 `env_worker.py` 有预期 current/old 双方修改冲突；三个新增文件走 direct-add 路径。check 后 worktree 仍 clean。
- 判断：应用限定 6 文件增量，再只对两个冲突文件做 current-aware semantic resolution；仍需审查自动合并的 HF worker。

## PI0-DVAC-IMP-003 — 应用 source-locked 6 文件增量

状态：`COMPLETED`（old increment已应用并完成current-aware resolution；后续验收见IMP-004/005）。

- 输入仅为 `git diff 6d0db56... 61996e15... -- <exact six paths>`；不包含 training-weighting、R-only、control trace 或其他历史文件。
- 预期中间态：三个新增文件 + HF worker应用，两个 existing files保留 unmerged markers，随后在本条之下记录精确 resolution。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_apply_old_increment_20260822.sh`
- SHA-256：`702eb3527b997cdfdf92b549495e0f4c2b11a8af25ef99e4a304109a7ecc718e`
- 应用结果：`git apply` rc=1且仅两个预期 unmerged paths；新增 YAML/writer/test 与自动合并 HF worker均已进入工作树，没有额外文件。
- 解冲突：
  - OpenPI 同时保留 current `rtc_context` 与 opt-in `return_dvac_telemetry`；普通 native sampler传 telemetry；RTC guidance + telemetry显式拒绝，避免把非目标 sampler伪装成同一chain。
  - Env 同时保留 current `SmoothInterveneController` 与 DVAC episode writer/config；其余自动三方结果不改。
- 本地解冲突文件先做 `py_compile`，exit 0；上传前 SHA-256：OpenPI `41407c5921d4c7be0fe17d27d4a68a445e389cb7bcd74e80cc623751594ef0db`，Env `2918fe82f6fccb93fa4a3cff4a080c26995ed7c3c31831884a1100925ceb3644`。
- current-aware 审查后另做三项窄调整：HF worker把 `rtc_enabled` 纳入 telemetry-on 禁用 mode；YAML把 base/compatibility locks 更新为 `7d07a421...`/`0008ae68...`；test补 current signature（RTC+default-off telemetry）与跨 batch query metadata merge。
- 三文件上传前 SHA-256：HF `31fb3846667b360be6cc6149d0bd841ae6a30b6a08e3844449da16eb5298b8d4`；YAML `ead027934a8f8725cf08aca41feb50d9133d3630915b03cd5449ab425c3cbde7`；test `24c778354316d54f32cf80700896d839314de863a38fc148f8f3b6e27e3d4905`。本地4个Python文件整体 `py_compile` exit 0、无冲突 marker。

## PI0-DVAC-IMP-004 — resolution 校验与限定 6 文件 stage

状态：`COMPLETED`（复跑 exit 0）。

- 目标：核上传 hash与关键 current/telemetry 双边接缝，确认无 marker，只 `git add` 精确 6 paths，随后跑 staged `diff --check` 与 name-set gate。
- 不执行 commit/push、不运行模型/GPU/仿真。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_resolve_and_stage_20260822.sh`
- 首次 SHA-256：`75be66270e6caabb5f0b9066b4e604b51fd7466a6582086b395d8854531b2e47`。
- 首次结果：5个上传文件 SHA 均 `OK`，随后一个只读文本 gate 因目标报错字符串在源码中分成两行而提前退出；尚未执行 `git add`。问题是检查写法，不是实现错误。
- 窄修复：将 grep 从跨行完整句收窄到实际单行 `telemetry is not supported`，其余命令不变，重新计算脚本 hash 后复跑。
- 复跑 SHA-256：`0368c99abe15be9154f974049f17891cdade3a23ddd573937d9d3449e1665f22`。
- 结果：无 unmerged/marker；staged `diff --check` 通过；精确6文件集合为3 modified + 3 added，`1143 insertions / 9 deletions`，没有其他路径。

## PI0-DVAC-IMP-005 — CPU-only focused pretests

状态：`COMPLETED`（第三次完整复跑 exit 0）。

- 检查：Ruff lint/format-check、5个Python文件 compile、focused telemetry pytest、default-off 与显式 on 两份 Hydra `--cfg job --resolve`。
- 配置解析只使用 physical 2–3 的 placement 字符串，不初始化 GPU/Ray/model/simulator；环境显式 `CUDA_VISIBLE_DEVICES=''`、`NVIDIA_VISIBLE_DEVICES=none`。
- 不创建 run/output，不执行真实 chunk parity 或 episode；后二者属于 P0 批准包。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_cpu_pretests_20260822.sh`
- SHA-256：`37a920a08fc9ec44d335632146636764f43be226ada8facd059410296529160b`
- 首次结果：Ruff在第一项立即发现新增 test 缺 RLinf copyright header（`CPY001`）；没有进入 format/compile/pytest/compose，未启动 GPU/Ray/仿真。
- 修复：仅给新增 test 加标准 Apache-2.0 header；随后上传、stage该 exact path并完整复跑同一脚本。
- 修复后 test SHA-256：`706806020bd1fe6d98d1c6489bd91e5d2ab7968fed2e7cb7761262a602fb91ce`；本地 `py_compile` exit 0。
- 精确 restage command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_stage_test_header_fix_20260822.sh`；SHA-256 `ba957f3c28a939032d44f35c69fc3eacfc9d131e9a1c3adf12b6c35b9c7025c4`。
- 第二次结果：Ruff lint通过；format-check指出5个改动Python文件需要项目标准机械格式化，尚未进入 compile/pytest/compose。
- 处理：只对上述 exact5 Python paths执行项目已有 `ruff format`并 restage；不改变YAML、不扩路径。
- format command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_format_exact_files_20260822.sh`；SHA-256 `6d13c9a9c392f2c8624f2f3bb6ee6d6fffc1881cd3ea8a29013ea249035ad499`。
- format结果：exact5 Python files已机械格式化并restage；限定6文件变为`1135 insertions / 9 deletions`。
- 最终结果：Ruff lint=`All checks passed`；Ruff format=`5 files already formatted`；`py_compile` exit 0；focused pytest=`4 passed`（11.16s）；telemetry off/on两份 Hydra compose均通过；全过程 `CUDA_VISIBLE_DEVICES=''`、未启动GPU/Ray/model/simulator。
- 非阻断输出：Torch JIT deprecation 与当前 Hydra `version_base/defaults _self_` warning；均来自现有runtime/官方config，不是本实现失败。

## PI0-DVAC-IMP-006 — 独立 staged-diff 审查

状态：`COMPLETED`；独立reviewer结论为无阻塞，主任务已明确放行发布。

- 已向主任务提交 current-aware model/HF/env/writer/config/test 接缝摘要与全部CPU验收结果。
- 主任务安排独立只读 diff reviewer；按其要求在结论返回前暂停 commit/push。
- 期间不改源码、不启动真实 chunk parity/P0。

## PI0-DVAC-IMP-007 — commit / non-force push / remote verify

状态：`COMPLETED`（commit/push/remote verify exit 0）。

- 目标分支：`personal/codex/sz-current-pi0-dvac-observe`。
- commit subject：`feat: add opt-in pi0 DVAC telemetry`。
- 脚本先锁branch/base、Git identity、精确6-path staged set与clean unstaged tree；commit后做无proxy、无force、120s有界push，再以`ls-remote`和upstream SHA双重核对。
- 脚本可在“commit成功但push网络失败”后安全重跑；不会amend/force/reset。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_commit_push_verify_20260822.sh`
- SHA-256：`b64ea2e58d69ce47ae1bd84ac4c24586339e2ce6f5764bc5ab90085ae17f1090`
- 首次结果：脚本在任何输出前退出；按命令顺序最可能是worktree未配置`user.name`/`user.email`，因此尚未执行commit/push。先以只读probe核HEAD/status、当前identity与旧personal telemetry commit author，再决定只对本次commit使用process-local identity，不持久改shared repo config。
- identity probe command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_commit_identity_probe_20260822.sh`；SHA-256 `7c7a160e8a7cae349892abef03216d6addf7271b2f194b475a8a9c6a1c6af133`。
- probe确认：HEAD仍为base、精确6文件仍stage，commit/push均未发生；shared repo没有local identity。旧personal telemetry commit `61996e15...` 的author/committer一致且属于用户既有Git历史。
- 窄修复：发布脚本只在当前进程继承该旧personal commit的author/committer，不写global/repo Git config；其余commit/push/verify gate不变。
- 修复后发布脚本 SHA-256：`e57b5f7356d9185dc690cdbe4bb4ae07bb1c9530a6952af1f40932491bf12bf4`。
- commit：`f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5`，subject `feat: add opt-in pi0 DVAC telemetry`，parent exact `7d07a421...`。
- remote：`personal/codex/sz-current-pi0-dvac-observe`；non-force new branch push成功。
- 验收：`ls-remote`=`f7cf0f60...`，upstream同SHA，worktree clean且ahead/behind均0。
- 全程未运行真实GPU/model/simulator；没有改Git global/repo identity或其他worktree。

## PI0-DVAC-IMP-008 — P0批准包的source/seed只读预计算

状态：`COMPLETED`（exit 0）。

- 目标：只读锁定发布commit、compatibility commit、seed-file SHA，并按current `partition_success_seeds(base_seed=0,total_processes=1,num_group=2)`预计算单env-worker P0两个fixed reset IDs。
- 显式禁用GPU；不加载模型、不建env、不运行仿真。
- command file：`local_scripts/remote_commands/shenzhen_pi0_dvac_p0_seed_source_probe_20260822.sh`
- SHA-256：`97ce53f371e82a1f3b635e7c3ece5540d7ffca7a743e80d76b556dd2973204cc`
- 结果：RLinf clean `f7cf0f60...`、RoboTwin `0008ae68...`；seed file SHA-256 `194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f`，`adjust_bottle`有150个official success seeds。
- 对单env-worker、`seed=0`、2 env/group1的current partition语义，首个固定block为reset IDs `100100052, 100100066`。未使用GPU/sim。

## PI0-DVAC-IMP-009 — 真实 P0 批准包草案（未批准、未执行）

### Source / runtime / output

- RLinf：`800baf80d6eab64169cf0e691eb04a681a093ee9`，branch `codex/sz-current-pi0-dvac-observe`，
  parent telemetry commit=`f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5`；启动时要求clean且remote/upstream同SHA。
- RoboTwin：`0008ae6800df9f75fc8de7098bacb01735fd8fd2`；π0 SFT revision `92684e50...`；norm SHA `649ed92b...`；seed SHA `194164f7...`。
- env：现有 `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`，不安装/更新依赖。
- candidate output：`/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p0-2ep-800baf80-v1`；必须在启动前不存在，不覆盖official eval或PPO产物。

### Gate A：real checkpoint 单query off/on parity

- physical GPU 2；同一 `adjust_bottle` fixed reset initial observation，模型只加载一次。
- 保存调用前CPU/CUDA RNG state；先`return_dvac_telemetry=false`，恢复完全相同RNG state，再`true`；比较env action、model action、chain/final endpoint及两次调用后RNG state。
- 预算：1次simulator reset、0个environment action slot、同一观测2次policy query、每次4 denoise steps；只写一个parity JSON/NPZ。
- 通过条件：actions bitwise相同、post-call RNG state相同；on trace为`x[1,5,50,14]`、`z[1,4,50,14]`且`z=x-t*v`。任何不等价即停止，不进入Gate B。
- 精确harness仍需在启动包中给出并CPU compile/review；本条不授权编写后直接运行。

### Gate B：P0 两episode telemetry eval

- physical GPU 2、单env/rollout worker、`total_num_envs=2`、`rollout_epoch=1`、`auto_reset=true`、`ignore_terminations=true`、`max_episode_steps=max_steps_per_rollout_epoch=200`、`C=50`。
- fixed reset IDs由current official partition预注册为`100100052,100100066`；不按outcome挑seed。
- 统计预算：2 episodes、最多400 action slots、最多8 policy queries、32 denoise model steps；0 train trajectory、0 optimizer update、0 checkpoint。P0不进入成功率主分析。
- exact Hydra主入口保持official：

```bash
/usr/bin/time -v timeout --signal=INT --kill-after=120s 1800s \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421/evaluations/eval_embodied_agent.py \
  --config-path /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421/evaluations/robotwin \
  --config-name robotwin_adjust_bottle_openpi_dvac_eval \
  'cluster.component_placement={env\,\ rollout:2}' \
  runner.logger.log_path=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p0-2ep-800baf80-v1 \
  rollout.model.model_path=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 \
  env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support \
  env.eval.total_num_envs=2 env.eval.rollout_epoch=1 \
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 \
  env.eval.use_fixed_reset_state_ids=true \
  rollout.dvac_telemetry.enabled=true \
  rollout.dvac_telemetry.run_id=pi0-adjust_bottle-p0-2ep-800baf80-v1 \
  rollout.dvac_telemetry.source_commit=800baf80d6eab64169cf0e691eb04a681a093ee9 \
  rollout.dvac_telemetry.seed_file_sha256=194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f \
  'rollout.dvac_telemetry.launch_command=verified_password_ssh command-file P0 v1'
```

- 启动前另以同一ARGS做`--cfg job --resolve`保存resolved YAML；沿已成功official eval环境设置`ROBOTWIN_PATH`、`ROBOT_PLATFORM=ALOHA`、`REPO_PATH`、`EMBODIED_PATH`、`PYTHONPATH`、`OPENPI_DATA_HOME`、`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`。
- 验收：driver自然exit0；2条episode rows、8条以内query rows且reset IDs精确；每query `x[5,50,14]`、`z[4,50,14]`、times=`[1,.75,.5,.25]`、final/env actions和三相机query PNG可join；manifest source locks正确；video存在。资源只记录不做经验阈值gate。
- 启动边界：先live刷新GPU2/host RAM/目标路径；若PPO仍处于高主存运行态，默认不并发。只允许timeout时终止本包owned process，不停止PPO或其他用户任务。

## PI0-DVAC-IMP-010 — real-checkpoint fixed-observation parity harness / packet

状态：`COMPLETED / INDEPENDENT REVIEW PASS / COMMIT+PUSH VERIFIED`；
只编写与CPU静态/聚焦验收，**未启动 GPU、Ray、model 或 simulator**。

- 目标：把 IMP-009 Gate A 从文字草案收敛为 source-locked 可审查 harness 和
  resolved packet；它在真正执行时只加载一次真实π0 checkpoint，从同一个
  `adjust_bottle` fixed reset initial observation 依次做 telemetry off/on 两次 query，并通过
  保存/恢复 CPU/CUDA RNG state 保护等价边界。
- 实施时操作边界：先核对 branch=`codex/sz-current-pi0-dvac-observe`、
  HEAD=`f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5`和clean tree；再实现最小工具。
  只跑 `py_compile`/静态 import 或 mock/focused CPU test；最终只 stage，不 commit/push，
  等独立 reviewer 审 exact diff。
- 预定执行资源：physical GPU 2；单进程；Gate A timeout `900s`，
  `kill-after=120s`；不执行 environment action，不进入 Gate B。
- preflight command file：
  `local_scripts/remote_commands/shenzhen_pi0_dvac_real_parity_preflight_20260822.sh`，
  SHA-256 `75136a94798d583f63f350cd66f78f4e46fa42e3814e6e0b2a88ac4ea21e0742`。
  现场确认 HEAD/upstream=`f7cf0f60...`、branch精确、tree clean，runtime/model/RoboTwin路径存在。
- 实现收敛为唯一新文件 `toolkits/probe_pi0_dvac_real_parity.py`（359行）：
  真实run只加载一次model，从一次真实fixed reset取obs，不step env；保存
  Python/NumPy/Torch CPU/all-visible-CUDA RNG，off后恢复pre再做on，比较env/model action、
  chain/logprob/value/final endpoint和post-RNG。on路径只在harness内包装现有
  `sample_mean_var_val`捕获已算`x/v`，不改返回或抽样；据此对真实chain验
  `x[1,5,50,14]`、`z[1,4,50,14]`和bitwise `z=x-t*v`。唯一产物目录含
  `parity.json`/`parity.npz`，并拒绝覆盖。
- CPU/stage command file：
  `local_scripts/remote_commands/shenzhen_pi0_dvac_real_parity_cpu_stage_20260822.sh`，
  SHA-256 `aade2c5aadbd980c4bf8b092db60fa9eca70dbbce189fc287a5ec25024bb7de2`。
  首次Ruff只报unused `os`，在stage前退出；只删该import后完整复跑：Ruff check/format、
  `py_compile`、内置CPU focused self-test均通过，`diff --check` 通过。当前精确只stage
  该1文件，`359 insertions`，并停在review点。最后自审另将formula验证的`timestep`
  显式cast回sampler dtype，与source的`x - v * t_i`运算顺序/dtype完全一致；随后
  完整复跑上述全部检查通过。
- 真实执行命令合同（**未执行**）：独立review通过并commit/push后，先将同一
  resolved config固化为`total_num_envs=1`且model path精确；再以
  `CUDA_VISIBLE_DEVICES=2 timeout --signal=INT --kill-after=120s 900s <venv>/bin/python
  toolkits/probe_pi0_dvac_real_parity.py run --resolved-config <resolved.yaml>
  --model-path <pinned-SFT> --expected-head 800baf80d6eab64169cf0e691eb04a681a093ee9 --reset-state-id 100100052
  --inference-seed 0 --output-dir <new-parity-dir>` 运行。其中expected head必须是review后唯一
  commit SHA；当前已固定为
  `800baf80d6eab64169cf0e691eb04a681a093ee9`，但真实Gate A仍未获批/未启动。
- 独立reviewer结论：`PASS`，无阻塞项。随后发布脚本
  `local_scripts/remote_commands/shenzhen_pi0_dvac_real_parity_commit_push_verify_20260822.sh`，
  SHA-256 `eb1d6aa49da40ad27dec0d78485f546135070d3f292f31ca2301682d1757840d`，先精确gate
  HEAD=`f7cf0f60...`、唯一staged文件`359/0`、unstaged=0，再提交
  `800baf80d6eab64169cf0e691eb04a681a093ee9`，subject
  `test: add real pi0 DVAC parity gate`，parent精确`f7cf0f60...`。
- 以无force普通push发布到`personal/codex/sz-current-pi0-dvac-observe`；local/remote/upstream
  均为`800baf80...`，ahead/behind=`0/0`，worktree clean。`FORCE_USED=0`、
  `GPU_RAY_MODEL_SIM_USED=0`。

## PI0-DVAC-IMP-011 — Gate A complete resolved packet（CPU-only closure）

状态：`READY_FOR_REVIEW_NOT_EXECUTED`；完整packet已落数据盘，真实Gate A未获批/未运行。

- packet generator：
  `local_scripts/remote_staging/pi0_dvac_real_parity/generate_pi0_real_parity_packet.py`，
  SHA-256 `f56b354ad9735003187d8956beaabb28e96cfe2f8127e5947c16d3229832a360`。
  它在创建目录前精确核对RLinf HEAD/branch/upstream/personal remote均为
  `800baf80d6eab64169cf0e691eb04a681a093ee9`、worktree clean、RoboTwin=`0008ae68...`、
  exact runtime/model存在，且packet/output目标都不存在；任一已存在即拒绝覆盖。
- official Hydra compose在`CUDA_VISIBLE_DEVICES=''` / `NVIDIA_VISIBLE_DEVICES=none`下通过同一
  `evaluations/eval_embodied_agent.py --cfg job --resolve`生成；锁定`total_num_envs=1`、
  fixed-reset=true、pinned model/RoboTwin assets和output path，telemetry writer保持false（Gate A由
  harness显式off/on）。只有官方现有Hydra migration / missing `_self_` warning，无compose错误。
- packet path：
  `/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-real-query-gate-a-800baf80-v1`；
  真实候选output：
  `/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-real-query-gate-a-800baf80-v1`，
  终态仍不存在。
- packet包含`resolved_config.yaml`、`compose_command.sh`、`launch_gate_a.sh`、`manifest.json`、
  `budget.json`、`STOP_CONDITIONS.md`和`SHA256SUMS`。关键SHA-256：
  - resolved：`f197ab27aa42501004d8eff0d5817caddbff7da675b02445732191bece04841c`；
  - launch：`a971a5dac532a7f1392fd7f13221b306dd5a421a35f726e78f97124841c18c4b`；
  - manifest：`51ca4ce213b4ddeccc83c321836e97c3a535f9024d989061dd051a2ce74af1da`；
  - budget：`8e9c9f8f7974525f9c806e2d2038a9d1741997569710cc7396ae84f9603b3504`；
  - stop：`f1c88771276fb486a9034345343476bcc81eedd4eff6d6d35be43c77e4fc025b`。
- 预注册budget：physical GPU2、1 simulator reset、2 policy queries、每query 4 denoise steps，
  总8 denoise steps、0 environment action slots、0 train trajectory/update/checkpoint；reset ID=`100100052`、
  inference seed=0、timeout=`900s`、kill-after=`120s`。launch只在实际执行时设
  `CUDA_VISIBLE_DEVICES=2`，本条没有运行launch。
- 只读验收command file：
  `local_scripts/remote_commands/shenzhen_pi0_dvac_real_parity_packet_verify_20260822.sh`，
  SHA-256 `90958352ee0797b491f266ee0a356e2bf695f4125a63cac2ede6cff0d97d0c22`。
  `sha256sum -c`全部OK，两条shell `bash -n`通过，resolved/manifest/budget semantic verify通过，
  source仍clean/remote一致，real output仍不存在，`GPU_RAY_MODEL_SIM_USED=0`。

## PI0-DVAC-RUN-001 — 21:27 CST：取消 Gate，直接启动 P1 fixed-16

- 用户明确要求去掉off/on Gate与2-episode机制检查，允许在工程检查无问题时直接做正式推理。PPO已停止、
  Ray/GPU资源释放后，按该授权启动首个固定16-episode数据块。
- command file：
  `local_scripts/remote_commands/shenzhen_pi0_dvac_launch_p1_16env_gpu2_20260822.sh`，3,222 bytes，
  SHA256=`bd4c4c1d7fc8160e243e399550e43ec316b96e9e9619a1fac384efaa02c76718`。
- source/runtime：clean `800baf80d6eab64169cf0e691eb04a681a093ee9`、RoboTwin `0008ae68...`、
  pinned π0 SFT `92684e50...`、既有 RLinf venv；不安装依赖、不改源码。
- resolved预算：physical GPU2；`total_num_envs=16`、rollout epoch1、fixed reset IDs、H=C50、M4、
  max episode200；正好16 episodes，最多3,200 action slots、64 policy queries、256 denoise steps；
  telemetry只旁路保存，不训练、更新或保存checkpoint。
- output：`/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1`；
  resolved SHA256=`f9fc8f33d4ac156ff417fffa8ca951fa60a2a261ff273d0fcda34b1dab9e5dcc`。
- 21:27 CST后台启动PID=`619674`；10秒后PID alive，local Ray已启动，只有既有Hydra migration/
  missing `_self_` warning，没有启动错误。hard timeout 2小时，只约束本轮owned process。

### 21:29–21:30 CST只读健康快照

- command files：
  `shenzhen_pi0_fastwam_live_health_20260822.sh` / `shenzhen_pi0_fastwam_log_health_20260822.sh`；
  SHA-256分别为`b525f7b4b1bffcaeb585a01ad7fb00f83a99beedd1b5bcb563665d33c24cc028` /
  `ff99b7000d08f262a497133f6e3c897f733f65cadc969b4689d2959e6c9ab7e0`；均exit0且只读。
- timeout PID `619674`与Python/Ray child均alive；driver完成resolved config与placement，π0 norm stats真实加载，
  已进入`Evaluating Rollout Epochs: 0/1`。fatal/traceback/OOM/illegal-instruction扫描为空。
- physical GPU2从`15,610 MiB`升至`23,029 MiB`，两个compute app分别为rollout/env worker；GPU3及4–7空闲。
  host当时约`35 GiB used / 1.9 TiB available`，没有与旧PPO的主存压力重叠。
- output当时只有7个启动/config/log文件，尚无telemetry query或episode终态产物；这是模型加载后的早期健康点，
  不冒充16 episodes完成。全程未发送signal。

### 21:38–21:39 CST自然完成与终态计数

- 只读command files：`shenzhen_pi0_fastwam_completion_refresh_20260822.sh` / 
  `shenzhen_pi0_fastwam_csv_counts_20260822.sh`；SHA-256分别为
  `e96e9248793aaeedb2f28615f4c586a1af752861a997ace283a1e1c630926e0e` /
  `6f5681ebce0ff7e115a9e92aedd9723eb4ea974a832082e268aa81edc9b115ea`；均exit0。
- PID`619674`已自然退出，`chenyiteng` Ray core count=`0`，GPU2无compute app；driver完整输出
  `Global Step 1/1`、16 trajectories、`success_once=success_at_end=0.75`，即fixed-16为`12/16`，
  elapsed=`4:56`、step time=`296.874s`，fatal扫描0。
- telemetry精确为16条episode rows、64条query rows、1个consolidated NPZ、192张三路query PNG、
  1个132,966-byte combined MP4和2个manifest。路径分别为
  `dvac_telemetry/episode_index_env_rank00.csv`、`query_index_rollout_rank00.csv`与
  `trace_rollout_rank00.npz`。全程没有stop/restart或产物修改。

## PI0-DVAC-P2-PLAN-001 — 23:00 CST：新official fixed64主集执行包

状态：`LOCAL_PACKET_COMPOSED_NOT_LAUNCHED`。

- 22:54 CST现场确认既有GRPO/Ray因控制面RPC失联已退出，8卡空闲；因此按15号统计计划利用当前单用户
  Ray空窗新跑一份fixed64主集，不把旧single-rank fixed16拼入主集，避免seed partition重合风险。
- 本地command file：
  `local_scripts/remote_commands/shenzhen_pi0_dvac_launch_fixed64_gpu0_3_20260822.sh`，3,279 bytes，
  SHA-256=`e7bdc9242b8dbd6c45f44eefa887390988557b4778cc2e2b57209b4b99a8399b`。它锁定telemetry
  source`800baf80...`、RoboTwin`0008ae68...`、π0 SFT`92684e50...`和既有venv，不安装或改源码。
- resolved草案：physical GPU0--3、placement`{env, rollout:0-3}`、64 fixed reset episodes、
  `adjust_bottle`、H=C50/M4/max episode200；上限12,800 action slots、256 queries、1,024 denoise steps；
  0训练/optimizer/checkpoint。output预定为
  `/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1`。
- 历史同机non-telemetry fixed64约5:58、约11GiB/卡、host used约113GiB；本轮含PNG/NPZ预算10--15分钟、
  预留2GiB。启动前由同一命令保存resolved YAML；only-owned hard timeout 1小时。成功率或DVAC形状不作为
  停止条件。尚未上传、remote `bash -n`或启动。

## PI0-DVAC-RUN-002 — 23:08 CST：official fixed64 telemetry 主集启动

状态：`COMPLETED`（唯一一次run自然exit；未stop/restart/overwrite）。

- 连接：日常账号`chenyiteng`，固定host-key的低层Paramiko密码route；本轮连接无认证或pre-auth异常，
  密码只注入当前进程。
- 先完整读取根规则、当前交接、专题SSOT、15号64-rollout计划与本账本，再核本地launcher exact
  SHA-256=`e7bdc9242b8dbd6c45f44eefa887390988557b4778cc2e2b57209b4b99a8399b`。
- 持久packet：
  `/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-fixed64-800baf80-v1`；
  本地prepare command file为
  `local_scripts/remote_commands/shenzhen_pi0_dvac_fixed64_packet_prepare_20260822.sh`，SHA-256
  `0022fccb8a5d222249a244e6e96855d68638df125cbe8502242276ffa710781a`。它只在packet原本不存在时创建，
  23:08 CST exit0。随后用SFTP把已审launcher普通上传到该packet，不覆盖任何run。
- remote preflight/resolve command file为
  `local_scripts/remote_commands/shenzhen_pi0_dvac_fixed64_preflight_resolve_20260822.sh`，本地SHA-256
  `9b051a780da76fe80a499250615465a45fe66713dae803d2a8297e8b5a77a2ae`。remote launcher hash和
  `bash -n`、RLinf clean `800baf80...`、RoboTwin `0008ae68...`、两份SFT shard、run output不存在、
  chenyiteng Ray不存在、physical GPU0--3没有compute process、`/data`至少2GiB可用均先通过。
- 同一exact Hydra overrides在CPU-only环境保存到packet `resolved.yaml`，SHA-256=
  `6c30dc3599acd70cb2ff71ecf10a60715ad6ebeb319d7ed14cbeb47d85798d51`。首次semantic assertion把
  official Hydra的placement误预期为两个dict key，实际resolved是`{'env, rollout': '0-3'}`；这只是
  本地检查表达错误，resolved其余关键字段经只读逐项核为`64 env / epoch1 / max200 / fixed reset /
  telemetry on / source800baf80 / pinned model / exact new output`。未改resolved、未绕过任何live predicate。
- 执行入口：packet内上传的
  `shenzhen_pi0_dvac_launch_fixed64_gpu0_3_20260822.sh`；通过command-file普通调用一次，没有重试。
  23:09 CST启动owned timeout PID=`792654`，run为
  `/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1`，runtime再次生成的
  `resolved.yaml` SHA与packet一致。10秒后Ray已启动，launcher marker
  `SZ_PI0_DVAC_FIXED64_LAUNCHED`出现。
- 15:10:28 UTC首个只读快照：PID alive、Raylet count1、fatal0；4个rollout与4个env worker均按
  hardware ranks0--3创建，仍处模型初始化；GPU0--3分别约`742/674/674/674 MiB`，host
  `MemAvailable=2,025,897,144 KiB`，尚无episode/query终态产物。没有发送signal。
- 中段少量只读快照：四rank真实加载pinned norm/model并进入official eval；观测到的GPU0--3最高
  分别为`25,336/24,915/25,015/24,915 MiB`。没有独立observer，因此这里只称snapshot-observed peak，
  不冒充连续采样峰值。同期最小host `MemAvailable=1,943,183,716 KiB`，即整机used proxy约
  `162.35 GiB`；fatal始终0。
- official rollout主体`167.99s`完成，随后writer与Ray自然收尾；最终metric为
  `64 trajectories / success_once=success_at_end=0.65625 / return=0.65625`，即`42 success / 22 failure`。
  RLinf表内elapsed=`04:50`、step time=`290.032s`；launch manifest为15:09:34 UTC，最后metric log为
  15:15:28 UTC，故launch到最后结果行约`5m54s`，15:16:28 UTC只读现场已观察到进程完全退出。
- 终态artifact：64 episode rows、256 query rows、4个rank NPZ、768张三相机PNG、4个rank tiled MP4；
  MP4合计`555,148 bytes`，run总计`40,760,738 bytes`。每个rank NPZ均为
  `x_chain[64,5,50,14]`、`z_endpoint[64,4,50,14]`、`final_model_action/env_action[64,50,14]`、
  `robot_state[64,14]`、`timesteps[4]`，逐数组`finite=true`；timesteps来自resolved official M4链。
- 清理终态：owned PID已退出，chenyiteng `raylet/gcs_server=0`，physical GPU0--3无compute process且
  memory归零，host `MemAvailable=2,091,661,116 KiB`；fatal/traceback/OOM/illegal-instruction命中0。
  全程未发送signal、未删除或改写source payload、未启动第二次run。
