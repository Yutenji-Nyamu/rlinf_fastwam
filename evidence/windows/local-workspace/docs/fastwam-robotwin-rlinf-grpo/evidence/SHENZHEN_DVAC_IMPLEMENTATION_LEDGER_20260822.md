# SZ-H100 official Fast-WAM DVAC telemetry 实现流水（2026-08-22）

目标：在exact `7faa711...`独立worktree为official deterministic `infer_action`增加default-off观测旁路；
保存已有`x/v/timestep/delta/x_next`并离线派生endpoint，不增加forward/RNG、不改action queue。
真实episode P0/P1另行提交resolved packet。

Git边界：Fast-WAM是与`rlinf_fastwam`不同的Git历史；先在本地独立worktree完成commit。推送前使用独立的
Fast-WAM fork/repository与独立deploy key，不把无关历史塞进RLinf仓库。

## FW-DVAC-IMP-001 — source/worktree preflight

状态：`PASS`。

- 已完整读取工作区 `PROJECT_CONTEXT.md`、`HANDOFF.md`、深圳主计划、13号双模型 DVAC
  观测计划、Fast-WAM standalone 完成计划/短复现说明及本专题接口契约。
- 冻结实现边界：base=`7faa71108368fbb3b6885649f112af607427a2d4`；只改独立 worktree；默认
  `enabled=false`；开启时只复制 official Euler loop 已经计算的 `x/v/timestep/delta/x_next`，不增加
  forward/RNG draw，不改变 action、queue、vendor RoboTwin 或训练路径。
- 下一条操作是只读核对 worktree/branch/dirty tree、目标函数真实签名和现有测试布局；在核对前不写源码。

第一次只读 preflight：

- 时间：`2026-08-22T09:37:41Z`（17:37:41 CST）；账号/机器：`chenyiteng@admin`，UID 1003。
- command file：
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_source_preflight_20260822.sh`；SHA256
  `8f4e5f811e128d9e8754028236e92d331099c7c6ab6dbffa7216088336004b96`。
- 结果：exit `20`。初次将它解释为 worktree 缺失；主协调线程随后指出该 worktree 已在 17:33 CST
  建立。复核发现是本地 preflight 的判断错误：Git linked worktree 的 `.git` 是**文件**，脚本误用了
  `test -d`。
- 单一修复：只把存在性判断改为 `test -e`；没有改服务器路径、worktree 或 branch，也没有在 canonical
  checkout 开发。修复后复跑同一 source/dirty/symbol 检查。

第二次只读 preflight：

- 时间：`2026-08-22T09:38:53Z`；修复版 command SHA256
  `9cc12516abc1efd90afad1fe293d0e1619fc549a90fac2ce2d188cc109abeeaf`。
- 已确认 worktree、branch、HEAD、merge-base 和 clean status 全部正确；三个目标文件 base SHA256 分别为
  `154c339e...`、`57a34b52...`、`b96b579a...`；只有 official HTTPS `origin`，尚无个人 Fast-WAM remote。
- 检查在 symbol inventory 阶段 exit `127`：服务器没有 `rg`。这是只读脚本工具假设，不是 source/runtime
  问题；按工作区规则只加 `command -v rg` 后的 `grep -nE` fallback，再复跑，不安装工具。

第三次只读 preflight（`PASS`）：

- 时间：`2026-08-22T09:39:42Z`；最终 command SHA256
  `4d07e329b887ee705ced3a27907aca171e627a40f896aa02ad4d59f08d26fd9d`；exit 0，marker
  `FASTWAM_DVAC_SOURCE_PREFLIGHT_OK`。
- worktree=`/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711`；
  branch=`codex/sz-fastwam-dvac-observe`；HEAD/merge-base 均为 exact `7faa711...`；tracked/untracked
  status 均为空。
- canonical detached checkout 与该 linked worktree 是仅有两棵 Fast-WAM worktree；remote 只有 official
  HTTPS `origin=https://github.com/yuantianyuan01/FastWAM.git`。
- 目标基线：`fastwam.py` 54,376 bytes / SHA256 `154c339e...ac09`；`deploy_policy.py` 15,379
  bytes / `57a34b52...56ac`；`sim_robotwin.yaml` 975 bytes / `b96b579a...056f`。仓库没有既有 Python
  test tree；新增 focused test 需保持自包含、避免引入新的 test dependency。

## FW-DVAC-IMP-002 — exact-base guard 与源码上传

状态：`PASS`（上传后服务器focused tests亦已完成，见下文）。

- 本地 staging 仅由 source-locked worktree 的 SFTP 副本产生，所有源码编辑均用 `apply_patch`；本地
  `py_compile` 通过，`tests/test_fastwam_dvac_telemetry.py` 为 `3 tests / OK`。
- 上传前 guard：
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_upload_guard_20260822.sh`，SHA256
  `4bcdf0582b6ccc5508bcb0541aea708af0398b053a13e7dacd4fb6b2ee32aee0`；exit 0，marker
  `FASTWAM_DVAC_UPLOAD_GUARD_OK`。它重验 exact HEAD/branch/clean、三个已知文件 SHA256、另外两个
  tracked blob 与 HEAD 一致，并确认两个新增文件不存在；只创建空 `tests/` 目标目录。
- 随后经固定 host-key 的 SFTP 向精确 worktree 上传 7 个目标文件，7 次传输均 exit 0：
  `fastwam.py`、`deploy_policy.py`、`dvac_telemetry.py`、`eval_robotwin_single.py`、
  `deploy_policy.yml`、`sim_robotwin.yaml`、`tests/test_fastwam_dvac_telemetry.py`。
- 没有写 canonical checkout、vendor RoboTwin、环境、模型、结果目录或 PPO；没有运行 GPU/model/simulator。

首次 server CPU pretest：

- 时间：`2026-08-22T09:57:55Z`；command
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_cpu_pretests_20260822.sh`，SHA256
  `ae0f646d876b36e2cb98e63ba3c1d908282528bb43ea69d070df385501017719`。
- `git diff --check` 前的源码清单符合预期；tracked diff stat 为 5 个已有文件 `+132/-3`，新增 helper
  未计入 `git diff --stat`。随后在激活环境前 exit，原因是文档记录的 prefix
  `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128` 当前没有 `bin/activate`。
- 这不是代码失败；没有安装/修复环境，也没有改用系统 Python。下一步只读核对该 prefix 真实入口以及
  ignored test 文件状态，再用已存在的 official runtime 复跑。

runtime entry / ignore 复核：

- `2026-08-22T09:58:50Z` 首轮和 `09:59:39Z` 展开复核；最终 probe SHA256
  `308b91ab6d1e7ae44d43f6d59f295bfc39d49da03199f2c227c0309084339381`，exit 0。
- 该路径是完整 Conda prefix，存在可执行
  `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -> python3.10`，但确实没有
  `bin/activate`；修复为直接调用 exact prefix Python，不 source、不重建环境。
- repo `.gitignore:14` 明确忽略 `tests/`，所以 test 文件存在但不会出现在普通 status；不改全局 ignore，
  precommit 只对这一个精确 test 用 `git add -f`。

server CPU pretests（`PASS`）：

- 修正版 command SHA256
  `db85558bd1bbc17641fa50a7e38a09b6d7bc488badb68482f4b6d3dbf170ba84`。
- `2026-08-22T10:01:33Z` 首次完整通过；补入“失败 episode 在下一 reset 只写一行”和“不覆盖已有
  output”检查后，于 `10:03:15Z` 再次 exit 0，marker `FASTWAM_DVAC_CPU_PRETESTS_OK`。
- exact env Python `3.10.20`；5个 runtime Python + test `py_compile` 通过；focused unittest 为
  `4 tests / OK`；default-off OmegaConf 与真实 `FastWAM.infer_action` import/signature 检查通过，marker
  `FASTWAM_DVAC_DEFAULT_OFF_IMPORT_OK`。
- 测试没有 instantiate/load model、没有 CUDA tensor、GPU/simulator/episode、依赖安装或结果采集。

## FW-DVAC-IMP-003 — staged diff 与提交前合同复核

状态：`PASS`；独立审查无阻塞，提交证据见 `FW-DVAC-IMP-004`。

- command：`local_scripts/remote_commands/shenzhen_fastwam_dvac_stage_review_20260822.sh`；SHA256
  `0c5b79502ed61d539d09bd0f53903b53aac2665bcd08e15bc522af5987fbf4bb`；
  `2026-08-22T10:04:21Z` exit 0，marker `FASTWAM_DVAC_STAGE_REVIEW_OK`。
- 精确 7-file staged diff：`669 insertions / 3 deletions`。分解为 config `+7`、父 evaluator forwarding
  `+15`、policy/writer接线 `+80/-2`、子YAML `+3`、新writer `+330`、Euler capture `+27/-1`、
  focused test `+207`。test 因 repo `tests/` ignore 规则仅对该精确文件使用 `git add -f`。
- `git diff --cached --check` 通过。合同 grep 已确认：
  default 参数为 false；trace 只在 true 分支进入返回字典；x/v/t/delta/x_next 全部 detach→CPU→FP32→clone；
  policy 每个 query 写盘后清空 pending payload，不保留跨query GPU引用；`take_action` 后读真实
  `task_env.eval_success`；失败由下一次正常 reset 或进程退出完成且 unit test 保证唯一episode行；已有
  output直接 `FileExistsError`；telemetry+compile 错误信息明确。
- 当时 index 只含上述7文件；HEAD仍为base `7faa711...`。按主协调要求先停在独立审查点；
  独立审查通过后才执行下节 local commit。

审查前补齐可复现输入合同：

- 自查发现 raw query 若只保存三相机/state而不保存实际 `instruction`，将无法把同一 query 离线重放为
  exact Fast-WAM prompt。只给 `queries.csv` 增加 `instruction` 字段，并由现有 `_fill_action_queue`
  参数直接传入；不读取/修改模型、动作或环境。
- 3个受影响文件上传后，`2026-08-22T10:10:26Z` 用同一 exact Python 再跑4 tests、compile/import，
  全部通过；`10:10:56Z` 重新stage并复核。最终仍为精确7文件，累计 `675 insertions / 3 deletions`；
  `git diff --cached --check` 和 stage marker再次通过。前述 `669/3` 是补字段前中间值，最终以本条为准。

## FW-DVAC-IMP-004 — 独立审查与 local commit

状态：`PASS`；**已 local commit、未 push；无 GPU/model/simulator run**。

- 独立审查结论：无阻塞；default-off返回合同、Euler capture、CPU detach、writer/queue/query与
  success/reset join、父/子Hydra转发、compile拒绝和不覆盖目录语义均通过审阅。
- 首次 commit command 在 `2026-08-22T10:18:24Z` 只输出时间后停止；原因是 Fast-WAM repo、worktree
  和用户global均没有 `user.name/user.email`，脚本中的只读 `git config --get` 返回非零。该次没有commit，
  7-file staged index、base和branch全部保持原样。只读 identity probe
  `shenzhen_fastwam_dvac_commit_identity_probe_20260822.sh`（SHA256 `a92b6851...`）再次确认这一点；
  上游HEAD author只作为来源事实，未被冒用。
- 经主协调明确给出个人仓既有公开身份后，最终 command
  `shenzhen_fastwam_dvac_local_commit_20260822.sh`（SHA256 `2a07479c...`）仅对本次
  `git commit` 使用 process-local `-c user.name/-c user.email`；没有写 global/repo Git config。
- `2026-08-22T10:19:52Z` local commit成功：
  commit=`fc652fb49cd32350eca15734b5c7124c0b8c2c02`；
  parent=`7faa71108368fbb3b6885649f112af607427a2d4`；
  tree=`51ff4c2c695501a63833bca2e4303f56430ba9cd`；精确7文件、`675 insertions / 3 deletions`。
  提交后 `git status --short --branch` 只有branch行，tracked/untracked clean；commit tree等于提交前
  staged tree。没有配置remote或push。

## FW-DVAC-P0-DRAFT — 真实收集前的只读路径核对

状态：`DRAFT ONLY`；未准备/启动真实 run。

- 只读 command：`local_scripts/remote_commands/shenzhen_fastwam_dvac_p0_layout_probe_20260822.sh`；
  SHA256 `307a9aa9cb6898b3166eba22bc743f4b5c37519fc2a566526fe00c4e6bfbdc5e`；
  `2026-08-22T10:08:03Z` exit 0，marker `FASTWAM_DVAC_P0_LAYOUT_PROBE_OK`。
- 新 worktree 有完整 tracked vendored RoboTwin code，但不含 standalone 准备阶段的 untracked
  `assets`、`task_config` 或 policy symlink；canonical official tree保留16-GiB assets、精确 task_config和
  指向canonical policy的symlink。模型12,041,813,092 bytes、stats88,715 bytes和exact env Python均存在。
- 因此 P0 不应复制第二份16-GiB assets，也不应临时改 canonical policy symlink。待批准的最窄准备是仅在
  **Fast-WAM telemetry worktree** 的 vendor 下建立两个只读来源 symlink（assets/task_config→canonical
  已锁内容）；official evaluator会在同一worktree创建/验证第三个 policy symlink→instrumented policy。
  这些准备属于真实P0 packet，当前尚未执行。
- 候选输出
  `/data/chenyiteng/results/dvac-observation/fastwam-adjust-bottle-p0-{off,on}` 均确认不存在；它们只是
  layout probe 的占位名，不代表已经决定跑两套完整 evaluator。

### P0 source budget

- 只读 command `shenzhen_fastwam_dvac_p0_budget_source_probe_20260822.sh`（SHA256 `31fd3d80...`）于
  `2026-08-22T10:17:59Z` exit 0，marker `FASTWAM_DVAC_P0_BUDGET_SOURCE_PROBE_OK`。
- source truth：`adjust_bottle` 每episode hard limit=`400` action slots；官方配置
  `seed=42 / mixed_precision=bf16 / eval_num_inference_steps=10`。
- 最终P0候选为一个**telemetry-on、2 episodes**的official single-evaluator run；不训练、不更新、不保存
  checkpoint、不进统计。`H=32 / C=24 / M=10 / D=14`，所以最坏上限为800 action slots、
  `2×ceil(400/24)=34` policy queries、340个action denoising steps；成功早停时实际值更小。
- 计划保留第11节要求的独立前置门：先用exact checkpoint对一个固定真实query做off/on action与local
  generator state parity；该门的helper和命令尚未在本轮实现/批准，因此以下episode command仍是
  **批准包草案，不可直接启动**。CPU synthetic parity已经4 tests通过，但不冒充真实chunk parity。

### P0 resolved command draft

唯一候选run id在真正批准/启动时冻结一次，例如：
`fastwam-adjust-bottle-p0-on-20260822_<HHMMSS>`。准备动作只复用canonical已经锁定的16-GiB assets和
task_config，不复制资产、不改canonical policy link：

```bash
WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
VRT="$WT/third_party/RoboTwin"
test ! -e "$VRT/assets" && test ! -L "$VRT/assets"
test ! -e "$VRT/task_config" && test ! -L "$VRT/task_config"
ln -s "$CANON/third_party/RoboTwin/assets" "$VRT/assets"
ln -s "$CANON/third_party/RoboTwin/task_config" "$VRT/task_config"
```

official evaluator会在同一worktree内创建/验证
`third_party/RoboTwin/policy/fastwam_policy -> $WT/experiments/robotwin/fastwam_policy`；不得把它指回
canonical policy。真实运行命令草案：

```bash
WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
RUN_ID=fastwam-adjust-bottle-p0-on-20260822_<HHMMSS>
TELEMETRY_DIR=/data/chenyiteng/results/dvac-observation/$RUN_ID

cd "$WT"
export PATH="$ENV/bin:$PATH"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

timeout --signal=TERM --kill-after=60s 3600 \
  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=2 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.action_horizon=null \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.replan_steps=24 \
  EVALUATION.rand_device=cpu \
  EVALUATION.tiled=false \
  EVALUATION.skip_get_obs_within_replan=false \
  EVALUATION.dvac_telemetry.enabled=true \
  EVALUATION.dvac_telemetry.output_dir="$TELEMETRY_DIR" \
  EVALUATION.dvac_telemetry.run_id="$RUN_ID" \
  EVALUATION.output_dir="$RUN_ID" \
  mixed_precision=bf16 seed=42 gpu_id=3
```

启动脚本必须先用`--cfg job --resolve`保存并核对同一组overrides，并确认：HEAD=`c63dc9b5...`、worktree
clean、exact checkpoint/stats/env存在、两个output都不存在、physical GPU 3现场空闲、RAM/disk/PPO进程
现场快照允许共存。`gpu_id=3`是物理卡号；parent会自行设置child的`CUDA_VISIBLE_DEVICES`。

### P0 resources, outputs and stop/accept contract

- 资源：1×H100 physical 3。相同official模型此前1 episode约134秒、peak GPU 30,274 MiB；本次
  `skip_get_obs_within_replan=false`会增加逐action观测开销，因此2 episodes保守估5–10分钟、hard timeout
  60分钟。host RAM以启动前现场值为准；不因PPO的Ray进程存在就停，也绝不停止/修改4–7卡PPO或他人进程。
- 空间：raw float trace本身每query约几十KiB；最多34份NPZ和三相机lossless query images，加2个official
  video/log/config，预计远低于1 GiB。telemetry落`/data/chenyiteng/results/dvac-observation/$RUN_ID`；
  official result落worktree的`evaluate_results/robotwin/robotwin_uncond_3cam_384/$RUN_ID`，不写checkpoint。
- 启动前停止：source/config/model/stats/env任一不符；目标目录已存在；GPU3有compute process；现场RAM/disk
  不足。运行中只停止本次owned进程：traceback、CUDA OOM/illegal instruction、official child非零，或60分钟
  timeout；保留失败证据，不覆盖同一run id。
- P0通过：real-chunk parity门先PASS；episode command exit 0；`run_manifest.json`、`queries.csv`、
  `episodes.csv`、每query NPZ/三相机图、official resolved config/log/result和2个episode视频齐全；正好2条唯一
  episode row；trace shape/schedule、executed slots、success和video path join闭合。success可以是true或false，
  不按outcome补挑seed。P0只证明机制/schema/行为旁路，不报告成功率结论。

## FW-DVAC-GIT-DRAFT — 独立 Fast-WAM fork/remote

状态：`WAITING FOR USER FORK/KEY REGISTRATION`；第二把key已生成，不阻塞local commit与P0草案，但fork/网页登记阻塞push。

- 该worktree唯一remote仍是official HTTPS
  `origin=https://github.com/yuantianyuan01/FastWAM.git`。Fast-WAM与`rlinf_fastwam`是不同Git历史，不能把
  commits `fc652fb...` / `c63dc9b5...` push进后者。
- 用户先在GitHub创建/确认个人fork（建议`Yutenji-Nyamu/FastWAM`）。服务器已经为这个repo生成第二把
  repo-specific Ed25519 deploy key；只把**公钥**交给用户，在新Fast-WAM repo的
  Settings → Deploy keys添加并勾选write。当前截图中的`SZ-H100 rlinf_fastwam` key只授权原RLinf repo，
  不应当成跨repo凭据复用；私钥不离开服务器。
- 用户完成登记后，才配置repo-local `personal-fastwam` SSH remote和repo-local `core.sshCommand`，验证
  `ssh -T`/read fetch，再push `codex/sz-fastwam-dvac-observe`。deploy key对单仓fetch/push已经够用；它不提供
  GitHub API或跨repo权限。若还要网页创建PR，由用户网页操作即可，不需要扩大到账号级SSH key/token。
- 第二把key已生成在`/home/chenyiteng/.ssh/github_fastwam_deploy_ed25519`，private/public mode为600/644，
  fingerprint=`SHA256:AluK6VvoK6VcCUVhlILEh6F77cimXdywhQkLOUt6BVE`；private key未输出。公钥与逐命令
  证据见RLinf专题
  [`evidence/11_GIT_WORKTREE_AND_IMPLEMENTATION_LEDGER_20260822.md`](../../rlinf-shenzhen-pi0-ppo-rlt/evidence/11_GIT_WORKTREE_AND_IMPLEMENTATION_LEDGER_20260822.md)。
  当前没有改Fast-WAM remote/config、没有push，因此不存在把不相关Fast-WAM历史带入`rlinf_fastwam`的风险。

## FW-DVAC-PARITY-001 — real fixed-query parity harness（staged review）

状态：`COMPLETED / FOCUSED REREVIEW PASS / LOCAL COMMIT / NOT PUSHED / NOT RUN`；
已local commit、未push，未加载模型、未启动CUDA或RoboTwin。

- `2026-08-22` preflight用
  `shenzhen_fastwam_real_parity_preflight_20260822.sh`（SHA256
  `adedba4fe4b482a6adf7d6dd12cb85fd21c2850f3ff748028d0885fc0529327d`）确认HEAD精确为
  `fc652fb49cd32350eca15734b5c7124c0b8c2c02`、branch正确、worktree clean，official env/checkpoint/stats
  及canonical assets/task_config均存在。
- exact-base upload guard为
  `shenzhen_fastwam_real_parity_upload_guard_20260822.sh`（SHA256
  `b41ac70296234c507da418cbcf19bed8b2d91462c30f98e4e402e5b50f4dad41`）；只上传3个目标文件。
- staged diff恰为3文件、`+384/-1`：
  `FastWAM.infer_action`增加默认`None`的显式`action_generator`注入口（原official `seed`路径不变，且禁止
  seed/generator同时传）；新增单用途real-query harness；新增2个CPU focused tests。
- harness只加载一次official模型，按official evaluator顺序取得同一真实query：`setup_demo(seed) →
  play_once/expert check → 同seed setup_demo → instruction → get_obs/state`。off后保存local generator及
  Python/NumPy/Torch CPU/全部可见CUDA RNG，恢复调用前状态再on；比较normalized/env action和所有post RNG，
  并验证trace shape、resolved schedule、`x_next=x_chain[1:]`、Euler step、final action和
  `z=x-(t/1000)v`。这条Gate有 **0次policy `take_action`**，但在固定query前按official路径做
  **1次expert feasibility rollout**（`play_once`）用于确认accepted reset seed；因而不再记为
  “0个env action”。
- exact env、强制`CUDA_VISIBLE_DEVICES=`的CPU检查脚本
  `shenzhen_fastwam_real_parity_cpu_stage_20260822.sh`（SHA256
  `fc8043b3d5c7fef720ed6a131be8510222bd52f7f9160e6850f1a4adb357713f`）完成3文件`py_compile`、CLI
  `--help`静态入口和`2 tests / OK`，随后精确stage并通过`git diff --cached --check`。

待独立review通过且执行前packet获批后，唯一候选命令为（本条只记录，尚未执行）：

```bash
WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
OUT=/data/chenyiteng/results/dvac-observation/fastwam-adjust-bottle-real-query-parity-v1
cd "$WT"
export CUDA_VISIBLE_DEVICES=3
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
timeout --signal=TERM --kill-after=120s 900 \
  "$ENV/bin/python" experiments/robotwin/fastwam_real_query_parity.py \
  --checkpoint /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt \
  --dataset-stats /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json \
  --robotwin-root "$WT/third_party/RoboTwin" \
  --reset-seed 4300001 \
  --output-dir "$OUT"
```

执行前仍需按前节packet建立/核对worktree内只读assets/task_config symlink、确认OUT不存在及physical GPU3、
RAM/disk现场允许；产物为`fixed_query.npz`、`parity_payload.npz`和`parity_report.json`。该命令未获批、未运行。

### Independent-review blocker 窄修（2026-08-22）

- reviewer指出official expert check会把起始seed `4300000`过滤为accepted seed `4300001`；
  原harness默认`4300000`会在不进入off/on query时就被feasibility gate拒绝。
- 唯一代码修复：`experiments/robotwin/fastwam_real_query_parity.py`的
  `--reset-seed` default从`4300000`改为official accepted `4300001`；保留且未更改
  `setup_demo → play_once/expert check → same-seed setup_demo → fixed query`。上述candidate命令也
  显式锁定`--reset-seed 4300001`。
- 修复前guard确认HEAD=`fc652fb...`、3个既有staged files且无unstaged diff，
  `play_once()`仍唯一存在；只上传上述单行替换。
  preflight/guard command SHA-256分别为
  `12abb67f43969df0dc7e28eda796e94d7bd93ff13574c5038244dea5965341d3` /
  `7a50f0157eb73eac2c0d660ad5ae97630fc5c775577f80a2267bebe1e3f4a07c`。
- 原CPU check/stage脚本`shenzhen_fastwam_real_parity_cpu_stage_20260822.sh`完整复跑通过：
  3文件`py_compile`、CLI `--help`、`2 tests / OK`、`git diff --cached --check`；marker
  `FASTWAM_REAL_PARITY_CPU_STAGE_OK`。最终仍精确3个staged files、`384 insertions / 1 deletion`，
  无unstaged diff，当时未commit/push，GPU/model/simulator=0。
  终态verify command SHA-256为
  `06e79de12f1eb2316a371d9230296ce9156a6e8193e721ff7a4439eacbd200e0`，输出
  `DEFAULT_RESET_SEED=4300001`、`OFFICIAL_EXPERT_PLAY_ONCE_COUNT=1`和`UNSTAGED_DIFF=0`。

### Focused rereview 与 local commit

- 独立reviewer对seed窄修后exact staged diff复审结论：`PASS`，remaining blocker=`0`。
- local commit command file：
  `local_scripts/remote_commands/shenzhen_fastwam_real_parity_local_commit_verify_20260822.sh`，
  SHA-256 `1a67219407c763e7c59ba3af883ffe64fdd5948821058dfe8b7597eaf942e58a`。
  脚本先精确gate HEAD=`fc652fb49cd32350eca15734b5c7124c0b8c2c02`、exact3 staged files、
  总`384/1`、unstaged=0和`diff --check`，再仅用process-local既有Git identity提交。
- local commit=`c63dc9b5384d6637a93cc862dbe2815d0332801d`，subject
  `test: add real Fast-WAM DVAC parity gate`，parent精确`fc652fb49...`；提交后worktree clean。
  没有配置Fast-WAM personal remote、没有push，`PUSH_USED=0`、
  `GPU_RAY_MODEL_SIM_USED=0`。后续仍等用户完成独立fork/deploy-key登记。

## FW-DVAC-RUN-001 — 21:27 CST：取消 Gate，直接启动 P1 sequential-16

- 用户明确要求去掉real-query parity Gate与2-episode P0，允许在工程检查无问题时直接做正式推理。
  PPO停止并释放主存/GPU后，按official single evaluator语义启动首个16-episode数据块。
- command file：
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_launch_p1_16ep_gpu3_20260822.sh`，3,418 bytes，
  SHA256=`91ed57307e821929b770539768d1d73cfed8100e6fe9fd1c13cdd72a4464545c`。
- source/runtime：clean local HEAD=`c63dc9b5384d6637a93cc862dbe2815d0332801d`、official release
  checkpoint/stats与既有Fast-WAM venv；只建立worktree到canonical assets/task_config的必要symlink，
  不复制16-GiB资产、不改模型或依赖。
- resolved预算：physical GPU3；official B1 evaluator顺序16 episodes；H32/C24/M10、episode hard limit400，
  最多6,400 action slots、272 policy queries、2,720 denoise steps；telemetry只旁路保存，不训练或保存checkpoint。
- telemetry output：
  `/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v1`；official video/result
  位于同worktree `evaluate_results/.../fastwam-adjust_bottle-p1-16ep-c63dc9b5-v1`；resolved SHA256=
  `f2371e2483eb32748f0ffc5d92cdfd0865c147181c1dd6b028162e30e7295eb1`。
- 21:27 CST后台启动PID=`619300`；10秒后PID alive，真实SAPIEN render已输出`Render Well`并进入official
  evaluator。现有Vulkan ICD warning与此前成功1/1运行一致，不是新错误。hard timeout 3小时，只约束
  本轮owned process。

### 21:29–21:30 CST终态只读快照：首次启动失败于output目录合同

- command files：
  `shenzhen_pi0_fastwam_live_health_20260822.sh` / `shenzhen_pi0_fastwam_log_health_20260822.sh`；
  SHA-256分别为`b525f7b4b1bffcaeb585a01ad7fb00f83a99beedd1b5bcb563665d33c24cc028` /
  `ff99b7000d08f262a497133f6e3c897f733f65cadc969b4689d2959e6c9ab7e0`；均exit0且只读。
- PID`619300`已经退出，GPU3回到4MiB且无compute app。driver为3,770 bytes；official child log一份
  2,093 bytes；没有query/episode telemetry或video产物。
- 精确首错不是模型、CUDA或SAPIEN：launcher先在telemetry output目录写`resolved.yaml`、manifest、PID和
  driver.log，而`FastWAMDvacTelemetryWriter`按合同拒绝任何已存在output目录，抛
  `FileExistsError: DVAC telemetry output already exists`；official child rc1，parent随后抛RuntimeError。
- 结论：run metadata与telemetry payload必须使用两个不同路径；失败v1目录保留为证据，不覆盖、不删除。

## FW-DVAC-RUN-002 — retry1在任何远端写入前由clean gate拒绝

- retry1 command file：
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_launch_p1_16ep_gpu3_retry1_20260822.sh`，SHA-256
  `4aab8f880a5d62139a149239fa1f77ceb379768f0f9560012f97f092cb5ddb76`；候选run id为
  `fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`，已把metadata与payload路径分离。
- helper会话无输出结束后先只读核验。`shenzhen_fastwam_v2_retry_readonly_20260822.sh` / 
  `shenzhen_fastwam_v2_preflight_predicates_20260822.sh` SHA-256分别为
  `d6ce78501126ae4c9e43352aa5d5d19ba9787bb63a9767b1931f4f704814dcb9` /
  `2d39737a8307e1dc92f7f2301d807c07ec4a76fde93d21b641b4d4de7cf2079a`；均exit0且只读。
- v2 metadata、payload与official result三个目标均完全不存在，GPU3空闲；说明retry1没有执行到
  `mkdir -p "$META"`，也没有启动模型/仿真。
- 精确最早失败是全树clean断言：worktree HEAD仍精确`c63dc9b5...`、tracked文件无diff，但official evaluator
  setup留下三个预期untracked symlink：`third_party/RoboTwin/assets`、`policy/fastwam_policy`、
  `task_config`。`test -z "$(git status --porcelain)"`因此退出。checkpoint/stats/env、GPU3和两个canonical
  asset links均正常。
- 正确窄处理应保留并逐项验证这三个链接，只把tracked diff或额外untracked项视为失败；本条只定位，
  没有删除链接、改脚本、修复或再次启动。

## FW-DVAC-RUN-003 — v2 clean predicate窄修后直接启动真实P1 sequential-16

- 失败边界不改写：RUN-001的v1已经进入official evaluator/SAPIEN setup并输出`Render Well`，但在首个
  telemetry query/episode前因metadata与payload复用同一路径而退出；没有query/episode telemetry、video
  或已完成model inference/GPU model workload证据。v1目录及driver/child log作为失败证据原样保留，
  没有覆盖或删除。RUN-002的第一版v2 retry1则更早被全树clean断言拒绝；后续只读确认它尚未创建v2
  metadata、payload或official-result路径，也没有启动模型、GPU workload或simulator。
- 最终launch脚本仍为
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_launch_p1_16ep_gpu3_retry1_20260822.sh`，修订后
  3,866 bytes，SHA-256
  `966dfbb5cd07737a60e8e7c00717c9611cf3c382b58e0a9eb08404c5a7d3b81c`。唯一针对已观察问题的窄化是：
  tracked diff必须为空；untracked集合必须精确等于`third_party/RoboTwin/{assets,task_config}`与
  `third_party/RoboTwin/policy/fastwam_policy`三个既有symlink，并逐一核验其resolved target；额外
  untracked项仍会拒绝启动。
- 本次没有再执行real-query off/on parity Gate或2-episode P0，也没有成功率/信号阈值Gate。source、
  checkpoint/stats、output absence、GPU3 idle与上述三个symlink target通过确定性preflight后，脚本直接调用
  official `experiments/robotwin/eval_robotwin_single.py`进入真实episode evaluator；这些启动前合同不是另一次
  模型/simulator Gate。
- resolved预算保持不变：physical GPU3；official `B=1`顺序16 episodes；`H=32/C=24/M=10`、每episode
  hard limit 400；上限6,400 action slots、272 policy queries、2,720 denoise model steps；telemetry仅旁路
  保存，不训练、更新或保存checkpoint。hard timeout为3小时，只约束本轮owned process。
- v2路径彼此分离且没有复用失败v1：metadata为
  `/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`；payload为
  `/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`；official result为当前
  worktree的`evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`。
- 最终v2后台PID=`637492`，resolved SHA-256=
  `bf9d41c7e435377e0c3973a35336c51816a40ccd0b41519c4fc5a6c3d2c639fd`。launch stdout已出现
  `Render Well`并进入真实official evaluator；同时出现`missing pytorch3d`提示。启动检查点只把后者记录为
  待live-health定性的提示，不在缺少后续fatal/exit证据时擅自记为致命错误。
- 本条结论严格是“v2真实P1成功启动”，不是“16 episodes已经完成”。episode/query/telemetry/video终态、
  success与进程exit仍须后续只读health/completion证据确认；本条没有发送signal、删除失败产物或覆盖路径。

### 21:38–21:39 CST v2运行中只读快照

- 只读command files：`shenzhen_pi0_fastwam_completion_refresh_20260822.sh` / 
  `shenzhen_pi0_fastwam_csv_counts_20260822.sh`；SHA-256分别为
  `e96e9248793aaeedb2f28615f4c586a1af752861a997ace283a1e1c630926e0e` /
  `6f5681ebce0ff7e115a9e92aedd9723eb4ea974a832082e268aa81edc9b115ea`；均exit0。
- 后续主线已用窄化后的预期symlink gate真正启动同一v2 run；本条只记录现场，不复述启动实现。
  timeout PID`637492`及child`637496` alive；GPU3约`30,610 MiB`、100% util，fatal扫描0。
- `episodes.csv`已有4条completed rows且全部`success=true`，即当前`4/4`；每条5个query，共20条completed
  query rows。另有episode5 query0 trace，表示第5条正在运行，不能计入成功率分母。
- payload当时已有21个NPZ；前4条完成episode各有5条trace和三路query PNG；official目录已有4个完整
  success MP4，另一个48-byte临时MP4正在写入。metadata、payload、official路径分别为：
  `/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`、
  `/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`、
  `evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`。
- 同时GPU4–7无compute app，host`MemAvailable≈1.941 TiB`；没有与PPO或π0残留竞争。全程未发送signal。

### 21:49自然完成；21:51–21:52 CST终态只读核验

状态：`COMPLETE_RC0_16_OF_16_SUCCESS`。21:49:11--21:49:14 CST official child与driver日志自然写完，
明确出现`Success rate: 16/16 => 100.0%`、`Data has been saved`与
`Evaluation finished successfully`；没有通过signal、timeout或人工停止结束。

- 终态复用前节两份只读command file：
  `shenzhen_pi0_fastwam_completion_refresh_20260822.sh`
  （SHA-256=`e96e9248793aaeedb2f28615f4c586a1af752861a997ace283a1e1c630926e0e`）与
  `shenzhen_pi0_fastwam_csv_counts_20260822.sh`
  （SHA-256=`6f5681ebce0ff7e115a9e92aedd9723eb4ea974a832082e268aa81edc9b115ea`）；
  21:51:36和21:52:01 CST均exit0，密码仍只进入当前Paramiko交互进程。
- owned timeout PID`637492`与Python child均已自然退出；GPU3=`4 MiB`、0% util且compute-app列表无GPU3
  进程，说明Fast-WAM模型显存已经释放。此时GPU4--7上的进程属于另一路GRPO formal，不计入Fast-WAM。
- `episodes.csv`为17行（1 header + **16 completed episodes**），16行的`success`均为`True`；因此当前
  fixed block实测为**16/16 success**。`queries.csv`为81行（1 header + **80 completed queries**），
  即每条episode精确5个query；最后一条在query 4的action slot 117成功。
- telemetry payload终态共有：`episodes.csv ×1`、`queries.csv ×1`、`run_manifest.json ×1`、
  **NPZ 80**、**PNG 240**（每query三路图像）。official result终态有**16个完整MP4**，全部文件名为
  `success-true.mp4`；先前第16条的48-byte临时MP4已被完整的316,803-byte视频取代。
- driver与official log的fatal关键词扫描为**0**。official目录同时生成`_result_clean.txt`、resolved YAML
  与完整eval log；metadata/payload/official三路径保持分离，没有覆盖失败v1证据。

本块只报告accepted clean seeds上的16条official sequential evaluator轨迹，16/16是该fixed block的
经验结果，不外推到随机seed全集或其他RoboTwin任务。它已经满足本轮“official Fast-WAM RoboTwin推理、
完整视频、DVAC query/trace旁路信号均成功落盘”的执行目标；后续信号统计与视频对齐应直接读取上述80条
query记录，不需要重跑本块。

## FW-DVAC-P2-PLAN-001 — 22:45 CST：64条统计集中的三任务扩量包

状态：`LOCAL_PACKET_COMPOSED_NOT_LAUNCHED`。

- 用户授权继续按原推理路径扩量，并希望Fast-WAM覆盖多任务、同时观察成功与失败。首块
  `adjust_bottle×16`已经完成，因此总计64条的新增部分固定为：
  `move_stapler_pad×16 + turn_switch×16 + pick_diverse_bottles×16`。三项分别提供搬运放置、接触旋转和
  多样物体抓取场景；选择依据是官方release checkpoint为RoboTwin多任务权重、official manager原生枚举多任务，
  且官方论文表中三项不像`adjust_bottle`一样饱和。论文成功率只用于选择有信息量的任务，不能代替本机实测。
- 运行语义不变：official `experiments/robotwin/eval_robotwin_single.py`、physical GPU3、每任务B=1顺序
  16 episodes、`demo_clean`、unseen instruction、`H/C/M=32/24/10`、sigma shift 5、CPU RNG、BF16、
  `skip_get_obs_within_replan=false`以保存逐action视频帧；三个任务的official step limit均为400，避免首批选择
  `open_microwave`（1500）或`hanging_mug`（900）无谓放大运行时间。DVAC telemetry仍只是旁路记录，不改action或RNG。
- 本地新增精确command file：
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_launch_p2_multitask_3x16_gpu3_20260822.sh`，
  4,781 bytes，SHA-256=`22faa66c72844011d811856a0ba5e67946cb97b2332edbc576de004dd27c1534`。
  它只做一次GPU/source/asset/output合同检查，然后由一个owned wrapper按上述顺序调用三次official single-task
  入口；任一入口非零即停止后续任务，不自动重试。总新增预算48 episodes、每任务最多6,400 action slots和
  272 queries；wrapper hard timeout 12小时，只约束本轮owned process。
- 预定输出：parent metadata
  `/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-multitask-p2-3x16-c63dc9b5-v1`；
  每任务payload为`/data/chenyiteng/results/dvac-observation/fastwam-<task>-p2-16ep-c63dc9b5-v1`，
  每任务metadata和official result使用同名run id。尚未上传、`bash -n`、compose或启动；启动前仍需刷新GPU3、
  任务代码/assets和GRPO内存现场。

## FW-DVAC-P2-RUN-001 — P2 packet上传、compose与排队启动（2026-08-22）

状态：`RUNNING_MOVE_STAPLER_PAD / PARENT_ALIVE`。

- 本地exact launcher为
  `local_scripts/remote_commands/shenzhen_fastwam_dvac_launch_p2_multitask_3x16_gpu3_20260822.sh`，
  4,781 bytes，SHA-256=`22faa66c72844011d811856a0ba5e67946cb97b2332edbc576de004dd27c1534`。
- 预定持久packet目录为
  `/data/chenyiteng/results/dvac-observation/packets/fastwam-multitask-p2-3x16-c63dc9b5-v1`；
  上传后将在服务器核对exact SHA-256并`bash -n`，再用同一overrides分别CPU compose三任务resolved配置。
- 23:13 CST第一次只读刷新：π0 fixed64 owned PID仍alive，用户内Raylet=1，GPU0--3约
  `29.2/29.1/29.1/29.4 GiB`；fatal=0，eval epoch已到`1/1`，但owned PID/Ray/GPU尚未自然清理。
  因此本轮不争抢GPU3，只低频只读等待。
- 23:16 CST，packet目录在确认不存在后创建，launcher经SFTP上传；服务器核对
  SHA-256精确为`22faa66c...c1534`，`bash -n`通过。强制`CUDA_VISIBLE_DEVICES=`后用将实际
  启动使用的同overrides生成三份resolved：`move_stapler_pad=dd8f2902...79ae1b`、
  `turn_switch=2bff3ce1...f60d9e`、`pick_diverse_bottles=122fceeb...4059ad`。
- 23:16:36 CST最后只读调度点：π0 owned PID已自然退出，`raylet=0/gcs=0`，GPU0--3
  无compute process且显存均为0 MiB；fatal=0，终态fixed64为`42/64=65.625%`。host
  `MemAvailable=2,091,826,296 KiB`，`/data` available=`3,317,354,516,480 bytes`。Fast-WAM P2的
  source/model/stats/symlink/task/output断言由exact launcher在创建任何run目录前再完整执行。
- 23:17:32 CST，exact launcher的source/model/stats/symlink/task/output/GPU3断言全部通过，
  不增加模型Gate即启动。parent timeout/wrapper PID=`819933`，parent metadata为
  `/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-multitask-p2-3x16-c63dc9b5-v1`；
  第一任务`move_stapler_pad`的run id为`fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1`。
- 23:19:51 CST首任务健康点：parent及child仍alive，`Render Well`、official 5B/1.02B
  experts初始化和checkpoint override路径均已进入；GPU3 child=`819945`，约`30,248 MiB`，
  util=`63%`，query rows=`5`（第1条episode进行中），fatal=`0`，host
  `MemAvailable=2,084,001,456 KiB`。当前只能声明“首任务已健康运行”，不把0条completed
  episode误记为outcome；后续保持同一owned wrapper自然串行，不stop/delete/retry。

### 23:25 CST：`move_stapler_pad` 首个成功/失败混合节点

- 只读command file：
  `local_scripts/remote_commands/shenzhen_fastwam_p2_readonly_monitor_20260822.sh`，2,581 bytes，
  SHA-256=`33374449188f84c0be7acd2a4e4560b1f758590ee16bc02e6ac06a5924f0b1d0`；通过固定host-key的
  Paramiko密码路线以`chenyiteng`执行，没有向远端写文件或发送signal。
- `2026-08-22 23:25:41 CST`，parent=`819933`、official child及policy child=`819945`均alive；
  当前任务仍为`move_stapler_pad`。已有**6条completed episodes：5 success / 1 failure**，即当前
  fixed顺序块`5/6=83.3%`；52条completed query、53 NPZ（第7条已经开始）、159 PNG与6个完整MP4。
  task driver fatal扫描为0。
- GPU3为`31,142 MiB / 100%`，唯一compute PID为本run的`819945`；host
  `MemAvailable=2,082,946,960 KiB`，`/data` available=`3,317,343,023,104 bytes`。首任务已自然获得
  用户所需的成功/失败混合信号；不据此停止或补挑seed，继续同一wrapper自然完成16条并切换后续任务。

### 23:40--23:42 CST：`move_stapler_pad`自然完成并切换`turn_switch`

- parent log明确记录`task_complete=2026-08-22T15:40:33+00:00 task=move_stapler_pad`，紧接着同秒
  `task_start=... task=turn_switch`；第一任务不是signal、timeout或异常退出，official末尾为
  `Data has been saved`与`Evaluation finished successfully`。
- `move_stapler_pad`终态为**16 completed / 11 success / 5 failure = 68.75%**；162 query rows、
  162 NPZ、486 PNG与16个完整MP4，fatal扫描0。最后一条accepted seed日志为`4300015`；本块没有按
  outcome补挑seed。
- `2026-08-22 23:42:16 CST`，wrapper parent=`819933`仍alive，`turn_switch`的新official/policy child
  已进入5B/1.02B experts初始化并打印`Task Name: turn_switch`；此时还没有completed episode，不把初始化
  记为推理结果。GPU3=`26,056 MiB`，host`MemAvailable=1,812,860,008 KiB`，`/data`
  available=`3,317,267,386,368 bytes`。保持同一wrapper自然运行。

### `move_stapler_pad`独立phase标注代表样本（不读DVAC数值）

- 只读source command：
  `local_scripts/remote_commands/shenzhen_fastwam_move_stapler_representative_source_20260822.sh`，
  817 bytes，SHA-256=`ffe3ea14bc8da9c696c47902a13831de46f63ae73f04d74de02ec84c42cae4c9`；
  `2026-08-22 23:49:37 CST`只读取已完成run的`episodes.csv`、official log中的16个`current seed`和
  完整MP4文件名/大小，没有读取任何NPZ或DVAC派生值。
- 选择规则严格固定为：在success/failure各自outcome内，以`total_action_slots`到该outcome中位数的
  绝对距离最小者为代表；并列先取accepted seed更小者，再取`episode_id`更小者。accepted seed按official
  log的完成顺序与16条episode row一一join。
- success的11个长度排序中位数为`149`；两个精确命中者中accepted seed较小的是：
  `episode_id=4 / reset_id=3 / accepted_seed=4300003 / total_action_slots=149`，视频：
  `/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode3_randomized-false_success-true.mp4`。
- failure的5条长度均为`400`，所以中位数也为`400`；按tie规则选择最小accepted seed：
  `episode_id=6 / reset_id=5 / accepted_seed=4300005 / total_action_slots=400`，视频：
  `/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode5_randomized-false_success-false.mp4`。

### 两条代表视频的12帧phase候选contact sheet

- 用户授权后使用现有Fast-WAM venv中的OpenCV只读上述两条代表MP4；command file为
  `local_scripts/remote_commands/shenzhen_fastwam_move_stapler_contact_sheets_20260822.sh`，
  2,593 bytes，SHA-256=`8310a8a860f9d017b1ef983ab857930cd341874e705aab3d9eb2ce302a94cef5`。
  每条视频在完整frame范围内均匀取12帧，4×3排版；图中文字只有原视频frame index和time，不读取、
  显示或使用DVAC/模型指标。没有回写source video、payload或official result。
- 全新派生目录：
  `/data/chenyiteng/results/dvac-observation/phase-candidates/fastwam-move_stapler_pad-p2-v1`。
  输出为：
  - `success_episode_id4_seed4300003_contact_sheet_12f.png`：460,873 bytes，SHA-256
    `8439f464b0e50ab5b9ac3b6cec3fccbb631483d6937652b01194c92cfb7d3188`；
  - `failure_episode_id6_seed4300005_contact_sheet_12f.png`：478,232 bytes，SHA-256
    `1cf55389c64f7639c2e90ca5f900239f55a922d51abbe12f05826b51eddc6f63`。

### 独立phase CSV加入后的CPU离线派生

- 用户先只看raw contact sheet、未看本任务DVAC即完成16行query-level phase CSV；其SHA-256为
  `a949e73c...8afc`。目标packet文件与analysis output均确认不存在后，phase CSV经SFTP普通上传，并用
  已核`009b4ea4...ff152`分析器、现有Fast-WAM venv、空`CUDA_VISIBLE_DEVICES`执行唯一一次离线分析。
- 新output：
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1`；
  15:56:04--15:56:12 UTC自然exit0，wall8.18秒、max RSS336,772 KiB。它只读已自然完成的
  `move_stapler_pad` payload/video，不影响GPU3上正在运行的后续task wrapper。
- 后检：16 episodes、162 queries、10,368 horizon rows、3,741真实executed-action/frame rows；标签覆盖
  16 query/365 action rows，得到18条phase-episode与60条phase-summary；success ep4/failure ep6两张
  storyboard均创建。28 files / 17,591,880 bytes，12 PNG全部解码；分解恒等式误差最大`1.78e-15`，
  all-finite、Euler reconstruction与final action合同均通过。完整命令/SHA与一次只读postflight字段假设
  窄修见RLinf专题`evidence/15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md`第07节。

### 2026-08-23 00:01--00:06 CST：`turn_switch`自然完成并切换最终任务

- parent log记录`task_complete=2026-08-22T16:01:55+00:00 task=turn_switch`，下一秒自然
  `task_start=... pick_diverse_bottles`；official尾部为`Data has been saved`和
  `Evaluation finished successfully`，没有signal/timeout/retry。
- `turn_switch`终态：**16 episodes / 10 success / 6 failure = 62.5%**，133 query rows、133 NPZ、
  399 PNG、16个完整MP4，fatal0；最后accepted seed日志为`4300020`。
- `2026-08-23 00:06:06 CST`，最终任务`pick_diverse_bottles`已完成前2条且均success，14 completed
  query、15 NPZ（第3条已开始）、45 PNG与2个完整MP4，fatal0。wrapper parent=`819933`及新child
  alive；GPU3=`31,447 MiB / 100%`，host`MemAvailable=1,626,842,964 KiB`，`/data`
  available=`3,317,201,465,344 bytes`。保持自然运行至16条。

### 00:24--00:28 CST：P2三任务wrapper自然完成终态

状态：`COMPLETE_RC0_SEMANTICS / 48_EPISODES / GPU3_RELEASED`。

- parent log记录`task_complete=2026-08-22T16:24:45+00:00 task=pick_diverse_bottles`与同秒
  `wrapper_complete=...`；最终official日志为`Data has been saved`和`Evaluation finished successfully`。
  `2026-08-23 00:28:18 CST`只读核验时parent PID=`819933`及所有Fast-WAM child均已自然退出；
  GPU3=`5 MiB / 0%`且compute-app列表为空，模型显存已释放。没有signal、timeout、人工停止或retry。
- `pick_diverse_bottles`终态：**16 episodes / 12 success / 4 failure = 75.0%**；128 queries、128 NPZ、
  384 PNG、16完整MP4，fatal0；最后accepted seed日志为`4300025`。
- 本次P2新增三任务合计：**48 episodes / 33 success / 15 failure**，423 queries、423 NPZ、1,269 PNG、
  48完整MP4，三份driver fatal均0。逐任务为：
  - `move_stapler_pad`：`11/16`，162 query/NPZ、486 PNG、16 MP4；
  - `turn_switch`：`10/16`，133 query/NPZ、399 PNG、16 MP4；
  - `pick_diverse_bottles`：`12/16`，128 query/NPZ、384 PNG、16 MP4。
- 加上此前冻结的`adjust_bottle 16/16` P1，当前Fast-WAM首批四任务总集为：**64 episodes / 49 success /
  15 failure**，503 queries/NPZ、1,509 PNG、64完整MP4。该数值只描述四个固定顺序块，不外推为benchmark
  总体成功率。
- 终态host`MemAvailable=1,587,474,976 KiB`，`/data` available=`3,317,162,700,800 bytes`；此时GPU4--7
  的GRPO v2继续独立运行，本wrapper从未改变其source/config/process。
