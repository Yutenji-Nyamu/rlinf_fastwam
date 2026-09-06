# Fast-WAM clean GRPO / OIDN off / fresh100 实施账本

## 1. 本轮授权与唯一变化

2026-09-04用户最初授权：基于此前完成Step33的Fast-WAM GRPO，全部训练参数原样，关闭OIDN，从头训练。
启动前追加授权：本次采样翻倍，环境并行数不变，因此rollout4→8；其余配置原样。
新建独立RoboTwin分支，从`0008ae6800df9f75fc8de7098bacb01735fd8fd2`干净基线出发，仅接入已验证的denoiser参数。
旧`8c7380c1`生命周期补丁不进入本次新分支，原分支/worktree保留；不原地撤销，不升级库。
复用Fast-WAM current GRPO实现、原始release SFT权重；不恢复Step10/30。新256档不复制旧256档的GB2048改动，GB1024保留。
只使用刷新后空闲的GPU6/7；保留Sidney GPU4/5、shared Ray与其他用户。不得因关闭OIDN顺便改变渲染器/SPP/光照/相机。

## 2. 逐操作账本

| 顺序 | 命令/目标 | 结果、原因与后续 |
|---|---|---|
| 1 | 完整读AGENTS/PROJECT_CONTEXT/HANDOFF/09-03窗口交接/current Fast-WAM SSOT | 继承同源配置与shared Ray隔离规则；旧快照不作当前现场 |
| 2 | 读取既有`tmp_oidn_resume_launch.sh`和运行绑定；准备`sz_fastwam_clean_nooidn_preflight_20260904.sh` | 沿用已验证launcher；即将通过固定host-key普通账户只读刷新身份、资源、源锁与原resolved |
| 3 | 12:34 CST运行只读preflight，取回原run resolved/command | 普通账户uid1003；GPU6/7各5MiB且无compute PID；RAM可用1.271TB；/data余1.3TiB。Sidney完整42步、所查fatal/OOM/traceback=0；shared Ray PID321933/322685未变。旧Fast完整33步/exit255；原源锁均符合预期 |
| 4 | 准备`sz_fastwam_clean_nooidn_prepare_20260904.sh`，新worktree从0008ae6 cherry-pick仅BaseTask开关提交b76d4ed | 将核验vector_env blob与干净基线相等；源代码仅2处，资源复用symlink；服务器compose/import并逐叶核对，旧补丁分支保留。暂无训练启动 |
| 5 | 12:37服务器prepare通过 | 新RoboTwin commit=`f3e30a83365cdda1165911422dc3ce73e703201e`，clean；vector_env与0008ae6完全相同。服务器import来源正确，AST/compose通过；9项resolved差异全在白名单：2个OIDN开关、resume=null、6个命名/输出路径。RLinf `4faade1d`相对原`7b2331c5`的rlinf/examples代码无差异 |
| 6 | 启动前收到用户追加授权：采样翻倍、并行不变 | 原32x4 packet已准备但从未启动。改成32x8；GB1024/update_epoch2不变，实际optimizer calls随数据翻倍到每步4次，已在聊天明示；准备新32x8 packet，不覆盖旧packet |
| 7 | 12:42服务器256 compose与逐叶diff通过；聊天展示完整resolved/精确命令/资源/预算/停止条件 | 相对33步run仅10项差异；新增唯一训练叶`env.train.rollout_epoch:4→8`。运行预算已按用户新增授权翻倍，其余无改变 |
| 8 | 上传单项wrapper并执行`sz_fastwam_clean_nooidn_launch_20260904.sh` | 启动前再次验证GPU6/7无计算进程、旧shared Ray/Sidney PID存活、源锁/hash/namespace可用；仅新run wrapper与资源observer，结果待启动检查 |
| 9 | 12:43:27启动，12:45:39第一次只读验收 | wrapper/PGID1052625、observer1052626、driver1052633；EnvWorker1053118/1053120、rollout1053115/1053117分别绑定GPU6/7，PYTHONPATH/ROBOTWIN_PATH均指向clean新树。resolved train/eval均none、resume=null；两rollout均从原始release构建。此时仍初始化，完整step=0；所查OIDN/pthread/fatal/OOM/Traceback/RayActorError均0 |
| 10 | 12:45执行有界45s push到既有`personal` | `codex/sz-robotwin-clean-oidn-off@f3e30a8`已推Yutenji-Nyamu/RoboTwin；remote-ref与本地HEAD一致，clean。没有push官方origin、改remote或全局Git配置 |
| 11 | 12:47—12:48只读追踪模型初始化与actor预算代码 | 两actor于12:47:06/07从原release构建完成，12:48:40已进入`Generating Rollout Epochs 0/8`。`embodied_fsdp_actor_worker.py:522-556`确认按GB/rank切分后循环update_epoch，4次optimizer calls/步不是修改update_epoch。初次只读探针误用fsdp_actor.py文件名后更正到实际文件，无训练代码变化 |
| 12 | 12:49:46最终启动只读刷新 | 新wrapper/driver/两rank存活，采样0/8（首步尚未完整）；GPU6/7=54,411/54,409MiB；OIDN/pthread/fatal/OOM/Traceback/RayActorError均0。Sidney仍完整42步且所查错误0，shared Ray原PID保持；不继续高频轮询 |

## 3. 当前启动前合同（已按追加授权更新为256轨迹）

以下为启动前已展示并按最新授权执行的合同；实际启动/进展见第2、4节，不将启动等同完整step或稳定性验收。

- [完整resolved](clean-nooidn-fresh100-20260904/256-resolved.yaml)、[精确训练命令](clean-nooidn-fresh100-20260904/256-command.txt)、[逐叶差异及预算JSON](clean-nooidn-fresh100-20260904/256-contract.json)、[完整代码diff](clean-nooidn-fresh100-20260904/robotwin.patch)。同目录无256前缀文件是未启动的原128准备记录，不是活动合同。
- run slug：`fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1`。
- 输出：`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1`。
- RoboTwin：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904`，分支`codex/sz-robotwin-clean-oidn-off`；旧两个补丁/诊断分支均不动。
- RLinf：仍使用独立于Sidney的`fastwam-current-grpo` worktree；旧Fast已终止，因此不是两个活动job共享该树。
- 权重：`/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt`；resume/ckpt_path均null。不是Step30续训。
- GPU：物理6/7，两张H100 80GB；RAM数百GiB量级，实际持续记录resource.csv。旧256档有OIDN时曾触及约81GB/卡并OOM；关闭OIDN后是否能承载尚待实测，不保证峰值。
- 渲染：train/eval均`ray_tracing_denoiser=none`，RT、SPP32、pathdepth8、相机、光照与旧实验不变；这是显式RGB协议变更，不与有OIDN指标无标记拼接。
- 训练：32env×rollout8=256轨迹，G8、32个独立G8组、2048条query/步；GB1024/MB2/update_epoch2不变，因每epoch现在有2个GB，实际每步4次optimizer calls。LR5e-6、noise0.3、H32/C24/M10、horizon192、offload同原run。
- 评估/保存：fixed32每5步、DCP每10步，eval视频开启、train视频关闭；无额外baseline评估。

| 预算 | 本次fresh100（采样翻倍，其他配置不变） |
|---|---:|
| 外层steps / train episodes | 100 / 25,600 |
| train提交action slots上限 | 4,915,200 |
| 新query records /累计呈现上限 | 204,800 / 409,600 |
| actor optimizer calls / critic updates | 400 / 0 |
| eval episodes | 640（20×32） |
| checkpoint代数 | 10（预计合计约289GB，另有视频/日志） |
| wall-clock粗估 / GPU-hours粗估 | 约50h / 约100 GPUh（关闭OIDN后速度待测） |
| 继承wrapper硬上限 / GPU预留上限 | 120h / 240 GPUh |

停止条件继承：完成Step100、driver异常/fatal/OOM退出、或120h timeout；不自动重试、不改参数继续跑，不重启shared Ray。
长程稳定性以及是否改善学习均待本次自然结果；关闭OIDN不保证消除其他原生渲染问题。

## 4. 启动验收与后续

新run已在GPU6/7从原始SFT启动，源路径/无旧补丁/none配置/原始权重均已核验。
首步8轮采样已开始；尚未完整Step1，未到首次checkpoint保存点，不宣称256档更新显存或DCP保存/恢复已经通过。
后续自然运行到100；所需资源监测由run内observer每60秒写resource.csv，未新增Codex定时任务。
长程结束后再回填最终训练轻量证据；当前生产小改动已commit+push用户RoboTwin分支。

原始证据：

- [启动前现场](CLEAN_NOOIDN_PREFLIGHT_20260904.txt)
- [干净分支与128准备](CLEAN_NOOIDN_PREPARE_20260904.txt)，[最终256准备](CLEAN_NOOIDN_256_PREPARE_20260904.txt)
- [真正启动](CLEAN_NOOIDN_LAUNCH_20260904.txt)，[12:45两rank绑定](CLEAN_NOOIDN_STARTUP_20260904.txt)
- [actor切批源码/首轮采样](CLEAN_NOOIDN_BUDGET_SOURCE_VERIFIED_20260904.txt)
- [最新只读启动验收](CLEAN_NOOIDN_ROLLOUT_STARTED_20260904.txt)，[用户fork发布](CLEAN_NOOIDN_PUBLISH_20260904.txt)
