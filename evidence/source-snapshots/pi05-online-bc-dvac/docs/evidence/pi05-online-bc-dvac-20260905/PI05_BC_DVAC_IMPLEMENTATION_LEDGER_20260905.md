# π0.5 BC＋DVAC：GPU7正式训练实施账本

## 当前授权与实施边界

用户授权在已跑通π0.5 BC之上迁入既有π0 BC-DVAC增量，权重范围[0.5,1.5]，简测、push、同参数正式训练；随后明确GPU7、**不跑smoke**。本轮仅服务器CPU/接线回归及resolved对照后直接fresh formal100；GPU6基线、GPU4/5 Sidney/shared Ray/其他用户不动。不升级依赖、不清理产物、不续smoke或基线在线权重。

方法沿已实现同次推理末L3 endpoint总体方差、past5 log-z、首轮等权、入池固定w、逐动作FM监督项加权；z_clip2时alpha0.125保证w=1+alpha*(z-有效位置均值z)在[0.5,1.5]且均值1，不额外minmax强行铺满范围。该强度替换旧alpha0.25，其余方法设置不变；模型M10不改。

## 逐操作

1. 完整读取根规则/交接、窗口交接和唯一DVAC设计§1—11；读取已有算法136行、11项DVAC回归、配置组及关键采样/监督/collector接点。确认不是新增训练框架；准备精确对比五个旧接线文件的基线hash，再移植九文件既有增量。最新用户no-smoke优先于历史smoke要求。
2. 09-05 23:52:48普通账号只读preflight：π0.5 BC HEAD6a93605d、旧π0 DVAC912808c7均clean；五个待接线文件与旧DVAC提交736b1416的父版本逐字一致。GPU7=10MiB/0%无compute，GPU6基线正在监督更新、无所查错误，原shared Ray和Sidney wrapper均在；RAM available1253.4GiB、/data余875.02GiB。证据`PI05_BC_DVAC_PREFLIGHT_20260905.json`。
3. 23:56从6a93605d建独立`codex/sz-pi05-online-bc-dvac`/`pi05-online-bc-dvac`树；仅对精确9文件应用736b1416既有patch，`git apply --check`/`git diff --check`通过，无冲突。源码五处接线＋136行模块直接复用；原BC/现役任务未改。原patch和逐文件hash已落本地证据。
4. 09-06 00:01完成范围适配：新增10行配置组`bc_dvac/bounded_half.yaml`继承原default，仅alpha0.125；既有采样器测试扩至M4/M10、状态测试扩至alpha0.25/0.125；新增π0.5组合配置等价及极端信号/partial mask范围测试。方法模块、采样器、collector、actor/FM接线相对旧DVAC逐字不改，不新建网络或渲染补丁。部署只上传两个测试/配置/方法说明；基线/source锁不变。
5. 00:02启动服务器CPU单元回归（CUDA_VISIBLE_DEVICES空，仅测试进程），然后用新分支真实Hydra配置对照GPU6正式resolved。用户要求不smoke，本轮无新GPU推理探针、试训或短评估；通过后直接正式。
6. 00:03全部28tests通过（9.83秒），无失败；M4/M10真实采样循环＋stub velocity的动作/RNG/调用次数一致，FM权重loss/梯度/sidecar、alpha0.125范围/mask及π0.5配置组合通过。注意不是实际模型GPU前向。Hydra/validate_cfg通过，17叶差异严格为7 DVAC＋GPU7＋9名称/路径；种子表、Sidney adapter及原模型/BC配置内容一致，无预算变更。只连接既有Ray检查节点，不重启Ray或训练worker；证据`PI05_BC_DVAC_TEST_CONFIG_20260905.txt`。
7. 00:06源码/配置/测试及小证据已commit/push：`626825e88aacedcf3e3000d177a364a170736e66`，新个人分支`codex/sz-pi05-online-bc-dvac`、clean。15文件中多数是既有方法/测试及resolved证据；生产增量为旧六文件完全复用，原五处接线+115/-12，加136行已有权重模块。真正新增逻辑无，新增方法配置10行，仅alpha0.125；新测试71行及旧测试参数化。
8. 完整正式resolved/命令/新输出路径/3200采集/1000Adam上限/640评估/10checkpoint/48h上限已在聊天展示，用户明确授权直接正式且不smoke；开始单次GPU7 fresh launch。原基线/模型/成功池不覆盖，固定seed表内容不变。
9. 00:09:46起单次launch命令成功：wrapper/PGID2223376、observer2223377，GPU7启动前无compute；/data余874.92GiB。所有11个方法/配置/测试文件逐一对照本地通过、目标目录先确认不存在；wrapper `bash -n`通过。新run `pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1`，随后仅检查健康启动，不等待完整首轮。
10. 00:12:17启动检查：实际started_at=09-06 00:09:52 CST；wrapper及Env2224188/Rollout2224184/Actor2224182均在，source-head626825e8/clean，无exit/finished、所查fatal/OOM/Traceback等0。三个worker日志均placement7，Env NOFILE4096，rollout已读原Sidney norm，actor仍加载；尚无完整Step1/正式fixed/checkpoint。GPU7初始化现场1.63GiB，不能当采样峰值；GPU6约64.72GiB/89%，RAM available1184.7GiB、/data余874.90GiB。原GPU smoke没有运行，不把初始化健康当完整端到端验证。证据`PI05_BC_DVAC_STARTUP_20260905.json`。
11. 00:14:39最终接线/启动验收：driver2223401实际cmdline含`+online_bc_model=pi05_sidney +bc_dvac=bounded_half`及正式100/eval5/save10/total1000；三个worker的GPU7、新repo/model/run路径一致，无Fast shim。actor加载原norm，expert_only=True、693,422,112可训练参数，已进入首轮采集；所查错误0。基线HEAD6a93605d/clean、原BC2143105/shared Ray321933/322685/Sidney602620均在。证据`PI05_BC_DVAC_BINDING_20260905.json`。结束主动启动检查，不等待首轮/U10结果，不声称非均匀权重已实际进入更新；第一轮设计上全1。
12. 00:16追加启动证据时`git diff --cached --check`仅拒绝launch.txt多余EOF空行，尚未commit；不是训练报错。只修本批5个证据文件的发布副本尾空行，检查dirty路径均属于本批后重新提交；生产代码/配置/模型/进程不动，不重放launch。原始本地输出仍保留。
