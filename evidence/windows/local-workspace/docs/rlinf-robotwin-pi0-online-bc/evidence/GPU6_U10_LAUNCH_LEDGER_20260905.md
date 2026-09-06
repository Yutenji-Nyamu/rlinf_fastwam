# GPU6 U10：本轮实施与正式启动账本

用户已确认U10、micro32/global1024，其余按已同意的32×1采集、eval16×2固定32初态、FD4096、expert-only、M4、原SFT/空池、正式100轮；授权GPU6正式启动，开始后即报告。保留已有“完整smoke通过再正式”的边界；无需重复已通过的精度/FSDP/FD隔离probe。GPU7、Sidney、shared Ray、其他用户不干预。

1. 恢复根规则/交接/窗口入口/BC唯一SSOT，读取已有合同、部署脚本及确切eval/reset接口。旧v6/旧formal脚本不执行。已知待实现：U2→10、eval32×1→16×2且保持原32不同固定初态；既有模型/方法逻辑不重写。
2. 首先普通账号固定host-key Paramiko只读核对身份、GPU6/7、资源、BC源码基线/dirty、活跃任务与磁盘。现场结果待本轮记录；不会用旧快照宣布可启动。
3. 15:41 CST现场身份chenyiteng、BC HEAD700b6846 clean，GPU6/7各11/4MiB，RAMavailable约1.4TiB、memory PSI0，/data余556GiB。原shared Ray/Sidney进程存活；不干预。四个目标文件hash已保存，待修改本地两份源码/config及test与服务器一致。证据`U10_PREFLIGHT_20260905.txt`。
4. 读取服务器seed_utils和环境配置，确认单worker先全局shuffle再取前N；本轮保持旧BC单卡32环境的原32种子，不冒称与旧GRPO双worker分区完全一致。只新增RoboTwin opt-in fixed_reset_batch_count=2，显式非auto-reset评估批次，复用既有finish_rollout推进seed后循环回首批；无需更改EnvWorker/rollout通信。配置U10、eval16×2、optimizer total20；测试验证两批拼接等于旧32、连续3轮重复同一集合及默认fixed行为不变。未改模型、loss、采集并发、图像、依赖或其他任务。
5. 15:48仅上传3个确切目标，服务器Ruff格式化/import顺序、diff检查、11/11 tests通过（10.69s），真实validate_cfg为物理[[6]]；实eval seeds两批各16不重合且恢复首批，smoke/formal相同。两份完整resolved与逐叶diff已生成，差异仅轮数/interval/total/名称及派生路径，未创建训练环境。既有shared Ray连接诊断正常退出，不重启head。
6. 已展示最终完整resolved、精确命令/输出/预算/资源/停止条件，并按当前授权继续。三文件81新增/8删除（含53行测试）已commit/push `cb01451fbbb01f509bde126029a0cec3d577aedb`，随后树clean；GPU6启动前11MiB、0%。新v7同容量smoke独立wrapper1025349、资源observer1025350已启动，未启动正式、未用旧smoke权重或任何他人GPU。只读watch等待两个完整轮次，不新建自动化。
7. 16:12首轮完整：train25/32，成功池25episode/75query；固定评估28/32，Step1 checkpoint与learner sidecar实体已出现。第二轮重新采集时GPU占用升至79319MiB、FD1002，不能凭第一轮通过就宣布容量足够。16:16—16:17日志继续进入第二轮SFT，GPU78783MiB/100%，环境worker50993MiB、actor26920MiB；没有所查错误。仍等待第二轮评估/保存/退出。首轮成功率不构成正式学习提升结论。
8. 16:23第二轮SFT更新及权重同步已结束，进入最后16×2评估。第二轮更新阶段GPU约78783MiB，FD1000—1002、无所查错误；不能将尚在执行的末次评估/保存提前记为成功。准备的最终验收仅读取tiny learner计数、TB有限指标、两代checkpoint实体及资源CSV；不重新加载大模型或复跑已通过probe。
9. v7于15:50:53启动、16:29:03正常exit0，16:30 CPU验收通过：两完整外轮，learner计数10/20，两代每代local_shard约10.39GB、full_weights约8.07GB及成功池/计数sidecar均非空；loss0.02321→0.01559、grad0.25044→0.10208均有限。train25/32→22/32，fixed28/32→24/32；TB横轴0/1对应完成轮1/2，不宣称提升。采样显存峰79319MiB＝77.46GiB，主机available最低1348.57GiB、PSI0、FD峰1003、Env RSS峰74.24GiB。第一次验收脚本将原生checkpoint子目录少写一层，已根据实际文件修正为local_shard_checkpoint/checkpoint_rank_0.pt并额外要求model_state_dict/full_weights.pt；这是验收路径修正，未修改或重跑训练。
10. 正式前16:30现场GPU6已回11MiB、无compute，7仍4MiB；/data余498GiB、RAMavailable约1.4TiB。shared Ray321933/322685及Sidney wrapper602620继续存活。正式10代checkpoint预估约172GiB另加replay，与其他任务共享磁盘；当前够本任务预算，但没有承诺覆盖其他任务未来全部checkpoint。准备只发布本BC轻量配置/验收证据与说明，然后原SFT/空池启动正式100；不清理旧产物、不动他人任务。
11. 轻量证据/完整配置/统一wrapper及README已commit/push `1d453fcbbf7fd38d99e07323bf74a5a7c045ce9d`，服务器树clean；该次仅文档/证据，运行源码仍为已测cb01451f。16:32:12 CST正式100轮已启动，独立wrapper1151769/observer1151770，输出`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1`。执行合同精确formal命令，原SFT/空成功池、resume=null，未使用v7权重。16:32:31 actor1152302/env1152306初始化中，Env soft4096生效、wrapper存活且无结束标记/所查错误；尚未完整Step1，短查确认采集开始后即回报。
12. 16:34:31启动确认：首轮Generating Rollout Epochs 0/1，expert_only=True、578036768可训练参数，原SFT norm stats路径正确；wrapper存活、无结束标记、所查fatal/OOM/Traceback/原生分配错误为空。启动期GPU6约18.43GiB（不是运行峰值）、RAMavailable约1.4TiB、PSI0，/data余496GiB；GPU7仍4MiB，原Ray与Sidney wrapper继续存活。证据`U10_FORMAL_STARTUP_20260905.json`；只读启动watch已自然结束，资源CSV observer随本run继续记录，不建立主动轮询/新自动化。按“开始就回来报告”结束本轮，不等待正式Step1或宣称已有学习提升。

## 本轮交付结论

正式已启动，GPU6、100轮、train32×1、micro32/global1024/U10、LR2.5e-5、M4、expert-only、增强off、示范混合0；固定eval16×2每5轮、save每10轮。保持合同其余全部参数。v7两轮闭环通过，但77.46/79.65GiB峰值余量有限；后续需关注原生长期稳定性和共享磁盘保存增长，不据smoke承诺100轮必成功。下一窗口先刷新现场，不重复启动，不动GPU7/Sidney/shared Ray；DVAC未实施。
