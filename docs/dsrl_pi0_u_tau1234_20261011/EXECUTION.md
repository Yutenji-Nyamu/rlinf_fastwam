# π0 DSRL U τ1/2/3/4：无 trick，300 轮

2026-10-11，深圳。按用户指令收束 GeoWAM 与 WM，用原四卡启动四组正式实验；先撤对应 RLT，再放正式任务，最后逐卡挂回候补。

## 本轮结果

| 位置 | 正式任务 | RLT 保留恢复点 |
|---|---|---|
| SZ2 GPU6 | U τ1，无 trick，R300 | CP350 |
| SZ2 GPU7 | U τ2，无 trick，R300 | CP350 |
| SZ3 GPU4 | U τ3，无 trick，R300 | CP325 |
| SZ3 GPU5 | U τ4，无 trick，R300 | CP350 |

13:42派发，13:53四组均完成真实采集轮，回放已增加；尚处 warmup500，不能称为学习改善。计算与图形均在各自指定物理卡。四项状态为 PRIORITY_RUNNING，原有每卡 RLT 等正式 driver/namespace/C/G 全部释放后接替。SZ1四卡、SZ2 EXPO4/5、SZ3 Attn6/7保持；没有新增常驻看护。

## 继承与唯一方法变化

- 复用已验收的单卡 π0 源码和实际配置；7个核心文件 SHA256 与发布基线 `ff64209` 完全一致，见 `provenance.json`。SZ2复制实际运行源文件，包含原 worktree 未入旧索引的 chunk-controls 模块；未重写算法。
- 原任务 `adjust_bottle`，N4训练/评估，H50/C20，主ODE4、U侧ODE2，B256/微批256，replay25000、warmup500、UTD20；相同 π0 初始权重、norm stats、随机种子、任务重置路径。
- 仅设置 U temperature=1/2/3/4，关闭 chunk-dropout 与 alpha-schedule；U强度系数保持1。预算200→300轮，是用户明确指定的额外变化。每13轮固定12条评估，每65轮及结束保存。
- 四组均从原始 π0 底座重新训练，非旧CP200续训。路径、namespace、物理卡和对应图形配置随部署更换；逐叶差异在各 `configs/tau*/diff-*.json`。
- 复用成功旧版图形绑定代码；SZ3仅为新物理4/5生成本用户、仅匹配各自marker的profile，并登记到候补既有校验。CPU预检首次因自有manifest权限失败，改600后通过，未产生失败正式训练。未加方法 smoke。

## 旧实验收束

- **GeoWAM**：SZ2原6/7已自然完成40000/40000，退出0，完整 `step_0040000` 保留。其后自动接上的两路RLT已各保存完整CP350；精确退出和清场回执见 provenance。
- **WM**：SZ3原4/5完成RL41，按指令中止R42采样。真实32条评估 R10/20/30/40 为10/5/1/2次成功，总体下降，末点略回升；R31–41想象成功率和策略梯度均0。最新完整策略CP40、WM040、RM040、日志与原生采集全部保留。
- WM根目录 `/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/wm-cycle-20261009-v2/run-retry3`；策略 `rl040/rl040/checkpoints/global_step_40`；WM `wm040/train/checkpoints/weights/step_001790.pt`；RM `rm040/train/best.pt`，阈值0.7410773634910583。final中的 signal15 为此次主动中止。
- GeoWAM根目录 `/data/chenyiteng/projects/geowam-real/runs/train_rgbd72_v1`，最终恢复点 `checkpoints/step_0040000`。本轮没有删除权重。

## 执行入口与证据

- 两机事务：`/data/chenyiteng/deployment-20261011/dsrl-pi0-u-tau1234-v1`，读 prepared/admitted 与 priority-g*.json；统一逐卡队列仍为 `/data/chenyiteng/deployment-20261008/bc-signal-tau-v1`。
- 启动沿用该队列 `ops/driver.py`，具体请求与参数见 priority JSON；每组单卡，达到300轮自然结束，异常退出清场后由同卡RLT接替。
- 输出：`/data/chenyiteng/results/rlinf-dsrl-pi0-single/pi0-u-tau{tau}-notrick-{sz2|sz3}-g{gpu}-r300-1011`。
- `configs` 保存实配、逐叶差异和候补引用；`provenance.json` 保存源文件比对及精确交还/派发回执；`first-round.json` 保存正式首轮与绑定证据；`curves.csv` 保存本次同族曲线原值。
- [单列大图](http://127.0.0.1:8179/tau4-1011/)：DSRL对比历史Clean/U、EXPO同族、WM原生评估与想象指标分开，滑动20、高区分颜色和标记，可勾选/全屏。旧基线非本次同时重新跑的对照。
- 新任务预计约24–30小时，依据旧单卡200轮耗时；warmup速度不能直接外推学习阶段，当前尚无稳定实测ETA。
