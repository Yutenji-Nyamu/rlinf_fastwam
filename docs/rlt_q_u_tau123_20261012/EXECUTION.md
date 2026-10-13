# RLT Q-U τ1/2/3：正式训练与近期 Git 同步

2026-10-12，深圳。用户授权停止DSRL τ4，并将SZ2 GPU6/7、SZ3 GPU5用于Q-U温度搜索；Attn修复后继续，逐卡RLT候补保持。

## 当前结果

10-13最新：Q-U三组均按用户结论停止；τ2/3完整恢复点为CP750/525。最新主日志见closeout1013/tau2、tau3；以下迁移和启动记录保留作来源。

10-13更新：用户判定τ1已得出结论，主动停止于351轮，保留完整CP350（66065更新）和最后R350固定评估20/20。driver/namespace/C/G清场通过，未追加评估；配置、主要日志和TensorBoard事件见[closeout1013](closeout1013)，压缩约0.71MiB。

为新EXPO组成同机双卡，τ3从SZ3 GPU5迁至SZ2 GPU6，完整CP150含19200更新及回放。实配只调整物理卡、进程环境、输出和恢复路径；算法逐项相同，源码仍b1d2d8d、无trick、总800轮。旧机未写入断点的轮次需从CP150之后重采，旧产物保留；新机00:38已恢复真实采集及actor/critic学习，更新数到22400。迁移实配与恢复证据见`closeout1013/tau3-sz2-g6-resolved.yaml`和`tau3-resume-proof.json`。τ2继续SZ2 GPU7，未中断。两卡均保留原逐卡RLT候补。

下表是10-12的启动记录，当前路由以上述更新为准。

| 位置 | 正式实验 | 启动验收 | 原RLT候补 |
|---|---|---|---|
| SZ2 GPU6 | Q-U τ1，无trick，800轮 | 至少2轮；回放160，真实U信号非零 | 移走扑克牌Clean，CP400 |
| SZ2 GPU7 | Q-U τ2，无trick，800轮 | 至少2轮；回放160，真实U信号非零 | 移走扑克牌DVCA，CP425 |
| SZ3 GPU5 | Q-U τ3，无trick，800轮 | 首轮完成；回放80、U信号均值0.015 | 敲块DVCA，CP350 |

以上均在预采集，尚无策略更新；不把启动验收当学习收益，也不使用首轮进度条ETA估算整套训练。

## 复用依据与差量

- 找到旧正式Q-U **τ2.5、无trick**，以及Q-Norm对照；未找到Q-U τ1/2/3正式记录。旧日志补在 `history/primary-logs.tar.gz`。
- 三组运行源码均为已正式跑通的 `b1d2d8d553fa057435e0169846ec99c3a2c48c28`。SZ2直接复用原目录；SZ3使用同一Git对象导出的源码。当前发布分支包含后来的可选trick模块，但**实际执行锁定旧无trick源码**。
- 实配继承 `rlt4-q-u-g6-formal-800-1006-v1`：adjust_bottle、同一Stage1 CP2000、N4、200物理动作、B512/微批256、UTD5、预采集10000、初始critic更新15000、单轮上限800次更新；每25轮固定20条评估及保存，正式800轮。
- U来自同输入/同噪声的ODE10与ODE5；只重加权actor Q项。Critic/TD、BC项、采样规则和原BC/Q权重日程不改。无chunk dropout；U alpha恒为1，无U退火。
- 算法配置逐项对比通过：唯一变化为 `algorithm.rlt_q_weighting.temperature: 2.5 → 1/2/3`。日志、namespace、源码路径及物理计算/图形绑定随部署调整；完整配置与method-diff在各机evidence中。
- SZ3原来缺该任务的Stage1权重，已从SZ2经固定host-key SSH复制原10,018,009,814字节文件；未换模型或使用其他任务的断点。

## 逐卡接续

唯一owner仍为 `/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/owner.py`，本轮没有修改owner/driver源码或新增常驻看护。

本次事务为SZ2/3 `/data/chenyiteng/deployment-20261012/rlt-q-u-tau123-v1`。先暂停对应派发、精确结束旧任务并核对namespace及C/G释放，再挂正式priority和fresh fallback runtime。旧RLT完整断点保留，正常结束或失败释放后同卡接RLT。启动命令仍由既有owner执行 `PYTHON S/ops/driver.py --plan S/plan.json --request T/priority-gX.json`；旧事务不得重放。

三机12个候补均有完整断点和配置，owner存活；SZ3 6/7的Attn两批完成后已经自动接上提锅RLT CP200。三机0–3卡无计算/图形上下文，本轮未动其他用户、共享Ray或其他训练。

## 本轮已推的近期产物

| 实验 | 已发布内容 | Git提交 |
|---|---|---|
| DSRL U四组；WM/GeoWAM收束 | τ4主动停止于258轮，exit143且清场正常；τ1/2完整400轮末评12/12、10/12。主要日志、TB事件、全量scalar CSV、实配与收束回执共90项，约3.95MB压缩包 | [345a5e3](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/345a5e3329e37c2b933394971314fef16aa674e4) |
| EXPO FM-U三组 | 完整主要driver日志、events.jsonl、配置、状态与真实评估回执109项，约2.88MB压缩包；训练未改 | [a1b3011](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/a1b301179a128390b353bec95b90f3eff4f9d329) |
| Attn | 修复缺失进度文件导致整路退出；移除旧扩量门槛；7项CPU检查通过。两批完整验收，1408/1600条、1400个不同任务/实际seed组合；自动归还RLT。原生失败/暂缓192条仍未解决 | [4ada309](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/4ada30958732f1764b6b8a4a026d83916a41b4b6) |
| 旧RLT-BC六组 | 已有停止轮次、配置、曲线和回执，本轮未重复提交 | [f1df47a](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/f1df47a2f5b786fcbe71dc5589b2476d210333db) |

之前的Git主要覆盖方法、启动与部分早期曲线，后续日志确有缺口；本轮补齐上述明确范围。日志按原文件名保存在压缩包，文件索引与关键结果可直接阅读。模型权重、回放、原始Q/K、视频与凭据不上传。


## 2026-10-13 remaining Q-U closeout

User concluded Q-U tau2 (SZ2 GPU7) and tau3 (SZ2 GPU6). Both exact drivers stopped; their Ray namespaces and compute/graphics contexts cleared, cleanup_error=null. Latest complete recovery points are CP750 and CP525, respectively; weights/replay remain on the server. Primary TensorBoard files, runtime logs/configs and completion/stop evidence are under `closeout1013/tau2` and `tau3`. Tau3 earlier SZ3 progress and CP150 migration evidence remain in the prior closeout packet. These two GPUs now host GRPO U tau3; original per-card RLT fallbacks wait behind it.
