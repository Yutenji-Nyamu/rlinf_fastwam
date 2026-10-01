# 深圳3 Wan Goal 监控修复与重启

2026-10-01。用户重新授权：深入定位、修复并尝试稳定训练。原 r5 失败证据保留；本轮只修运行监控与资源归还，不改 GRPO、seed、奖励过滤、模型输入或训练参数。

## 当前判断与证据

- r5 完成 4 轮真实更新，随后内层监控记录 `AssertionError()`、`exit_code=null`；清理 SIGTERM 在后。不是已取得的训练 driver 异常退出码。
- 原监控只存 repr，没有 traceback，故不能确定原断言 PID 或根因。
- `w103` 深圳3 CPU 基线：kernel `5.15.0-50-generic`，2,000 个自有短命子进程，旧 identity 没有复现断言。
- `w104` 自有子进程目录 FD 的实际属性：活着时 UID=20001；退出且被回收后，同一 FD 的 inode UID=0。这验证了 proc 元数据随生命周期变化；不是 r5 原 traceback 的直接复现。
- 已确认代码缺陷：身份读取分散在多个路径，目录 UID 断言可把动态扫描变化升级成整体故障；cleanup 重新读取 current 后传给 signal，丢失原注册 start/boot；finally 再次 scan 可掩盖原错误并跳过 cleanup。

## 最小修复

1. 打开 `/proc/pid` 目录 FD，按相对路径读取同一代进程；前后 status 的四个 UID、stat start 复核。stat/status 用 bytes 解析，兼容非 UTF8 进程名。
2. 明确消失或不同代际为不匹配；同代 UID 变化、持续不一致或注册身份不可读时拒绝证明释放，不把未知状态当作停止。
3. 信号保留原注册 UID/start/boot，继续使用 pidfd 和发送前复核；只记录实际发送的动作。
4. 先持久化已知 root catalog，原监控 traceback、最终扫描和 cleanup 监控错误分别保存。独立 cleanup 和 GPU 释放核查仍是归还前提。
5. 新 owner 使用明确的 WM→RLT 直接分支，跳过 Dojo 评测，无竞争 guard/restorer；旧 v5、旧结果和回执不重放。

## 原训练配置与本轮执行

只用深圳3物理4–7，独立 Ray。原 SFT、头图＋腕图 mask、N64/G8/R8/L320/C8、global2048/micro128、H10、5步去噪、1000 runner epochs/save40不变。OFT/π05 smoke验收复用，不额外跑模型smoke。r5无正式CP，修复后是从原SFT重新开始的新尝试，不声称接着r5的优化器续训；r5的4轮记录单列保留。

准备、CPU检查全部通过后才暂停原四RLT；新cycle绑定当时最新完整RLT checkpoint。WM正常结束或失败时，由唯一owner核清释放并原任务/方法/seed累计3000预算恢复。最新资源安排：深圳1原RLT继续；另一窗口EXPO借深圳2物理4–7，由对方唯一owner停止并归还原RLT；WM借深圳3物理4–7。

验收分层：CPU身份/清理检查通过；真实训练超过原失败点且有有效更新；第40轮完整checkpoint及有限参数；之后按原1000轮持续训练。启动和首轮不等同长期稳定完成。

细日志：`local_logs/wan-goal-20261001/steps/w100-*`起；后续实际执行结果在本页和WAN_GOAL_RUNLOG追加。
## 12:53：新尝试已唯一启动

- w108服务器CPU验收通过：16项身份/清理检查＋2项真实owner控制流fixture；无CUDA初始化，无实际GPU训练或RLT启停。回执SHA49b42ca7e94a471a19b734b3749c989365db4b70004e70c2b3d2bb5c1b9c66b8。
- w106新identity对2,000个短命进程无异常；w107核固定d34d4c3已包含官方reset合并db66ac5，不改环境后端。
- w110 12:52准备成功：新v6 ready.return_rlt_direct=true，原Dojo2313回合保留，新cycle绑定原四任务完整CP125。
- w111 12:53:23唯一launch退出0：新的v6 owner、独立r6输出，原源码/两YAML SHA检查通过。w112 12:54处于精确STOPPING_CURRENT_FOUR_RLT_RUNS，GPU上下文已释放，但尚未声明真实训练更新或稳定验收。
- 启动命令：原Python执行`/data/chenyiteng/projects/wan-goal-sz3/scripts/prepare_monitor_repair.py --launch`。w111细日志先输出原四任务/CP/配置/命令/输出/资源/停止条件。
- 后续只读入口：`local_scripts/wan_goal_20261001/status_wake.sh`（含启动/退出），`repair_runtime_evidence.sh`（模型加载且有TensorBoard后核原配置、placement、有效更新和CP目录）。不重放w110/w111。

12:55 w114：owner活、RUNNING_WM，真实placement4–7已通过，模型加载中；训练标量尚空。12:57用户再次确认两窗各自推进，另一窗口compact状态仍在SZ2物理3做EXPO准备，无资源冲突。

13:05 w120发布成功：远端f4e54725672728df2f728c40c244643801a7a751，16新增/8修改/0删除。w116跨任务root的两个轻量回执被allowlist拒绝、w117四md EOF检查失败，均未commit；w118精确定位，w120仅补正四md，从已审暂存完成。原失败manifest保留，活的runtime不改。

## 13:18：真实首轮更新通过

w123只读实查：r6完成step0，grad norm=0.5559873、有效mask=1.376953%、优势[-1.6201816,0.5400605]、loss=0.0001563128，均有限；单轮1066.72秒。owner活，已进入下一轮采集，近期无primary error或monitor诊断。四卡约62GiB显存、可用内存约1.71TiB。尚未超过原4轮故障点，也未到原save40，不声称长期稳定或真实LIBERO成功。

继续验收入口和首次checkpoint只读CPU检查见[进展与验收](WAN_GOAL_REPAIR_PROGRESS_20261001.md)。原训练和活的owner源码保持。
