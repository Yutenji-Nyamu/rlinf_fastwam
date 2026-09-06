# π0 两卡 PPO pair：2026-09-01 11:30 CST 只读刷新

- Control：完整 Step 58，raw / MA5 / MA10=`93.75% / 91.41% / 91.68%`。
- PPO-DVAC Action-Adv `[0.5,1.5]`：完整 Step 56，raw / MA5 / MA10=
  `91.80% / 93.83% / 92.30%`。
- 共同 Step 1--56：训练成功率均值 Control / DVAC=`86.77% / 86.57%`，差值
  `DVAC-Control=-0.20 pp`；最近5/10步差值为`+3.13/+0.51 pp`。
- Step 5--55 的11次 fixed-32累计：Control=`328/352`，DVAC=`323/352`；最新Step55分别为
  `30/32`与`31/32`。训练近期曲线略偏DVAC，但固定评估累计尚未领先。
- 两条wrapper与shared Ray均存活，下载日志中未命中 traceback、OOM、worker death 或 nonfinite。
- 快照时 GPU4--7 分配给本 pair；GPU0--3无计算负载。主机 `MemAvailable`约157.7 GiB。

主图：[`01_ppo_control_vs_dvac_raw_ma5_ma10_fixed32.png`](01_ppo_control_vs_dvac_raw_ma5_ma10_fixed32.png)。
原始输入与可重画数据位于`raw/`与`curves.csv`；`live_status.txt`保存同一时刻只读现场。
