# π0.5 BC／BC＋DVAC／GRPO：09-06上午只读刷新

现场：2026-09-06 **10:01:29—10:01:47 CST**；环境私有内存补查10:03:26。普通账号chenyiteng固定host-key密码认证。三项均继续运行，本轮没有修改训练、运行GPU测试、停止任务、升级依赖、清理文件或干预shared Ray。所有动态数字来自此次服务器读数，不沿用昨晚启动快照。

[交互图（固定评估／采集／优化／资源）](pi05-three-runs-20260906/index.html) · [固定评估PNG](pi05-three-runs-20260906/success.png) · [训练采集PNG](pi05-three-runs-20260906/training.png) · [优化PNG](pi05-three-runs-20260906/optimization.png) · [资源PNG](pi05-three-runs-20260906/resources.png) · [逐轮CSV](pi05-three-runs-20260906/metrics.csv)。

## 1. 三项目前如何

| 实验 | GPU | 完成轮次／目标 | 最新采集成功 | 最近10轮采集均值 | 最新fixed32 | 最新checkpoint |
|---|---|---|---|---|---|---|
| π0.5在线BC | 6 | **44/100**，第45轮采集 | 23/32＝71.88% | 66.56% | Step40 **21/32＝65.63%** | Step40 |
| π0.5 BC＋DVAC | 7 | **42/100**，第43轮更新 | 23/32＝71.88% | 58.44% | Step40 **19/32＝59.38%** | Step40 |
| Sidney π0.5 GRPO | 4/5 | **155/200**，第156轮采集2/4 | 184/256＝71.88% | 66.91% | Step155 **21/32＝65.63%** | Step150 |

三wrapper和相关worker均在，所查运行日志fatal/OOM/Traceback/原生渲染错误等命中0，所有TensorBoard标量均有限。GRPO只对**正在运行的`runtime-resume100-to200`日志**作当前故障判定，原100步的exit0不是当前结束标记。完成轮数同时由Global Step表和`time/step`等TB事件核验；TB raw_step+1是完成轮次，不伪造更新前Step0。

BC/DVAC仍为32条/轮、micro32/global1024/U10；GRPO256条/轮、U2。已完成训练尝试分别1,408、1,344、39,680条；不能按相同Step直接比较交互效率，也不能把不同采样机制的训练成功率视作同一固定评估。

## 2. 有没有提升、DVAC有没有用

- **BC有初步改善，但评估仍波动**：采集前10轮55.94%→最近10轮66.56%；fixed Step5/10/15/20/25/30/35/40依次 **17/14/16/17/16/19/15/21 /32**。Step40为当前最高，尚非连续稳定提高。
- **DVAC同样有改善，但未显示稳定优于BC**：采集前10轮46.25%→最近10轮58.44%；fixed同一组步数为 **11/13/16/17/20/15/18/19 /32**。最新比BC少2条，但最近4次fixed均值BC55.47%、DVAC56.25%，方向随窗口变化；不足以下单点胜负结论。两run首次更新前采集已不同（16/32与9/32），不能把整个曲线间距直接认作方法因果。
- **GRPO后期进入平台／波动区**：最近6次fixed Step130/135/140/145/150/155为 **23/22/19/16/24/21 /32**；尚未超过Step70最好26/32。最近10轮训练66.91%，相较早期41.68%有提高，但不代表持续泛化增益。
- **DVAC确实生效，不是全1空跑**：第42轮新记录权重min/mean/max＝**0.6919/1.0000/1.3189**，std0.08294；全已记录轮次极值0.6451—1.4218，处于授权[0.5,1.5]内。首轮w=1，从后续轮次开始非均匀加权；这是逐动作权重日志，不是奖励／优势。BC和DVAC的FM loss分别0.01008→0.00435、0.00838→0.00371；后者是加权且数据池不同，不能按loss大小判方法优劣。

累计成功episode／query：BC916／3159，DVAC774／2619；池持续增长。32条fixed每一条对应3.125个百分点，目前没有额外多seed复评，本轮也未执行新推理。

## 3. 服务器整体与优先风险

| 项目 | 10:01现场 |
|---|---|
| GPU0 | 其他用户进程约9.67GiB；只读归属元数据，不查看其文件或干预 |
| GPU1/2/3 | 各4MiB，无compute进程，可用 |
| GPU4/5 | GRPO 67.07／67.35GiB；利用率94%／51% |
| GPU6/7 | BC 55.73GiB／DVAC72.88GiB；利用率0%／100%，阶段切换的瞬时读数，不据单次0%判挂住 |
| 本轮历史采样显存峰 | BC73.73、DVAC73.73GiB；GRPO续训4/5为74.64／74.93GiB；卡实际总量79.65GiB |
| CPU | 128逻辑核，load 7.30/7.02/6.88，1秒实测约95% idle，iowait0 |
| RAM | 总约2015.51GiB，可用**664.54GiB**；10:03复核665.09GiB |
| Swap／压力 | 已用5.95GiB，接近小型6GiB swap总量；当前si/so=0、memory/io PSI avg10=0；不代表整夜无压力，BC观察器曾记录memory some avg10峰12.83 |
| /data | 可用**630.13GiB**，82%已用，inode1% |
| /home | 可用**1246.15GiB**，47%已用，inode2% |
| 系统盘 / | 可用222.64GiB，22%已用，inode3% |
| GPU健康 | 当前31—68°C；BC/DVAC采样温度峰71/73°C；GPU1有2次可纠正SRAM ECC，所查8卡不可纠正ECC均0，无row-remap pending/failure |
| shared Ray | 原gcs321933／raylet322685继续，运行约13天；本轮未重启或改配置 |

**优先风险是环境进程主机内存的累积增长，不是当前已OOM。** 两BC环境RSS曲线有明显台阶上涨；10:03补查`smaps_rollup`，PSS和私有匿名内存都接近RSS，因此不是单纯把共享映射重复算大：

| 环境进程 | PID | PSS约GiB | FD数 |
|---|---|---|---|
| BC | 2143761 | **246.39** | 878 |
| DVAC | 2224188 | **192.64** | 878 |
| GRPO rank0 | 603555 | **365.04** | 852 |
| GRPO rank1 | 603565 | **354.88** | 859 |

四个EnvWorker合计PSS约1158.95GiB，主要是各自私有匿名内存；BC/DVAC的成功池在actor侧，不能把EnvWorker数百GiB简单解释为累计成功池。BC/DVAC Env软FD上限4096，当前878，未触及；GRPO FD亦未触及当时实际1024上限。主机仍有约665GiB可用且即时无换页，但不应据当前余量承诺余下全程安全。

**本轮仅确认增长和所在进程，未定位具体Python／C++对象；不直接认定OIDN泄漏，也未添加GC、卸载或重启等修补。** 若继续调查，应优先在现有隔离边界内追环境私有内存增长的来源，不动这些正在运行的任务。

## 4. checkpoint、速度与源码

最新文件逐项stat存在且非空：

- BC Step40：native rank0、full_weights、success_replay、learner，合计**20.76GiB**；已存Step10/20/30/40共80.16GiB。
- DVAC Step40：上述四种＋`dvac.pt`，合计**20.53GiB**；四代共79.55GiB。
- GRPO Step150：native rank0＋rank1＋full_weights，合计**26.85GiB**；现存Step10—150共15代402.73GiB。

未加载大权重、未做恢复测试，因此这里只称必要文件在，不称完整resume已再次验证。无新清理；成功数据每轮追加与每10轮checkpoint分开计。

近期10轮平均：BC14.08分钟/轮（采集5.41、更新7.44分钟）；DVAC13.96（采集5.29、更新7.49）；GRPO22.51（采集21.00分钟、更新24.24秒）。整轮均值含该窗口已有的定期评估／保存，嵌套timer不重复累加。按最近速度及最后完整轮的时间外推、**仅在无故障和资源增长不改变速度时**：BC约09-06 23:04、DVAC23:24、GRPO09-07 02:38结束，不是保证，也未建立新监控。

三树本次Git status均clean，HEAD分别：

| 树 | 当前HEAD | 原wrapper |
|---|---|---|
| pi05-online-bc | `6a93605d91dbfdc321c5602108ccca5e2f044001` | 2143105 |
| pi05-online-bc-dvac | `776ebc987cd7891a3d80b868a6353e6e4554b229` | 2223376 |
| sidney-pi05-current-rlinf | `81be3193d91fe9950a3fc1bdedd14063a85e72d8` | 602620 |

这些是当前Git树HEAD，包含后续轻量证据提交；不把它们混称运行启动时源码hash。运行runtime/source-head及精确路径存于原始JSON。

## 5. 证据与复现

- [原始服务器刷新JSON](PI05_THREE_RUNS_READONLY_REFRESH_20260906.json)：精确run/runtime、PID/proc、三树Git、完整TB标量、checkpoint文件metadata、资源观察器抽样及真实采样极值、整机指标和GPU ECC。
- [内存私有占用补查JSON](PI05_MEMORY_FOLLOWUP_20260906.json)：10:03四个本人EnvWorker的smaps_rollup/status/FD，未访问其他用户进程内存。
- [派生摘要](pi05-three-runs-20260906/summary.json)：统计窗、均值、实际权重和条件ETA。
- 只读脚本：`local_scripts/remote_commands/sz_pi05_three_runs_readonly_20260906.sh`、`sz_pi05_memory_followup_20260906.sh`；Windows经已有`pi05_bc_execute_20260905.py command --file ... --output ...`、固定主机key与getpass执行。只通过stdin执行读取，未上传文件／改远端树。
- 制图：`local_scripts/render_pi05_three_runs_20260906.py`只读当前JSON，生成四PNG／离线交互HTML／CSV。按资源文件路径和timestamp去重（原始JSON的BC资源键包含同路径重复读取，渲染与统计不累加），采样峰值来自全量行而非下采样曲线。
- 本轮边界：服务器只读；本地仅新增读数、图和报告并更新入口；不push任何远端Git、不新建自动化、不接管GRPO既有heartbeat。
