# 深圳 π0 / Fast-WAM：首批真实 DVAC 分解结果

日期：2026-08-22--23。首批统计/图片根：
[`evidence/dvac-analysis-p1-fixed64-20260822`](evidence/dvac-analysis-p1-fixed64-20260822)；独立phase复核根：
[`evidence/dvac-analysis-fastwam-move-stapler-phase-20260823`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823)；
最终五source统一根：
[`evidence/dvac-analysis-all-four-tasks-20260823`](evidence/dvac-analysis-all-four-tasks-20260823)。逐命令账：
[`evidence/15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md`](evidence/15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md)。

## 1. 数据与证据边界

- π0 official fixed-64：`adjust_bottle`，42 success / 22 failure，256 queries，H=50、M=4。
- official Fast-WAM P1：`adjust_bottle`，16 success / 0 failure，80 queries，H=32、实际执行C=24、M=10。
- official Fast-WAM P2：`move_stapler_pad=11/16`、`turn_switch=10/16`、
  `pick_diverse_bottles=12/16`；三任务共423 queries。Fast-WAM四任务总计49 success / 15 failure、
  503 queries、1,509 query PNG与64个完整MP4。
- 最终五source统一主统计使用`success_before=false`的717 queries；其中首版π0 + Fast-WAM adjust
  两source切片为294 queries。π0成功后的42 queries只保留在all-query descriptive伴随表。
  Position/MAD按policy、checkpoint、task、run、cohort、L、h分别估计。
- 本结果是观察性信号分析。DVAC是去噪路径稳定性proxy，不是校准不确定性；跨模型不比较raw V/y。

## 2. 信号怎么读

共同定义：

$$
V_L(q,h)=\sum_d\operatorname{Var}_{i\in\text{last }L}z_i(q,h,d),\qquad
y(q,h)=\log(V_L(q,h)+10^{-12}).
$$

### 2.1 两个大通道

$$
y(q,h)=b_h+r(q,h),\qquad R(q,h)=\frac{r(q,h)}{s_h}.
$$

- `b_h`：同一个run/cohort中future位置h通常有多高，是固定chunk位置通道。
- `r`：当前query在同一个h上偏离通常值多少。
- `R`：再除以该h自身的MAD尺度，得到可比较的标准化偏离。

### 2.2 四项记账

原始尺度：

$$
y=\mu+P_h+S_{raw}(q)+I_{raw}(q,h).
$$

标准化尺度：

$$
R=S_{std}(q)+I_{std}(q,h).
$$

- `S(q)`：一个query的整条chunk共同升降，更像当前state/task-stage信号。
- `I(q,h)`：扣掉该query整体升降后，某个future action额外突出，更像chunk内部局部形状。
- 不能写`y=b+S_std+I_std`，因为前后量纲不同。本批三条正确恒等式的最大误差均小于`1.8e-15`。

## 3. 主要真实结果

### 3.1 固定future位置效应确实存在

π0的`b_h`不是平的：L=3时h0--4均值约`-4.884`，h45--49约`-3.966`；最低h0=`-4.943`，
最高h49=`-3.795`，跨度`1.148 log units`。曲线先升、20--33附近回落/平台，再在chunk尾部明显上升。
因此直接拿raw DVAC排序会系统性偏向后部future action；先扣`b_h`是必要control，而不是美化信号。

Fast-WAM的固定位置曲线更锯齿、非单调：L=3跨度`0.743 log units`，最低h2、最高h14；这进一步说明
不应强拟合直线。两侧所有L/h的MAD floor计数都是0，标准化不是由人为floor主导。

对应图：[`π0 position`](evidence/dvac-analysis-p1-fixed64-20260822/figures/pi0__adjust_bottle__run-pi0-adjust_bottle-fixed64-800baf__cohort/01_position_profile.png)、
[`Fast-WAM position`](evidence/dvac-analysis-p1-fixed64-20260822/figures/fastwam__adjust_bottle__run-fastwam-adjust_bottle-p1-16ep-c6__co/01_position_profile.png)。

### 3.2 π0中，query-wide的S比局部I更明显地区分成功/失败

episode-first、pre-success描述性bootstrap结果：

| 指标 | success n=42 | failure n=22 | success - failure | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| mean `S_std` | -0.135 | 0.301 | **-0.436** | **[-0.630, -0.242]** |
| mean `abs(I_std)` | 0.572 | 0.592 | -0.020 | [-0.055, 0.015] |
| mean `abs(R_std)` | 0.753 | 0.869 | -0.116 | [-0.223, -0.017] |

当前批次最清楚的现象不是“失败时某几个h普遍更尖”，而是失败episode的整条chunk共同偏高：
`S_std`差异明显；`abs(I_std)`差异很小且区间跨0。`abs(R_std)`也更高，但主要包含S的贡献。
这是相关性，不证明S造成失败，也还不能直接当RL权重。

对应图：[`π0 outcome summary`](evidence/dvac-analysis-p1-fixed64-20260822/figures/pi0__adjust_bottle__run-pi0-adjust_bottle-fixed64-800baf__cohort/05_outcome_summary.png)。

### 3.3 S与执行阶段存在首批可见对应

π0典型成功episode由“各outcome内动作长度最接近中位数、再按seed/index固定tie-break”选出；不是按
DVAC挑图。它的query-boundary帧与`S_std`为：初始水平瓶`+0.07`，接触/抓取`+0.50`，已经抬起并
旋转`-1.04`，瓶子竖直`-1.34`。典型失败则为：`+0.84、+0.43、+0.41、+1.25`，后两帧瓶子仍未
完成目标。全体轨迹同样从action slot100开始明显分叉，而不是只靠两个代表样本得出。

对应图：[`π0 S timeline`](evidence/dvac-analysis-p1-fixed64-20260822/figures/pi0__adjust_bottle__run-pi0-adjust_bottle-fixed64-800baf__cohort/03_S_timeline.png)、
[`typical success frames`](evidence/dvac-analysis-p1-fixed64-20260822/query_frame_strips/pi0__adjust_bottle__run-pi0-adjust_bottle-fi__ep000033_reset100100005.png)、
[`typical failure frames`](evidence/dvac-analysis-p1-fixed64-20260822/query_frame_strips/pi0__adjust_bottle__run-pi0-adjust_bottle-fi__ep000050_reset100100017.png)。

Fast-WAM adjust-bottle的16个成功episode呈稳定U形：按query index汇总的`S_std`均值约为
`+0.50、-0.58、-0.45、+0.34、+1.00`。代表视频对应初始接近、抓取/中段操作、最后调整/成功；
这支持“S含task-stage结构”的首批观察，但该任务没有失败样本，且本首版phase尚未盲标，暂不做
MOVING/OPERATING统计或成功/失败结论。

对应图：[`Fast-WAM S timeline`](evidence/dvac-analysis-p1-fixed64-20260822/figures/fastwam__adjust_bottle__run-fastwam-adjust_bottle-p1-16ep-c6__co/03_S_timeline.png)、
[`query frames`](evidence/dvac-analysis-p1-fixed64-20260822/query_frame_strips/fastwam__adjust_bottle__run-fastwam-adjust_b__episode0002_reset1.png)、
[`action storyboard`](evidence/dvac-analysis-p1-fixed64-20260822/storyboards/fastwam__adjust_bottle__run-fastwam-adjust_b__episode0002_reset1.png)。

### 3.4 I保留了chunk内部形状，但当前不支持简单top-k结论

`I_std(q,h)`热图显示同一个query内确有局部正负结构；Fast-WAM还清楚标出`h>=24`为未执行future tail。
但π0 outcome汇总中`abs(I_std)`几乎不分成功/失败，因此不能据本批就说“挑最大I的action训练”有效。
它更适合下一步与动作phase、contact或transition对齐后再判断。

对应图：[`π0 I heatmap`](evidence/dvac-analysis-p1-fixed64-20260822/figures/pi0__adjust_bottle__run-pi0-adjust_bottle-fixed64-800baf__cohort/04_I_representative_heatmap.png)、
[`Fast-WAM I heatmap`](evidence/dvac-analysis-p1-fixed64-20260822/figures/fastwam__adjust_bottle__run-fastwam-adjust_bottle-p1-16ep-c6__co/04_I_representative_heatmap.png)。

### 3.5 `move_stapler_pad`复现了S的outcome差异，并给出首个独立phase对齐

在没有查看本任务DVAC之前，先从raw视频固定选择动作长度接近各outcome中位数的success ep4和failure
ep6，并只凭12帧contact sheet标注approach、grasp/contact、transport、place/release与verify。随后才运行
DVAC分析；failure q9--16因视频语义不够确定，始终保持`UNLABELED`，没有用数值反补phase。

全16 episode的episode-first outcome结果如下：

| 指标 | success n=11 | failure n=5 | success - failure | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| mean `S_std` | 0.054 | 0.653 | **-0.599** | **[-1.015, -0.123]** |
| mean `abs(I_std)` | 0.661 | 0.803 | -0.142 | [-0.320, 0.070] |
| mean `abs(R_std)` | 0.848 | 1.184 | -0.335 | [-0.654, 0.044] |

因此第二个模型/任务上仍是query-wide `S_std`的成功/失败差最清楚；`I`与`R`方向一致，但区间跨0。
这提升了“S可能描述当前执行状态/困难程度”的可信度，仍只是固定seed块上的观察性证据。

phase统计只来自两条预先选定代表episode，不是16条全体的phase估计。其`S_std / abs(I_std)`依次为：
approach `-0.263 / 0.560`、grasp `-0.382 / 0.522`、transport `0.229 / 0.581`、place
`0.663 / 0.888`、verify `0.280 / 0.478`（verify只有1条）。从这两条看，S在抓取阶段较低、运输后转正、
放置阶段最高；失败代表轨迹在后段出现多次很高的S尖峰。但样本只有2条，现阶段只能叫phase-aligned
case study，不能叫总体阶段规律。

对应图：[`move-stapler outcome`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823/figures/fastwam__move_stapler_pad__run-fastwam-move_stapler_pad-p2-16ep_/05_outcome_summary.png)、
[`S timeline`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823/figures/fastwam__move_stapler_pad__run-fastwam-move_stapler_pad-p2-16ep_/03_S_timeline.png)、
[`independent phase timeline`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823/figures/fastwam__move_stapler_pad__run-fastwam-move_stapler_pad-p2-16ep_/03b_representative_S_phase_timeline.png)、
[`success storyboard`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823/storyboards/fastwam__move_stapler_pad__run-fastwam-move___episode0004_reset3.png)、
[`failure storyboard`](evidence/dvac-analysis-fastwam-move-stapler-phase-20260823/storyboards/fastwam__move_stapler_pad__run-fastwam-move___episode0006_reset5.png)。

### 3.6 四任务统一结果：S最有outcome差异，但正负方向由任务语义决定

统一分析在五个相互隔离的run/cohort内各自估计`b_h/s_h`，不会把不同模型或任务的raw尺度混在一起。
episode-first的主要标准化差值如下（均为success - failure）：

| policy / task | success / failure | `S_std`差（95% CI） | `abs(I_std)`差（95% CI） | `abs(R_std)`差（95% CI） |
|---|---:|---:|---:|---:|
| π0 / adjust bottle | 42 / 22 | **-0.436 [-0.630,-0.242]** | -0.020 [-0.055,0.015] | -0.116 [-0.223,-0.017] |
| Fast-WAM / move stapler | 11 / 5 | **-0.599 [-1.015,-0.123]** | -0.142 [-0.320,0.070] | -0.335 [-0.654,0.044] |
| Fast-WAM / turn switch | 10 / 6 | **+0.564 [0.119,0.979]** | +0.096 [-0.001,0.190] | **+0.462 [0.205,0.738]** |
| Fast-WAM / pick diverse bottles | 12 / 4 | **-0.541 [-0.965,-0.202]** | +0.039 [-0.143,0.181] | -0.071 [-0.459,0.220] |

最重要的修正是：`S`不是“越大越失败”的通用标尺。π0 adjust、move-stapler和pick-bottles里失败更高，
但turn-switch恰好成功更高。视频能解释这个反号：典型turn-switch成功轨迹真正接近并操作开关，操作query
出现很高的S；典型失败轨迹长期没有进入有效操作阶段，S反而低。因此当前最合适的解释是：`S`主要携带
query-wide执行状态/阶段/困难组合信息，而不是跨任务同号的校准不确定性。直接把raw S当通用RL权重还太早；
应先做task/phase-matched control。

`I`在四个有成功/失败对照的块里差异都明显弱于S，支持“局部future形状存在，但outcome主信号目前更多
来自整条chunk共同升降”。`R=S+I`有时跟随S显著，例如turn-switch；不能把R的效果误归因给I。

对应图：[`跨任务outcome与position汇总`](evidence/dvac-analysis-all-four-tasks-20260823/figures/cross_task_outcome_and_position_summary.png)、
[`turn-switch success storyboard`](evidence/dvac-analysis-all-four-tasks-20260823/storyboards/fastwam__turn_switch__run-fastwam-turn_switc__episode0009_reset8.png)、
[`turn-switch failure storyboard`](evidence/dvac-analysis-all-four-tasks-20260823/storyboards/fastwam__turn_switch__run-fastwam-turn_switc__episode0001_reset0.png)、
[`pick-bottles failure storyboard`](evidence/dvac-analysis-all-four-tasks-20260823/storyboards/fastwam__pick_diverse_bottles__run-fastwam-p__episode0005_reset4.png)。

## 4. 稳健性与下一批

- π0 query ranking对L=2/3/4的Spearman相关为`0.855、0.666、0.812`；有明显共同信息，但L2和L4并非
  等价。Fast-WAM L3/L5为`0.969`，对tail长度很稳定。
- 四个Fast-WAM任务均已自然完成；L3/L5 query ranking相关依次为adjust `0.969`、move `0.959`、
  turn `0.946`、pick `0.980`，主要排序不依赖单一tail长度。
- 所有run的L3固定位置曲线都有实质跨度：π0 adjust `1.148`，Fast-WAM adjust/move/turn/pick分别
  `0.743/0.597/0.564/1.775 log units`；所有位置MAD floor计数仍为0。pick-bottles的固定position成分
  尤其大，进一步证明raw DVAC不能直接跨h排序。
- 下一批最有信息量的工作不是继续无差别加episode，而是先对turn-switch和pick-bottles代表视频独立
  phase标注，再检验“S反号是否由进入有效操作阶段”解释；仍须先标视频、后读DVAC。

完整表：[`outcome_summary.csv`](evidence/dvac-analysis-p1-fixed64-20260822/outcome_summary.csv)、
[`L_sensitivity.csv`](evidence/dvac-analysis-p1-fixed64-20260822/L_sensitivity.csv)、
[`analysis_summary.json`](evidence/dvac-analysis-p1-fixed64-20260822/analysis_summary.json)。

最终五source完整表：
[`outcome_summary.csv`](evidence/dvac-analysis-all-four-tasks-20260823/outcome_summary.csv)、
[`L_sensitivity.csv`](evidence/dvac-analysis-all-four-tasks-20260823/L_sensitivity.csv)、
[`analysis_summary.json`](evidence/dvac-analysis-all-four-tasks-20260823/analysis_summary.json)。
