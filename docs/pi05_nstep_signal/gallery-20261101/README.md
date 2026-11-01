# N1/N5/N10同轨迹绘图材料

41任务、1312有效回合、728成功；9任务报错/超时缺测。精选80条，79成功、1失败；每任务1–2条。原始Sidney多任务π0.5，N10执行、N1/N5旁算，H50/C50，同观察、同四份噪声。

## 下载与阅读

[三个公开附件](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/nstep-signal-20261101)：

- `nstep-1101-figures.tar.gz`：80条曲线SVG/PNG/PDF、6帧联系表、网页、series JSON及全体选择统计。
- `nstep-1101-all-curves.tar.gz`：全部1312局的逐动作20路信号CSV、原提交前缀signals.csv、配置/种子/指令/成绩、控制器帧映射。统计分析用这一包，不能只用精选。
- `nstep-1101-selected-raw.tar.gz`：80条精选每次查询的原始动作、四份初始噪声、内部x/z估计、三相机观察、控制器路径与逐步记录；可重新算信号。

三个附件解压到同一目录即可组合使用；原始完整视频和全部批次大NPZ共约9.7GiB留服务器，不在公开附件。精选完整H264视频已另存本地图册。HTML中的视频入口在本地完整目录可用，下载的公开包不包含完整视频。

[本地图册](http://127.0.0.1:8194/nstep1101/) · [先看调瓶子](http://127.0.0.1:8194/nstep1101/?q=adjust_bottle) · [逐任务](coverage.csv) · [80条索引](selection.csv)

## 信号与作图口径

每条同轨迹展示：同噪声N1/N10、N1/N5、N5/N10差异；三种步数各自的K4绝对/相对分散度；N5/N10内部DV。K3版本也留CSV供复核。N1无内部多步DV，保留缺测，不能补0。K4为动作标准差类信号，不等于概率熵。

图只用控制器投影覆盖的实际执行前缀，提前成功后的未执行位置不画；纵轴不跨信号强行对齐，无平滑。每块选择运动较多的手臂，把模型动作位置投影到真实控制路径，再找最近视频帧。曲线A–F与6张真实帧一一对应，并保留映射误差。

选例继承10/20成功U/K4图册：成功优先、N1/N10峰形清晰度及映射质量；不以期望相关系数筛例。`place_dual_shoes`本次无成功，保留一条失败；`scan_object`只有1条成功，保留1条；其余39任务各2条成功。

N5与N10的DV分别覆盖全部5步和最后5步，内部时间区间不同，不能只比数值大小解释强弱。N10也不是动作真值。调瓶子的N1分散度较低而与N10差距明显，是“低分散不自动等于正确”的例子，尚非全体统计结论。

## 复现与留存

复用[旧图册](../../pi05_u_video/noise-gallery-20261020/README.md)的控制器投影、完整视频及成功优先选例流程；本次增量脚本位于`tools/pi05_nstep_signal/{export_gallery,plot_gallery,build_gallery}.py`，纯CPU离线操作，无新GPU任务。

服务器原始：`/home/chenyiteng/results/pi05-nstep-video/20261031-v1`；导出：`/data/chenyiteng/exports/nstep-gallery-20261101`。本地图册：`E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/nstep1101`。

9项缺测：blocks_ranking_size、open_laptop、open_microwave、place_bread_basket、place_fan、place_object_scale、put_bottles_dustbin、put_object_cabinet、stack_blocks_three。
