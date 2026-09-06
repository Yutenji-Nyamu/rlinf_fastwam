# DVAC 细粒度时间轴完整图册

本图册包含本轮最终 77 张图。case 图坚持“一张图只讲一个信号”：每张上方的 A/B/C… 画面与下方曲线同字母点精确对应；Fast-WAM 是真实 action/frame 对齐，π0 仅是 query-boundary 三相机输入。

## 一、读图参考（7 张）

### 四个任务

![四个任务](figures/reference_00_task_overview.png)

### 八个量的记账链

![信号词典](figures/reference_01_signal_dictionary.png)

### 去噪 tail 窗口

![tail窗口](figures/reference_02_denoising_tail_window.png)

### query、future h 与环境时间

![轴与视频对齐](figures/reference_03_axis_and_video_alignment.png)

### L=3 与 L=5 敏感性

![tail敏感性](figures/reference_04_tail_length_sensitivity.png)

### slot239 / slot319 的 S/I 叠加与抵消

![S/I抵消](figures/reference_05_move_slot239_319_S_I_cancellation.png)

### 位置与阶段混叠

![位置阶段混叠](figures/reference_06_position_phase_aliasing.png)

## 二、固定位置参照（5 张）

### π0 adjust

![π0位置参照](figures/position_reference_pi0_adjust_bottle.png)

### Fast-WAM adjust

![FW adjust位置参照](figures/position_reference_fastwam_adjust_bottle.png)

### Fast-WAM move

![FW move位置参照](figures/position_reference_fastwam_move_stapler_pad.png)

### Fast-WAM turn

![FW turn位置参照](figures/position_reference_fastwam_turn_switch.png)

### Fast-WAM pick

![FW pick位置参照](figures/position_reference_fastwam_pick_diverse_bottles.png)

## 三、π0 adjust 成功代表（5 张）

<a id="pi0-adjust-success"></a>

### 裸 DVAC / y 与 b

![π0成功裸DVAC](figures/case_pi0_adjust_success__raw_dvac.png)

### raw residual r

![π0成功r](figures/case_pi0_adjust_success__raw_residual.png)

### standardized residual R

![π0成功R](figures/case_pi0_adjust_success__standardized_residual.png)

### query-wide S

![π0成功S](figures/case_pi0_adjust_success__query_wide_S.png)

### local I

![π0成功I](figures/case_pi0_adjust_success__local_I.png)

## 四、π0 adjust 失败代表（5 张）

<a id="pi0-adjust-failure"></a>

### 裸 DVAC / y 与 b

![π0失败裸DVAC](figures/case_pi0_adjust_failure__raw_dvac.png)

### raw residual r

![π0失败r](figures/case_pi0_adjust_failure__raw_residual.png)

### standardized residual R

![π0失败R](figures/case_pi0_adjust_failure__standardized_residual.png)

### query-wide S

![π0失败S](figures/case_pi0_adjust_failure__query_wide_S.png)

### local I

![π0失败I](figures/case_pi0_adjust_failure__local_I.png)

## 五、Fast-WAM adjust 成功代表（5 张）

<a id="fastwam-adjust-success"></a>

视频：[`fastwam_adjust_bottle_success_episode0002_reset1.mp4`](videos/fastwam_adjust_bottle_success_episode0002_reset1.mp4)

### 裸 DVAC / y 与 b

![FW adjust裸DVAC](figures/case_fastwam_adjust_success__raw_dvac.png)

### raw residual r

![FW adjustr](figures/case_fastwam_adjust_success__raw_residual.png)

### standardized residual R

![FW adjustR](figures/case_fastwam_adjust_success__standardized_residual.png)

### query-wide S

![FW adjustS](figures/case_fastwam_adjust_success__query_wide_S.png)

### local I

![FW adjustI](figures/case_fastwam_adjust_success__local_I.png)

## 六、Fast-WAM move 成功代表（5 张）

<a id="fastwam-move-success"></a>

视频：[`fastwam_move_stapler_success_episode0004_reset3.mp4`](videos/fastwam_move_stapler_success_episode0004_reset3.mp4)

### 裸 DVAC / y 与 b

![FW move成功裸DVAC](figures/case_fastwam_move_success__raw_dvac.png)

### raw residual r

![FW move成功r](figures/case_fastwam_move_success__raw_residual.png)

### standardized residual R

![FW move成功R](figures/case_fastwam_move_success__standardized_residual.png)

### query-wide S

![FW move成功S](figures/case_fastwam_move_success__query_wide_S.png)

### local I

![FW move成功I](figures/case_fastwam_move_success__local_I.png)

## 七、Fast-WAM move 失败代表（5 张）

<a id="fastwam-move-failure"></a>

视频：[`fastwam_move_stapler_failure_episode0006_reset5.mp4`](videos/fastwam_move_stapler_failure_episode0006_reset5.mp4)

### 裸 DVAC / y 与 b

![FW move失败裸DVAC](figures/case_fastwam_move_failure__raw_dvac.png)

### raw residual r

![FW move失败r](figures/case_fastwam_move_failure__raw_residual.png)

### standardized residual R

![FW move失败R](figures/case_fastwam_move_failure__standardized_residual.png)

### query-wide S

![FW move失败S](figures/case_fastwam_move_failure__query_wide_S.png)

### local I

![FW move失败I](figures/case_fastwam_move_failure__local_I.png)

## 八、Fast-WAM turn 成功代表（5 张）

<a id="fastwam-turn-success"></a>

视频：[`fastwam_turn_switch_success_episode0009_reset8.mp4`](videos/fastwam_turn_switch_success_episode0009_reset8.mp4)

### 裸 DVAC / y 与 b

![FW turn成功裸DVAC](figures/case_fastwam_turn_success__raw_dvac.png)

### raw residual r

![FW turn成功r](figures/case_fastwam_turn_success__raw_residual.png)

### standardized residual R

![FW turn成功R](figures/case_fastwam_turn_success__standardized_residual.png)

### query-wide S

![FW turn成功S](figures/case_fastwam_turn_success__query_wide_S.png)

### local I

![FW turn成功I](figures/case_fastwam_turn_success__local_I.png)

## 九、Fast-WAM turn 失败代表（5 张）

<a id="fastwam-turn-failure"></a>

视频：[`fastwam_turn_switch_failure_episode0001_reset0.mp4`](videos/fastwam_turn_switch_failure_episode0001_reset0.mp4)

### 裸 DVAC / y 与 b

![FW turn失败裸DVAC](figures/case_fastwam_turn_failure__raw_dvac.png)

### raw residual r

![FW turn失败r](figures/case_fastwam_turn_failure__raw_residual.png)

### standardized residual R

![FW turn失败R](figures/case_fastwam_turn_failure__standardized_residual.png)

### query-wide S

![FW turn失败S](figures/case_fastwam_turn_failure__query_wide_S.png)

### local I

![FW turn失败I](figures/case_fastwam_turn_failure__local_I.png)

## 十、Fast-WAM pick 成功代表（5 张）

<a id="fastwam-pick-success"></a>

视频：[`fastwam_pick_bottles_success_episode0004_reset3.mp4`](videos/fastwam_pick_bottles_success_episode0004_reset3.mp4)

### 裸 DVAC / y 与 b

![FW pick成功裸DVAC](figures/case_fastwam_pick_success__raw_dvac.png)

### raw residual r

![FW pick成功r](figures/case_fastwam_pick_success__raw_residual.png)

### standardized residual R

![FW pick成功R](figures/case_fastwam_pick_success__standardized_residual.png)

### query-wide S

![FW pick成功S](figures/case_fastwam_pick_success__query_wide_S.png)

### local I

![FW pick成功I](figures/case_fastwam_pick_success__local_I.png)

## 十一、Fast-WAM pick 失败代表（5 张）

<a id="fastwam-pick-failure"></a>

视频：[`fastwam_pick_bottles_failure_episode0005_reset4.mp4`](videos/fastwam_pick_bottles_failure_episode0005_reset4.mp4)

### 裸 DVAC / y 与 b

![FW pick失败裸DVAC](figures/case_fastwam_pick_failure__raw_dvac.png)

### raw residual r

![FW pick失败r](figures/case_fastwam_pick_failure__raw_residual.png)

### standardized residual R

![FW pick失败R](figures/case_fastwam_pick_failure__standardized_residual.png)

### query-wide S

![FW pick失败S](figures/case_fastwam_pick_failure__query_wide_S.png)

### local I

![FW pick失败I](figures/case_fastwam_pick_failure__local_I.png)

## 十二、总体时间曲线（20 张）

每张图只含一个信号；上为 episode 自身归一化进度，下为绝对 action slot，均先在 episode 内聚合再跨 episode 画 mean ± 1 SD，并显示 n-at-risk。

### π0 adjust

![π0 aggregate raw y](figures/aggregate_pi0_adjust_bottle__raw_y.png)

![π0 aggregate absR](figures/aggregate_pi0_adjust_bottle__abs_R.png)

![π0 aggregate S](figures/aggregate_pi0_adjust_bottle__S.png)

![π0 aggregate absI](figures/aggregate_pi0_adjust_bottle__abs_I.png)

### Fast-WAM adjust

![FW adjust aggregate raw y](figures/aggregate_fastwam_adjust_bottle__raw_y.png)

![FW adjust aggregate absR](figures/aggregate_fastwam_adjust_bottle__abs_R.png)

![FW adjust aggregate S](figures/aggregate_fastwam_adjust_bottle__S.png)

![FW adjust aggregate absI](figures/aggregate_fastwam_adjust_bottle__abs_I.png)

### Fast-WAM move

![FW move aggregate raw y](figures/aggregate_fastwam_move_stapler_pad__raw_y.png)

![FW move aggregate absR](figures/aggregate_fastwam_move_stapler_pad__abs_R.png)

![FW move aggregate S](figures/aggregate_fastwam_move_stapler_pad__S.png)

![FW move aggregate absI](figures/aggregate_fastwam_move_stapler_pad__abs_I.png)

### Fast-WAM turn

![FW turn aggregate raw y](figures/aggregate_fastwam_turn_switch__raw_y.png)

![FW turn aggregate absR](figures/aggregate_fastwam_turn_switch__abs_R.png)

![FW turn aggregate S](figures/aggregate_fastwam_turn_switch__S.png)

![FW turn aggregate absI](figures/aggregate_fastwam_turn_switch__abs_I.png)

### Fast-WAM pick

![FW pick aggregate raw y](figures/aggregate_fastwam_pick_diverse_bottles__raw_y.png)

![FW pick aggregate absR](figures/aggregate_fastwam_pick_diverse_bottles__abs_R.png)

![FW pick aggregate S](figures/aggregate_fastwam_pick_diverse_bottles__S.png)

![FW pick aggregate absI](figures/aggregate_fastwam_pick_diverse_bottles__abs_I.png)
