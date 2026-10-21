# 实现与验证

对每条完整轨迹，只在最后一个 C32 块的最后动作发放一次奖励：

$$
R_i=p_i(\mathrm{True}\mid\text{末尾至多16帧},\text{任务描述}).
$$

中间块奖励为 0，`terminated=False`；达到 384 动作时 `truncated=True`。Qwen 分数和历史仍逐块记录。`success_once` 在 WM 环境中不再代表任务成功、预期为 0；真实任务成功只看原生 RoboTwin `eval/success_once`。

新增 `reward_mode=terminal_score` 到 `opendw_adapter.map_frame_scores` 和 `OpenDWRobotwinEnv`，保留原有二值/ResNet 模式。要求 `use_rel_reward=false`、`reward_coef=1`。不用 .46 阈值、不裁剪为二值、不对完整分数求和、不取历史最大值。

现有 RLinf `calculate_scores` 会沿轨迹汇总奖励；GRPO 按同起点 G8 做 `(R-mean)/(std+1e-6)`。关闭旧 `filter_rewards` 的组均值 0.1–0.9 筛选，防止将连续低分但有差异的组全部丢掉；不新增排序算法或动态阈值。等分组没有有意义的梯度，实际有效信号需看组内分布和 actor 梯度。

验证顺序：

1. 服务器 CPU 回归：原有历史隔离/恢复、视频帧和服务排序测试；新增完整 12 块通过真实 GRPO 聚合的排序/末段奖励测试。
2. 服务器 GPU：真实 OpenDW B16 与 Qwen 同驻 5 卡，检查 finite、顺序、显存和 offload；再通过真实 RLinf rollout/advantage/actor 更新链路。
3. 正式配置复核与推送；正式训练沿用原生 32 seed 评估。奖励升高本身不能判定方法成功。

任何 GPU 实验前先保留占卡 RLT 的完整恢复点，按精确 driver/namespace 停止，只在 4/5 卡计算与图形绑定确认后继续。
