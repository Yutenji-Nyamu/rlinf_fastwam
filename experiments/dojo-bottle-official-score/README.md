# Dojo π0.5 GRPO · 丢瓶子 · 官方进度分

任务为 `put_bottles_into_dustbin`。按 RoboDojo 原生评估接口计分。成功局给 1，失败局由 `RewardManager.get_score` 返回阶段分，除以 100 后使用。投进一、二、三瓶分别累计 0.10、0.25、0.40。投完四瓶还需双臂归位才能得到 1。每局在结束的实际动作位置记一次总分；GRPO 按八局的轨迹总分计算相对优势。独立评估仍按官方成败计成功率。[官方任务](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/put_bottles_into_dustbin.py#L19-L50) · [原生评估](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/src/eval_client/eval_env.py)

与旧版逐瓶 0.25 对照，从原始 π0.5 开始。每轮 128 局，每组 8 局，两卡各 32 个仿真环境、采两批。每局 700 步，每 50 步看图，训练噪声 0.5，学习率 5e-6，每轮更新两遍，上限 300 轮。先做原模型 32 局独评，再在第 2 轮及每 10 轮评估同批布局。第 20、30 轮重点看完整成功率、平均瓶数、投完三瓶后最终完成的比例。

配置来自正式入口。[训练配置](formal.yaml) · [评估配置](final_eval.yaml) · [原始参考配置](../dojo-bottle-quarters/formal.yaml) · [训练实现](../../tools/dojo_grpo/native_entry.py) · [CPU 测试](../../tools/dojo_grpo/test_bottle_progress.py)。设置两个 native worker 的 `training_reward` 为 `bottle_official`。原始逐局记录存于各自 `native-g*/completed-waves.jsonl`，将部署目录传给 [summarize_progress.py](summarize_progress.py) 可复算三个指标。首轮回执以 [run.json](run.json) 为准。

