# 执行状态

2026-10-21：离线验收完成，当前TOPReward配置拒绝放行训练。已通过固定host-key和普通账户身份核实深圳2/3；12:12完成深圳2物理4/5的C20停止，并确认计算/图形上下文均已退出，[结论与证据](05_C20_CLOSEOUT.md)已登记。6/7既有GRPO保持运行。准备期间旧RLT候补保持WAIT_EXTERNAL；验收结束后通过新的真实释放回执交还原候补，配置不变。

深圳2目标目录：`/data/chenyiteng/deployment-20261021/wm-topreward-sz2-g45-v1`。

本地源目录：`worktrees/wm-topreward-20261021`，分支`codex/wm-topreward-sz2-20261021`。测试、GPU推理、训练均在服务器执行；Windows只维护源码与文档。

已完成收束、环境和数据准备、代码接线、CPU/GPU测试与完整离线标定；标定失败后中止后续WM联测、短对照与正式训练。[量化结果](06_RESULTS.md)。

已完成：128条轨迹一次性打包并迁至2机，共674,343,241字节；64/64校准/留出，每边32成32败，排除固定策略评估seed。该人为平衡样本不能估计自然部署中的阳性比例。旧RM独立test子集另算，避免把训练集诊断当泛化。

服务器CPU检查：8项unittest通过，涵盖校准/留出隔离、提前成功误报、float32阈值、无可行阈值、窗口边界、真实环境reset/恢复/活跃槽缩减、服务B16+尾批次历史对齐；新脚本Ruff通过，改动的旧环境/服务通过语法与未定义变量检查。真实Qwen处理器验证token3007及8/16/11/1帧行为。

离线运行配置：模型固定公开revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，bf16/eager，单视频≤16帧；5卡先1条smoke，再完整128条与旧RM同窗评分；4卡独立测16路管道请求和onload/offload。输出位于目标目录`native-smoke-v1`、`native128-scores-v1`、`b16-v1`。启动前核验两卡C/G空、CUDA UUID；smoke失败不启动后两项，评分结束自然退出，不在此阶段启动训练。

实际入口：`qwen-venv/bin/python source/local_scripts/topreward_20261021/evaluate_native.py --dataset native128 --model /data/chenyiteng/models/Qwen3-VL-8B-Instruct-0c351dd0 --legacy-code legacy --rm-checkpoint legacy/best.pt --rm-threshold 0.8137992024421692 --physical-gpu 5 --output native128-scores-v1`，设置`CUDA_VISIBLE_DEVICES=5`；其他完整argv登记于目标目录`offline-plan.json`。

12:49:41完成资源释放：离线owner及所有GPU子进程已退出，旧C20任务标签进程也已清空，4/5无C/G上下文；6/7原owner仍存活。只将原调度器4/5的external identity/release改到本次真实结束回执，RLT fallback请求和6/7配置不变。未伪造旧C20正常完成。[释放回执](evidence/released.json)、[调度交接](evidence/queue-handoff.json)。
