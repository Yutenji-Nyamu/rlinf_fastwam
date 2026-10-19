# 2026-10-19执行

## 启动前配置

原始Sidney π0.5，`move_pillbottle_pad`，64环境×4轮采样=256轨迹，group8，globalB512/micro32，update_epoch2，H50/D14，200动作上限，N10，Flow-SDE0.5。继承最近Uτ2 pillbottle的训练/评估种子文件、lr5e-6、优化器、奖励过滤。每版从原模型和新优化器开始。

两版local/chunk τ2、固定alpha1；与最近正式配置一致，dropout和退火默认关闭，功能和CPU覆盖保留。smoke仅一轮，无评估、无模型checkpoint。停止条件为完成一轮、进程失败、非有限权重或GPU越界；不设成功率门槛。

物理资源：深圳3 GPU2/3，为本轮核验到最高机器/卡号的空闲两卡。计算与图形均使用原UUID绑定机制；只扩展显式允许的2/3配对。两个版本顺序执行，不修改其他任务或RLT排队。

事务 `/data/chenyiteng/deployment-20261019/grpo-u-sign-smoke-v1`，源码在其`source/`；输出 `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/grpo-u-{signed,positive}-pillbottle256-t2-smoke1-sz3-g23-20261019-v1`。

准备命令：`/usr/bin/python3 -B source/tools/grpo_un/prepare_sign_smoke.py`。
启动命令：`/usr/bin/python3 -B source/tools/grpo_un/launch.py --plan plan.json --lane signed`；确认前一版driver、独立Ray命名空间及2/3计算/图形上下文释放后，再以`--lane positive`启动第二版。

启动器记录唯一PID/UID/start、命令、配置差异和GPU UUID；沿用原driver只清理本次Ray job/namespace。无共享Ray重启或GPU reset。

## 验收

服务器CPU检查：232项通过，38.75秒；包含现有U采样/RNG、两层权重、actor/损失/恢复、τ及trick兼容性和2/3图形选择。CUDA禁用。真实GPU验收进行中，尚无学习效果结论。
