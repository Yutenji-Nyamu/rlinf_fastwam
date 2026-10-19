# 2026-10-19执行

两版均通过真实完整一轮验收，18:36已确认深圳3 GPU2/3释放。方法定义见[README](README.md)。本轮仅smoke，没有启动长期训练。

## 启动前配置

原始Sidney π0.5，`move_pillbottle_pad`，64环境×4轮采样=256轨迹，group8，globalB512/micro32，update_epoch2，H50/D14，200动作上限，N10，Flow-SDE0.5。继承最近Uτ2 pillbottle的训练/评估种子文件、lr5e-6、优化器、奖励过滤。每版从原模型和新优化器开始。

两版local/chunk τ2、固定alpha1；与最近正式配置一致，dropout和退火默认关闭，功能和CPU覆盖保留。smoke仅一轮，无评估、无模型checkpoint。停止条件为完成一轮、进程失败、非有限权重或GPU越界；不设成功率门槛。

物理资源：深圳3 GPU2/3，为本轮核验到最高机器/卡号的空闲两卡。计算与图形均使用原UUID绑定机制；只扩展显式允许的2/3配对。两个版本顺序执行，不修改其他任务或RLT排队。

事务 `/data/chenyiteng/deployment-20261019/grpo-u-sign-smoke-v1`，源码在其`source/`；输出 `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/grpo-u-{signed,positive}-pillbottle256-t2-smoke1-sz3-g23-20261019-v1`。

准备命令：`/usr/bin/python3 -B source/tools/grpo_un/prepare_sign_smoke.py`。
启动命令：`/usr/bin/python3 -B source/tools/grpo_un/launch.py --plan plan.json --lane signed`；确认前一版driver、独立Ray命名空间及2/3计算/图形上下文释放后，再以`--lane positive`启动第二版。

启动器记录唯一PID/UID/start、命令、配置差异和GPU UUID；沿用原driver只清理本次Ray job/namespace。无共享Ray重启或GPU reset。

## 验收

服务器CPU检查：232项通过，38.75秒；包含现有U采样/RNG、两层权重、actor/损失/恢复、τ及trick兼容性和2/3图形选择。CUDA禁用。Ruff0.14.3检查/格式通过；格式化前后实际训练源码AST一致。

| 版本 | 完成时间 | 轨迹/更新遍数 | 权重范围 | 梯度范数 | 更新耗时 | 总耗时 |
|---|---|---|---|---|---|---|
| 成功高U、失败低U（signed） | 17:56 | 256/2 | 0.515231–1.850489 | 25.458437 | 30.52秒 | 36.76分钟 |
| 仅成功高U、失败原权重（positive） | 18:35 | 256/2 | 0.546352–1.850489 | 21.164389 | 29.59秒 | 38.27分钟 |

signed的262个正优势块、520个负优势块的块内/块间方向均通过；positive的成功块方向通过，失败块权重严格为1。各符号组平均权重为1，零优势/无效块保持1；总loss分别0.00309059/0.00242396，梯度和损失均有限。两版均退出0、cleanup_error=null，精确namespace和GPU2/3 C/G清空。其他任务及原RLT队列未修改。

18:36最终快照：GPU2/3各5MiB驱动底层读数，无C/G进程；GPU0/1各1MiB、无C/G进程。未执行GPU reset或共享Ray重启。

两版原模型采样均78/256（30.46875%），这是参数更新前的采样，不能当作训练后提升。本轮没有正式学习效果结论。

## 已保留的轻量产物

- [CPU检查](evidence/cpu-tests.txt)、[signed验收](evidence/signed.json)、[positive验收](evidence/positive.json)。
- [signed实配](evidence/signed-config.json)、[positive实配](evidence/positive-config.json)。
- [完整轻量归档](evidence/smoke-artifacts.tar.gz)：基线/实配、原种子、启动/完成回执、日志、TensorBoard事件及每卡逐动作U/加权系数/优势/mask诊断张量；[文件目录](evidence/contents.json)。这些张量不是模型参数。原始模型权重、回放和视频未打包。

后续使用两份overlay即可选择版本；本次原始控制输入保留在归档。部署脚本是单次事务，已有`started.json`的请求不得重放；正式实验另开唯一输出/namespace，复用已验证源码和配置。
