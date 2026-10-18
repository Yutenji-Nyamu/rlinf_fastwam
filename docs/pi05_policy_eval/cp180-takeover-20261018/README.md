# GRPO U τ1 CP180接管：深圳3物理6/7

CP180在配对128回合全程使用时102/128＝79.6875%，原始55/128，旧Clean CP140为72/128。本轮补CP180的high-U、low-U、random接管，各128回合，共384；之前7组仅包含近期RL全程评估和原始C20接管，未包含这三组。

复用已验收的v3源码、128种子、独立64回合校准、U计算、阈值、随机门控和完整视频实现，仅换strong_weights为保护库grpo_u_tau1_cp180。模型、N10/H50/C50、每卡32并行、200动作上限、原生成功判据不变。顺序high→low→random；每条件两卡各两批32。阈值low=0.011454011825020368，high=0.018057912783599753，随机概率1/3。记录实际接管决策/动作比例及配对成功差，不声称比例严格相等。

服务器源码为`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-cp180-takeover-20261018`，来源提交1dc941c42。仅修改lane允许显式指定SZ3 UID20001，以及权重等待覆盖接管策略；推理算法未改。完整录像仍使用300秒原生线程等待，批次3600秒/进度停滞600秒退出、不自动重试。

事务为`/data/chenyiteng/deployment-20261018/cp180-takeover-sz3-v1`；输出为`/data/chenyiteng/results/pi05-policy-eval/pillbottle-cp180-takeover-20261018-v1`。两路命令为原系统Python执行`tools/pi05_policy_eval/lane.py 6`或`7`，设置POLICY_EVAL_ROOT、POLICY_EVAL_TRANSACTION和POLICY_EVAL_UID=20001。GPU计算按单卡绑定，复用原sitecustomize和对应EGL进程名配置限定渲染到同卡。

服务器CPU协议4项测试和原生录像动作/物理/RNG一致性测试通过。原EXPO U-BC τ3已正常保存并退出，后代与C/G清场通过，其他卡对与所有RLT fallback保持。当前两路为WAIT_WEIGHTS；7.94 GiB从深圳1保护库复制到深圳3同路径，临时`.copying`完成后原子改名，再自动开始。未将等待状态声称为GPU推理已启动。

所有条件结束或错误退出，lane完成清场并释放，原owner逐卡接回RLT lift_pot CP225/225。不得重放cutover，不删除protected-eval保护目录。EXPO收束证据单独发布于codex/expo-ubc3-closeout-20261018。
