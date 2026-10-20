# Signed-U GRPO τ1/2/3：2026-10-20

用户授权用三组EXPO释放的六张卡，运行“正优势偏高U、负优势偏低U”，τ1/2/3，无dropout/退火。复用10月19日两版真实完整一轮smoke验收中的signed实现；本轮不修改损失公式、不重跑已通过的232项CPU测试。

## 配置与资源

| 主机/物理卡 | 本轮方法 | 结束后 |
|---|---|---|
| 深圳1·6/7 | signed-U τ1 | 原逐卡RLT |
| 深圳2·6/7 | signed-U τ2 | 原逐卡RLT |
| 深圳3·4/5 | signed-U τ3 | 原逐卡RLT |

三组均从原始Sidney π0.5与新优化器开始，任务`move_pillbottle_pad`。200训练轮；每轮64环境×4次采样=256轨迹，group8，global batch512/micro32，更新2遍；N10、H50、D14、每回合上限200动作、Flow-SDE噪声0.5、学习率5e-6。每2轮固定32回合评估、每10轮保存checkpoint。训练/评估种子继承最近同族正式GRPO。

唯一方法变化是已验收的`direction=advantage_sign, scope=both`；块内与块间τ同为1/2/3。正负按GRPO优势符号决定，并非只按原始成功标签；零优势不产生有效策略梯度。固定alpha1，chunk_dropout与alpha_schedule均关闭。

## 启动与停止

每机独立事务`/data/chenyiteng/deployment-20261020/grpo-u-signed-tau123-v1`，源码`source/`；输出`results/rlinf-shenzhen/pi05-sidney/runs/grpo-u-signed-tau{1,2,3}-notrick-pillbottle256-{host}-g{pair}-200-1020/`。

准备入口`tools/grpo_un/prepare_signed_formal.py`，继承该主机既有Ray端点；启动`tools/grpo_un/launch.py --plan <事务>/plan.json --lane signed`。GPU UUID及C/G绑定沿用验收机制。新EXPO收束回执、无本任务残留上下文确认后才启动；已有started.json不得重放。

停止条件为完成200轮、训练失败、非有限状态或GPU越界；不因早期成功率低自动中止。沿用原driver清理自己namespace，RLT owner等待本轮released.json，保留其他卡与共享Ray。SZ1部署时修正了继承的Ray地址并为重试日志使用独立文件，失败发生于采样启动前，没有重复训练。

## 记录

[三组EXPO收束与13条同族曲线](EXPO_CLOSEOUT.md)。当前首轮运行证据另存`startup-evidence.json`；运行中结果不是完成后的效果结论。截至11:51，τ1/τ2已完成首轮更新，τ3在第4次采样；权重、梯度及损失记录正常。正式方法源码、配置准备和轻量证据随本分支发布，不包含模型权重。
