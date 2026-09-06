# π0.5 GRPO 主动收尾

- 结束：2026-09-06T14:51:44.515267+08:00，用户要求提前停止，不是训练故障。
- 完整训练：168/200轮；每轮256条、两卡各32环境、串行4次。
- 最新固定评估：Step165，24/32；历史最高Step70，26/32。
- 最近10轮训练采集均值：67.66%；最近10次固定评估均值：63.44%。
- 最后完整可恢复checkpoint：Step160。之后记录的训练轮次不等于有对应模型快照。
- 已有提升，后段平台与波动并存；不宣称严格证明收敛。没有训练前Step0固定评估。
- 主代码分支：codex/sz-sidney-pi05-current-rlinf，封存前HEAD 81be3193d91fe9950a3fc1bdedd14063a85e72d8。

## 文件

- 01_success.png、02_timing.png：成功率和耗时图；index.html离线浏览。
- rounds.csv：按轮次整理；metrics.csv/scalars.json：全部原始标量。
- raw/runtime、raw/runtime-resume100-to200：两阶段完整driver日志、命令、配置和资源采样。
- raw/tensorboard：原始事件与配置；raw/metrics.log：原指标日志。
- checkpoint_manifest.json：保留权重清单；stop_receipt.json：精确停止和其他任务保护回执。
- FILE_MANIFEST.json：本包逐文件SHA256与体积。

不含大模型、checkpoint张量、视频、回放数据或凭据。大权重仍在服务器，没有删除。新的adv-DVAC运行从相同原始SFT开始，不承接已训练Control权重。
