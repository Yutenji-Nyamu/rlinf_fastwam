# 已存RLinf单任务π0/π0.5：权重与成绩

核验时间：2026-10-27 14:40（北京时间）。使用既有固定host-key SSH只读查看1机我方模型目录；未加载GPU、未修改队列。本次只验证文件存在、索引分片齐全和归一化文件存在，未做权重数值重验。

## 实际仍在的官方SFT权重

|模型|任务|本机权重|固定来源|
|---|---|---|---|
|RLinf π0 SFT|adjust_bottle（调瓶子）|2个分片，7.514 GiB|`RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50dca1a5f75adc8d332046c4cf4fa7a3d0`|
|RLinf π0.5 SFT|adjust_bottle（调瓶子）|3个分片，7.943 GiB|`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98`|

本地路径分别为：

```text
/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
```

两个目录均有`physical-intelligence/robotwin/norm_stats.json`。权重索引引用的2/3个分片全部存在。官方还有对应PPO权重，但本次检查的`models/rlinf`目录未见其本地副本；不将“未见”扩大为所有服务器绝对不存在。

## 官方成绩与我们历史记录分开

RLinf当前文档公布的RoboTwin `adjust_bottle/demo_clean`结果：

|模型|专家示范微调SFT|进一步PPO训练|
|---|---:|---:|
|π0|76.56%|98.44%|
|π0.5|85.94%|96.09%|

这些是[官方公布值](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)，不是本轮在我们当前评分器中重新测得的成功率。它们也不是当前Sidney 50任务基座的成绩。

旧本地记录证明这些权重和任务曾跑通：π0.5首轮PPO采样83.20%、clean GRPO采样80.08%。但这属于训练rollout，不能当作未更新原始模型的独立N1/N10基线。见`docs/rlinf-shenzhen-pi05-robotwin/evidence/IMPLEMENTATION_LEDGER_20260831.md`。

[官方评估说明](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html)规定OpenPI clean协议、ALOHA-AgileX、三相机、不做中心裁剪，示例H/C50、调瓶子上限200动作。其种子来自环境预检可执行种子；这与我们后来的RoboTwin原生400动作协议有区别。

**去噪步数也不能混称N10**：工作区锁定的旧参考配置为π0默认N4、π0.5任务覆盖N5；旧π0.5交接明确将M5定义为内部5次生成更新。不能把上表称为“原始N10成功率”。若研究统一10→1，需要在目标协议里重测。

## 不要混淆的其他模型

- **Sidney π0.5多任务**：1机仍有约8.712 GiB模型文件；50任务，绝对动作、mean/std，H50/N10。这是当前多任务蒸馏基座，和上面的官方单任务SFT不是同一个模型。
- **我们自己的pillbottle RL候选**：保护目录中Clean CP140与Uτ3 CP70的`full_weights.pt`仍各约7.941 GiB。它们属于另一个任务/实验族，不能写成RLinf官方adjust_bottle单任务权重。保护规则继续生效。
- 本次还看到了其他保护目录名，但没有相应大权重，故没有把空目录计为可用模型。
- RLinf公开合集还列有`Pi0-NEW...adjust_bottle`等新版本。本次没有替换已有固定权重。原单任务模型可作为近期试验桥梁，但不会自动提供另外四个任务的专家老师。

## 适配时必须保留的模型差异

|项目|现有Sidney π0.5|RLinf π0.5单任务|RLinf π0单任务|
|---|---|---|---|
|config_name|pi05_sidney_robotwin|pi05_aloha_robotwin|pi0_aloha_robotwin|
|动作|绝对关节值|关节相对当前状态的delta，夹爪仍绝对|同类delta处理|
|归一化|显式mean/std|随官方数据配置和该权重norm文件|随官方数据配置和该权重norm文件|
|坐标适配|adapt_to_pi=False|原数据配置继承adapt_to_pi=True，不能漏掉|原数据配置显式False|
|状态输入|离散化进前缀|离散化进前缀|连续状态进入动作侧|
|当前蒸馏动作模块选择|已支持|结构接近，可复用|需加入state_proj/action_time_mlp等|
|当前CP target-time投影|已支持|可沿用π0.5分支|现实现明确只支持π0.5，需要新接入|

`prepare()`已经调用模型的完整input_transform，故可以保留统一供数，但必须选对config_name/normalizer；不能手工先差分一次再让transform差分第二次。output_transform负责恢复绝对动作后交给机器人。

另一处容易误读：已下载π0.5目录的`config.json`写`action_horizon=10`，而RLinf实际加载由`get_openpi_config`构造模型配置，官方任务执行设置50。以后必须记录解析后的模型H、执行C及实际输出shape，不能凭文件夹中一个JSON字段推断真实协议。

本次只读现场记录保存在本线程E盘`fm_followup1027/evidence/inventory.out`；网络/身份信息不进入公开文档。公开来源：[RLinf RoboTwin合集](https://huggingface.co/collections/RLinf/robotwin)、[π0 SFT](https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle)、[π0.5 SFT](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle)。
