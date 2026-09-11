# Stage1来源与当前阶段

2026-09-12只读核验。深圳已有完整Stage1，两条正式实验均从同一个产物启动新的Stage2；没有重新训练Stage1，也没有跳过必要初始化。

- 任务：RoboTwin `adjust_bottle`，ALOHA双臂、14维关节动作；与本项目近期π0.5 BC实验分开看。
- 数据：该任务50条示范、7188帧；目录名`pi0-aloha-clean50-v1`。
- Stage1：冻结原π0，训练RLT的潜在表征/重建模块（current causal-AR），不是重新训练整个π0；2000/2000步，2026-08-24 00:44:51 CST结束，exit0。
- 完整权重：`global_step_2000/actor/model_state_dict/full_weights.pt`，9,556,454,857字节；manifest及实际文件大小已核，正式加载后两条都完成第一轮真实采集。
- 这是任务内数据训练出来的RLT表征产物。新任务是否可直接迁移需另验，不能把它当所有任务通用的完成凭证。
- 当前流程：载入Stage1及冻结π0 → 用π0采集20k条chunk转移 → 原协议30k次初始化更新 → 后续在线actor/critic训练。600指Stage2外层cycles，不是600次Adam。
- AutoDL协议一致；深圳Stage1由深圳自己训练，数据/源码身份由manifest记录，因此不是两服务器同一份权重逐位复现。
- 如果这份Stage1不存在或不兼容，原实现应先完成匹配的Stage1（或导入已核验兼容的产物），不能省略后仍称同一RLT协议。本轮无需补训。

[正式配置](../../FORMAL600_20260912.md) · [实测manifest](stage1-manifest.json) · [首轮健康回执](STARTUP_VERIFIED.json)
