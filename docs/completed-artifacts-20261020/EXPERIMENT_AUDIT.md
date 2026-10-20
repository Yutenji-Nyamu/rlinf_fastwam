# 10月17–20日完成实验归档核对

范围为深圳1/2/3既有`/data/chenyiteng/results`、`/home/chenyiteng/results`及专题指向的已知项目，不包括AutoDL或任意临时目录。源码分支与运行结果分别核对。

| 实验 | 对应仓库 | 核对与补齐 |
|---|---|---|
| GRPO Uτ1/τ2收束；signed/positive smoke | rlinf_fastwam | 方法/收束分支已推；补近期运行日志与完成回执 |
| EXPO Clean、U-BCτ3、Editor-Uτ2及10/20收束三组 | rlinf_fastwam | 收束证据已推；补结果树中的更新日志/评估 |
| 原始、强模型、C20等22条件比较 | rlinf_fastwam | 10/18 Release已有；本次保留更新快照 |
| CP180 30阈值扫描 | rlinf_fastwam | 原分支只有标定/启动记录；本次补最终汇总与原始轻量记录 |
| RoboTwin U、四噪声采集 | rlinf_fastwam；四噪声图册另在tts_consistency_fig | 源码和精选图已推；补完整/失败批次索引及新增轻量日志 |
| VLABench官方评估 | vlabench_pi05 | 最终结果已推59a71a4，远端HEAD相同，不重复上传视频 |
| RLT近期暂停/恢复段 | rlinf_fastwam | 补TensorBoard、配置、停止回执和曲线；不把暂停当整个实验结束 |
| Dojo U采集 | rlinf_fastwam | 尚未整体结束；补945回合/104成功快照、已完成/跳过任务及13条精选素材 |

10条相关本地分支HEAD与远端完全相同；这不等于所有后续运行产物已上传，因此另补54,446份轻量文件快照，113个压缩分片、约1.41 GiB。配置/日志/JSON/CSV/事件和绘图材料保留；权重、优化器、回放、原始大数组、完整视频和环境/凭据文件不上传，超过8 MiB的文本逐路径列在排除表。

分支`codex/completed-artifacts-20261020`，Release `completed-artifacts-20261020`。Git保留索引、512份汇总/完成记录、最新78条曲线和Dojo精选图，原始轻量分片走Release。发布完成状态以本聊天E盘`completed-audit1020/publication.json`为准；只确认推送和上传回执，不重复逐文件SHA256。

[仓库归档入口](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/completed-artifacts-20261020/docs/completed-artifacts-20261020) · [Release](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261020) · [Dojo图册](http://127.0.0.1:8194/dojo_u_1020/)
