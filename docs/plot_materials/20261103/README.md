# 绘图材料总索引 · 2026-11-03

当前说明与小数据在本目录；新增完整绘图包见[本轮Release](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/plot-materials-20261103)。公开包不含模型大权重、训练回放和完整视频。旧已发布材料使用原入口，不重复上传。

| 材料 | 用途与内容 | 入口／状态 |
|---|---|---|
| 最新蒸馏与GRPO | GAD四方法、OT、FQL v2全已有评估、逐回合成绩、训练指标、配置和TensorBoard；GRPO到58轮 | 本轮plot-backup-20261103.tar.gz，5043个文件 |
| 完整同族曲线与图稿 | 可勾选HTML、全点JSON、PNG/SVG/PDF；保留0–400轮和0–100%纵轴 | 本轮plot-galleries-20261103.tar.gz |
| 药瓶师生双熵 | Clean140与Uτ3CP70，32局12成功，5725到达位置；十query教师K4、逐位置CSV、十组及Top20统计 | [已推e9622c461](../../experiments/entropy-replay-20261102/)；本轮补原学生K4、原噪声、教师原预测与帧映射，shadow-pair-raw-20261103.tar.gz |
| Slot25与九类统计图 | 12阶段帧、双曲线、可编辑Draw.io、九统计图、所有轨迹CSV/JSON | 本轮图册包；[既有slot25说明](../../experiments/shadow-slot25-20261029g/)；bkp历史be6fc45b5 |
| N1/N5/N10与DV | 41任务1312有效局；全体逐动作曲线、80条精选原始动作/噪声/观察/内部估计、帧映射、图稿 | [三个附件](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/nstep-signal-20261101)，本轮API确认均uploaded |
| G仅前3块接管 | 2560回合、23412决策、标定/阈值、逐动作U、CSV及图稿 | [bkp目录](https://github.com/Yutenji-Nyamu/uaca_cv27_draw_bkp/tree/main/materials/g_first3_takeover_20261023)，本轮确认存在 |
| Dojo精选4条 | 叠碗、叠方块各2条；对应曲线、选帧和交接 | [bkp目录](https://github.com/Yutenji-Nyamu/uaca_cv27_draw_bkp/tree/main/materials/dojo_selected4_20261020)，本轮确认存在 |
| 旧四噪声图册 | 46任务75条精选与图稿 | [tts_consistency_fig](https://github.com/Yutenji-Nyamu/tts_consistency_fig)，本轮确认仓库存在 |

## 当前尚未生成

新Clean同run六点的固定观察池与重放需等200轮完成，不能把旧Clean140/Uτ3CP70拼成训练曲线。CP40完整恢复点已生成，尚有80/120/160/200；六点受保护。训练后自动生成平均/分位熵随训练、Top20重合和固定十组熵变化。

## 统计口径

- 旧shadow1029f图册使用5820个提交位置；11月2日双熵使用5725个控制器实际到达位置。二者保留各自口径，不能不加说明拼接。
- 四噪声分散度与对角高斯熵是不同定义；师生距离不是动作真值误差。教师只有一份动作能算距离，不能算教师自身的四噪声熵。
- 原始相机观察与完整视频留在原服务器和本地素材目录；此次补的原动作、噪声及映射足以重新计算双熵、距离和定位图中动作位置。
- 旧公开附件本轮仅查存在、uploaded状态和条目，未逐文件重下载或核对哈希。

Server overview and process identities remain local and are excluded from this public backup.
