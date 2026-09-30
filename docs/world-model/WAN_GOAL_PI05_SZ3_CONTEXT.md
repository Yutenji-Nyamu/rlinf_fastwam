# SZ3 Wan Goal → π0.5 GRPO 实施上下文

更新时间：2026-09-30。当前执行目标由用户明确选定为 RLinf 官方 LIBERO Goal + Wan：先 OpenVLA-OFT GRPO smoke，再完成 π0.5 接口适配并启动正式训练。此前 RoboTwin/OpenDW/WorldArena 的讨论留存，不作为当前启动配方。

## 授权与资源

- 仅深圳3，物理 GPU 4–7。环境、源码、模型、缓存与日志放 `/data/chenyiteng/projects/wan-goal-sz3`。
- 20:30实查 GPU 4–7 为 Dojo π0.5 全量评测，2072/6300 回合、166次成功，RLT已暂停；此数字只是切换前快照。
- 用户随后明确：**准备好后暂停 Dojo，WM结束再续 Dojo**。保留原结果和 resume manifests；Dojo最终结束仍交还原RLT。准备期间保持Dojo运行。
- 当前旧 outer 的 finally 会自动恢复 RLT；不能直接杀它后抢卡。实施采用旧链完整归还，再由新链精确借卡，WM阶段结束清理后续原Dojo。细节见 `WAN_GOAL_RESOURCE_SWITCH.md`。
- 其他用户、共享 Ray 和 GPU 0–3保持原状。每次信号操作重新核验 UID/PID/starttime/boot/命令摘要，借卡后核查真实GPU占用。

## 固定来源

|组件|官方来源|固定版本|
|---|---|---|
|RLinf|https://github.com/RLinf/RLinf|d34d4c320d08cb982de034aa9a011f08dc0fa217|
|Wan Goal|https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal|bd395971c3467de3dd19e7e6c7562af48a2894a6|
|OFT Goal SFT|https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1|d20e1d447dfd87c0daa121b0739e2a379f7fe334|
|π0.5 LIBERO SFT|https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT|45ccfcc4e28634f1576ebf78cab0fbe2fd82432d|

官方入口：https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html 。安装与文件清单见 `WAN_GOAL_OFFICIAL_RUNBOOK.md`。

补充用户追问：本次OFT由官方配置设为 `num_images_in_input=1/use_proprio=false`，只用外部主相机和语言。OpenSora的RLinf公开路径也无腕图、无proprio；发布Spatial/Object资产，可直接启动的固定版YAML为Spatial＋OFT，没有现成Goal组合。OpenSora因此不能补齐π0.5腕图。依据：[OpenSora官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opensora.html#run-it)。

## 实施顺序和验收

1. 在独立目录获取固定源码、环境和模型；CPU侧检查依赖、配置解析及模型文件。准备期间不借GPU。
2. 从官方 `wan_libero_goal_grpo_openvlaoft` 缩短预算做smoke，保留Goal模型/动作/奖励语义；记录明确预算与resolved配置后切卡。验收真实WM生成、策略轨迹、有限loss和参数更新。
3. π0.5接口审计：Wan官方只输出外部主相机RGB，没有腕图和proprio；选定的 `pi05_libero` 虽保留8D状态字段，`discrete_state_input=False`且PI05无state projection，策略实际不以state为条件。硬缺口是腕图。动作是7D末端增量，不能把最后一条动作当成下一8D实测状态。用户已接受显式image mask屏蔽腕图，训练/真实LIBERO评测保持同一单视角口径。
4. π0.5以匹配输入验证一次真实更新，再按记录的正式预算启动；真实LIBERO评测与WM内部reward分开记录。尚未通过接口审计时不宣称正式组合可运行。
5. WM正常结束或失败：只清理新WM owned进程，核4–7释放，续原Dojo。Dojo结束后由唯一outer恢复原RLT。

## 当前状态

- 官方模型清单已核，三套合计36.78GB（未含Python环境与LIBERO资产）。
- SSH主机指纹/账户身份已验；现场只读快照已保存。
- 独立目录和固定源码已建立，20:38起安装OFT环境并下载固定模型；尚未停止Dojo、尚未启动WM训练。
- 用户已明确接受π0.5单视角适配，要求真实更新通过后启动正式训练。两文件补丁已应用，三套独立配置已于21:23通过服务器Hydra解析；目标环境仍在安装。原生预测H10、执行C8、采样M5；不把网络H改成8。
- 22:02网络进展：读取服务器网络README后，模型与本任务安装切至实测较快的现有7897代理；模型持续约6–8MiB/s，保留TLS断连续传。PyTorch同版本官方wheel已校验并复用装入两venv，官方安装已恢复。具体结果见 `WAN_GOAL_NETWORK_20260930.md`；源码/日志的公开发布仍待本轮准备验证，不能称已推Git。
- 23:33：两套环境安装完成；765运行文件/7大文件SHA/742初态及tokenizer通过，OFT/π05实际模块导入与π05单视角CPU接口通过。π05的dm-control/MuJoCo已对齐LIBERO约束；剩余官方覆盖和非当前入口的依赖警告逐项保留，详见`WAN_GOAL_ENVIRONMENT_NOTES.md`。准备发布后切入GPU smoke；仍未切卡、未发生GRPO更新。
- 23:53：首个发布提交已生成但推送失败，原因是初始精简克隆缺旧版本Git对象；完整固定上游历史已下载，正补齐新的发布副本。切换预检另发现旧watchdog需要真实路径，仅在此接口将`/data`别名解析；实际只读绑定复验通过，无信号。启动配置/预算/停止条件见`WAN_GOAL_LAUNCH_20260930.md`。

粗日志见 `WAN_GOAL_RUNLOG.md`；每条真实远端命令、stdout/stderr、退出码、时间、SHA256保存在本地E盘证据目录，文档索引不包含任何密码或令牌。
