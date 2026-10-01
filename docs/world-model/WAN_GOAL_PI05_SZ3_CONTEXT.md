# SZ3 Wan Goal → π0.5 GRPO 实施上下文

更新时间：2026-10-01。当前执行目标由用户明确选定为 RLinf 官方 LIBERO Goal + Wan：先 OpenVLA-OFT GRPO smoke，再完成 π0.5 接口适配并启动正式训练。此前 RoboTwin/OpenDW/WorldArena 的讨论留存，不作为当前启动配方。

## 授权与资源

**2026-10-01 11:05：r5正式完成4轮后监控AssertionError中断；内外层释放已验，原v5 owner已唯一恢复原四RLT（CP125、11:03四driver活，首轮待验）。owner/guard现已正常结束，guard=RLT_RETURN_DISPATCHED。当前只维护原三机RLT，不自动新WM/Dojo；详见[中断与归还记录](WAN_GOAL_FORMAL_FAILURE_20261001.md)。**

**2026-10-01 10:04：r5正式第0轮已完成真实训练，grad norm0.79649、有效mask1.2598%、有限非零优势及loss；已进入第1轮采集。原1000轮/save40保持，当前归还仍WM→原RLT。另窗“0927 exp”已确认本轮EXPO-FT只研究、GPU需求0；本窗口维护[两窗资源表](../server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md)，后续另窗部署先刷新并登记具体卡/run/owner。**

**2026-10-01 09:51最新用户指令：WM结束后直接归还原RLT，取消Dojo续评测。w053已唯一部署当前v5绑定的direct-RLT CPU guard；当前formal仍运行，不发送训练中断信号，归还仅由原v5 owner执行。w044已启动r5正式，实际物理placement4–7和原正式resolved配置通过；首轮训练指标待确认。下方旧路线与准备状态为历史。**

**2026-10-01 09:35：r4原定2轮smoke已正常结束；首轮全组过滤、第二轮真实GRPO更新已证实（grad3.724、有效mask23.05%、CP1→CP2实际权重变化、两CP全部浮点参数有限）。原“两轮都有效”失败回执保留，独立receipt明确验收一轮有效更新，满足用户要求的真实参数更新。无追加smoke/改seed/关闭过滤。准备formal-only v5，从原固定SFT按原1000轮预算继续；当前w040已精确停止v4 outer进入资源归还，新v5尚未启动。详见`WAN_GOAL_RESUME_20261001.md`。**

**2026-10-01 08:49最新：用户起床后要求继续昨晚WM工作。三机RLT现场已刷新且均推进；SZ1/2维持，SZ3准备齐后借4–7继续π05 smoke，验收后原1000轮正式；归还链恢复为WM→原Dojo→原RLT。见`WAN_GOAL_RESUME_20261001.md`。以下睡前暂停和更早优先级均为历史。**

**2026-10-01 00:46再更新：用户睡前要求暂停WM、恢复原RLT，并检查三机。该指令覆盖以下WM优先顺序。WM heartbeat已删除，SZ3四RLT于00:50恢复派发；不再自动启动WM/Dojo。当前小结见`WAN_GOAL_PAUSE_20261001.md`。**

**2026-10-01最新指令覆盖下文历史恢复安排：WM第一、Dojo占卡第二，RLT暂不安排。不等待RLT首轮或补存checkpoint；新owner的`restore_rlt_after_dojo=false`，Dojo结束也不自动续RLT。**

- 仅深圳3，物理 GPU 4–7。环境、源码、模型、缓存与日志放 `/data/chenyiteng/projects/wan-goal-sz3`。
- 20:30实查 GPU 4–7 为 Dojo π0.5 全量评测，2072/6300 回合、166次成功，RLT已暂停；此数字只是切换前快照。
- 用户明确：**准备好后暂停 Dojo，WM结束再续 Dojo**。保留原结果和 resume manifests；Dojo最终结束后RLT仍保持暂停，待用户安排。
- 旧outer退出曾自动恢复RLT；该交接已完成。当前新owner复用已借的同一cycle，WM结束清理后续原Dojo，不再停启RLT或等待其首轮。细节见 `WAN_GOAL_RESOURCE_SWITCH.md`。
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
5. WM正常结束或失败：只清理新WM owned进程，核4–7释放，续原Dojo。Dojo结束后RLT仍保持暂停。

## 当前状态

- **09:15最新**：r3第一轮32环境rollout已完成79.36s，actor训练前向读mergerfs生成缓存文件暂不可见而失败，没有有效更新。原owned清理、Dojo退出和原RLT归还均核成功。仅改新run编译缓存为独立tmpfs，实际双CPU进程编译/前后向梯度检查通过，原两π05 YAML SHA不变。w023唯一启动v4 owner及WM r4，沿动态active pointer刷新；仍须严格smoke验收才正式。定时任务合并为原ID rlt一个检查，服从唯一owner。

- **08:57最新**：w006准备成功，四原RLT绑定完整CP125；w007唯一启动v3 owner，08:56已释放原RLT并进入π05 smoke。w011核actor/env/rollout均实际放在物理4–7，placement通过，尚在加载，未验收有效更新。OFT学习证据复用；正式仍受严格smoke gate控制。

- **00:53最新**：本次WM已按用户要求暂停，π05仅完成模型加载，未验收真实GRPO更新，未进入正式。两份WM运行目录、OFT学习证据和全部部署环境保留；SZ3四RLT已从CP25恢复派发，首轮尚待核验。下一次需明确借卡后再从π05 smoke继续。

- **2026-10-01 00:43最新**：OFT两轮真实更新、CP1/CP2完整性和权重变化已独立验证；原外层monitor在退出阶段遇/proc/environ权限异常并误触发清理，原退出-15保留。`learning-reconciled.json`结合实际学习与释放证据接受已完成OFT学习，不重跑其预算。monitor修复经8项CPU检查；新v2 owner已启动π05 smoke，仍要求其正常退出及真实参数更新才进正式。输出`runs/wan-goal-sz3-20261001-r2`；复用同一已借cycle，RLT保持暂停。上次Git31703e08；本次修复待增量发布。以下准备记录保留为历史。

- 官方模型清单已核，三套合计36.78GB（未含Python环境与LIBERO资产）。
- SSH主机指纹/账户身份已验；现场只读快照已保存。
- 独立目录和固定源码已建立，20:38起安装OFT环境并下载固定模型；尚未停止Dojo、尚未启动WM训练。
- 用户已明确接受π0.5单视角适配，要求真实更新通过后启动正式训练。两文件补丁已应用，三套独立配置已于21:23通过服务器Hydra解析；目标环境仍在安装。原生预测H10、执行C8、采样M5；不把网络H改成8。
- 22:02网络进展：读取服务器网络README后，模型与本任务安装切至实测较快的现有7897代理；模型持续约6–8MiB/s，保留TLS断连续传。PyTorch同版本官方wheel已校验并复用装入两venv，官方安装已恢复。具体结果见 `WAN_GOAL_NETWORK_20260930.md`；源码/日志的公开发布仍待本轮准备验证，不能称已推Git。
- 23:33：两套环境安装完成；765运行文件/7大文件SHA/742初态及tokenizer通过，OFT/π05实际模块导入与π05单视角CPU接口通过。π05的dm-control/MuJoCo已对齐LIBERO约束；剩余官方覆盖和非当前入口的依赖警告逐项保留，详见`WAN_GOAL_ENVIRONMENT_NOTES.md`。准备发布后切入GPU smoke；仍未切卡、未发生GRPO更新。
- 23:53：首个发布提交已生成但推送失败，原因是初始精简克隆缺旧版本Git对象；完整固定上游历史已下载，正补齐新的发布副本。切换预检另发现旧watchdog需要真实路径，仅在此接口将`/data`别名解析；实际只读绑定复验通过，无信号。启动配置/预算/停止条件见`WAN_GOAL_LAUNCH_20260930.md`。

粗日志见 `WAN_GOAL_RUNLOG.md`；每条真实远端命令、stdout/stderr、退出码、时间、SHA256保存在本地E盘证据目录，文档索引不包含任何密码或令牌。
