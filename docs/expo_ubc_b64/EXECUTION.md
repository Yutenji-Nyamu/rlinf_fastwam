# EXPO B64 U-BC τ1/2/3

2026-10-16 用户授权收束 GRPO τ3、深圳3两组 Editor-U，并用空出的6卡启动本组三个实验；深圳1新 Clean 保持训练到结束。

## 配置与资源

| 组 | 位置 | 输入配置 |
|---|---|---|
| U-BC τ1 | SZ3 GPU4/5 | `examples/embodiment/config/expo/ubc_b64_tau1_sz3_60000.json` |
| U-BC τ2 | SZ2 GPU6/7 | `examples/embodiment/config/expo/ubc_b64_tau2_sz2_60000.json` |
| U-BC τ3 | SZ3 GPU6/7 | `examples/embodiment/config/expo/ubc_b64_tau3_sz3_60000.json` |

基线直接取 SZ1 正在运行的 `expo-clean-entropy-b64-sz1-g45-60000-1015/inputs.json`，与已发布 Clean 配置一致。沿用 B64、turn_switch、seed42、60000真实动作（含预热、不含评估）、200动作上限、H50/C10/ODE10、候选8+8、Q20→FM1→Editor1→alpha1、40动作/学习、学习率与评估种子。每组从同一原始π0.5和新优化器开始，无resume。

熵目标继续使用 Clean 的 `core.target_entropy=-295.32130774077405`，修正范围仍±0.2，算法源码未改。U只作用于既有BC/FM加权项（`signal.target=fm`），Editor使用普通目标；两个温度均设τ。复用已验收FM-U的dropout=.2及R1→R200退火，未新增算法实现。精确配置差异保存在每组prepared证据，此外只有各机数据/种子路径、显卡编号、独立输出路径与说明元数据的适配。

源码仍为已验收的a1b301179快照：SZ2使用 `/data/chenyiteng/projects/expo-fm-u-20261010/source`；SZ3使用 `/data/chenyiteng/deployment-20261013/expo-editor-u-tau123-v1/source`。两机全体port_source_manifest通过，训练/评估种子哈希和模型归一化哈希一致。SZ3演示路径复用此前已验收的clean50镜像。

每组CUDA绑定所列两张物理卡的UUID，图形绑定该组首卡；复用原有应用配置和pair_owner。运行目录为 `/data/chenyiteng/results/expo-ubc-b64-tau{τ}-{sz2或sz3}-g{卡对}-60000-1016`。两机事务目录均为 `/data/chenyiteng/deployment-20261016/expo-ubc-b64-tau123-v1`。

## 启动与退出

每组精确命令见prepared.json的command字段，统一形式为：

```text
<原venv>/bin/python -u -B <已验收source>/examples/embodiment/train_expo_formal.py --inputs <独立run>/inputs.json --run <独立run>/run --max-physical-actions 60000 --enable-evaluation
```

60000动作完成或用户停止时退出；既有数值异常/越卡保护保留。只清理该组driver及后代，确认计算与图形上下文释放后，各物理卡接回原有RLT候补。统一owner的fallback字段原样保留，未重建共享Ray。

准备时在服务器CPU上执行既有validate_inputs、源码/种子/归一化指纹校验及FM-only signal contract检查；CUDA未初始化。一次性操作有唯一事务回执，不能重放cutover。启动验收、旧组结论和发布回执随本轮执行结果补入本目录。

## 本轮结果与验收

22:38 SZ2旧GRPO τ3完成119轮后正常收束，保留CP110；末5评估均值61.875%、最新65.625%、峰值75%。主要日志和结论已推原GRPO分支f46931cd7。SZ3旧Editor-U τ1/τ3保留34295/34701动作、816/822次学习的完整断点；末5均值66%/68%，最后一次80%/65%，原始日志与结论见`docs/expo_u_n/closeout1016-tau1`和`closeout1016-tau3`。

三组新driver：τ1=2856028、τ2=3265763、τ3=2856184（UID20001，完整start身份见证据）。均新模型/新优化器初始化，初始断点完成，原生initial-base评估已有真实动作；尚无新学习结果。计算分别位于SZ3 4/5、SZ2 6/7、SZ3 6/7；图形分别在4、6、6。无失败/提前释放回执，原pair_owner越卡检查正常。

三机统一owner存活，12槽WAIT_EXTERNAL；RLT配置、环境及完整恢复点有效。SZ2/SZ3的fallback字段逐项保留，SZ1整个plan及原driver身份未变；SZ1 Clean在22:43快照已15980动作/359次学习，继续原预算。GPU 0–3未新增本任务计算或图形上下文。

验收为启动、初评及资源接续检查；U-BC有效性需等后续固定评估，对新Clean的观察也不能提前写成稳定提升结论。证据见两机prepared/acceptance和sz1-preserved；图表及本地完整证据见本轮E盘cutover1016目录。
