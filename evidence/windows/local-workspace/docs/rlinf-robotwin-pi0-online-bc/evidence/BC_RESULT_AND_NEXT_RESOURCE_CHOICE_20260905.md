# BC本轮结果与下一项资源选择

更新：2026-09-05 13:39 CST。正式100轮已获授权，但完整容量smoke仍未通过，**没有启动正式、没有启动DVAC**。源码修复已推；全部历史失败run保留，不从smoke权重续训。

## 1. 官方DAgger还是旧π0 GRPO更干净？

按职责选择，不切换整套算法分支：

| 部分 | 优先参考 | 不带进BC |
|---|---|---|
| π0监督FM、梯度累积、Adam更新 | 固定官方π0 DAgger | teacher、干预筛选、逐动作LeRobot标签 |
| 本机模型/环境接口、同步/保存和资源约束 | 已验证的π0 GRPO | GRPO优势、概率比率/clip、SDE训练目标 |
| 成功经验到监督样本 | 当前小型Collector＋累计Replay | 第二套训练框架或Q/V |

保留官方干净base上的独立BC分支。先前额外BF16/use_orig_params选择和未核对SFT包裹图属于本次适配审计不足，不是BC定义要求。官方DAgger原配置全模型训练，旧GRPO与BC要求expert-only；联合SFT直接调用decoder内部模块，因此FSDP边界需按真实调用图配置。原生保存算法没有被重新发明。详见[逐轮流程/继承项/预算/故障分析](BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md)。

## 2. v6真实结果：训练跨过，但容量未通过

```text
32条尝试 → 29条成功 / 87个query → 64个micro / 2次Adam更新
                                      ↓
                            首次32并行评估初始化失败
                            完整评估0 / checkpoint0
```

13:06:48启动，13:20:25 exit255。原生相机`take_picture()`报`RuntimeError: cannot create buffer`。GPU6的5秒采样峰值**81075/81559MiB＝79.17/79.65GiB（99.4%）**；主机available最低1739.91GiB、PSI0。没有PyTorch `CUDA out of memory`字符串，不代表原生渲染分配成功；高显存占用与buffer分配失败强烈支持当前共驻留显存容量/分配边界，不能从泛化报错单独区分容量与碎片等因素。

| 已定位阶段 | 本轮处理与证据 | 尚不声称什么 |
|---|---|---|
| 原生精度/增强 | 恢复官方null；增强显式off，撤额外cast | 增强对成功率完全无影响 |
| SFT/FSDP导出 | 真实调用模块wrap＋orig false；64微批/2次更新后进入评估，778键probe导出通过 | 全部生产恢复/长期训练已通过 |
| 1024文件句柄 | 32＋32环境probe明确EMFILE；只抬EnvWorker soft4096，v6无重复该报错 | 所有Vulkan错误都同一个根因 |
| 32train＋32eval同时驻留 | v6近满80GB卡，camera buffer分配失败 | 训练32并行本身装不下，或要改变BC loss |

证据：[最终只读JSON](BC_WINDOW_FINAL_REFRESH_20260905.json)、[v6首栈/真实显存总量/成功池计数](BC_V6_CLOSEOUT_20260905.txt)、[FD隔离复现](BC_FD_EXHAUSTION_PROBE_20260905.md)。29/32来自更新前采集，不能称BC学习提升；当前没有固定评估曲线可画。

## 3. 下一步建议：保持32并行，环境按阶段释放

后续讨论已更新：用户提出评估16×2，不选择本节offload作为当前首选。见[最新问答§1/5](BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md)，明确原32种子分批及U/global batch的新讨论；本节保留为未执行备选，不据此启动。

推荐只讨论并确认两个现有运行开关：

```yaml
env.train.enable_offload: true
env.eval.enable_offload: true
```

RLinf现有EnvWorker已在`interact()`结束后关闭训练环境、`evaluate()`结束后关闭评估环境；RoboTwinEnv.offload调用VectorEnv.close。下一次reset重新建立场景，不需要新建训练框架或修改FM。这样32训练与32评估先后使用显存，而非64场景同时常驻。

保持：单卡GPU6、训练/评估各32并行、串行1、fixed32 ID集合、M4/C50/D14/H200、MB32/GB1024/U2、LR/精度/图像开关等。代价：场景重建/缓存释放可能增加耗时；生命周期仍需一次完整两轮验证，不能提前保证无长期原生问题。

**本轮未打开这两个字段，等待用户确认新的资源安排**。另一条路径是16并行评估分两批覆盖原32不同ID，但需要显式种子分批；不能只加rollout_epoch重复16个固定状态。GPU7已有后续变体预留，不擅自用作额外评估卡。

下次从本断点继续，不重跑已完成的精度/FSDP/FD隔离探针。确认资源安排后更新resolved/合同及少量相关测试，再完整smoke；通过后才从原SFT和空成功池启动正式100轮。

## 4. 服务器同时刷新（13:39）

- BC v6已退出，GPU6约11MiB、GPU7约4MiB，无BC训练活跃。
- Sidney由另一个窗口接续到200，当前完整101/200、102轮采样1/4；100步fixed19/32，Step100双rank/full checkpoint在。续训runtime无所查错误，wrapper602620在；本窗口只读，没有干预。
- GPU4/5约51.2/51.5GiB，GPU0约9.6GiB；GPU1/2/3基本空闲。未定向查看其他用户进程。
- 整机可用RAM约1.7TiB，CPU约96%idle，swap2.7/6GiB、采样无即时进出；/data余612GiB。shared Ray原PID321933/322685仍在。最新原始证据：[13:39只读JSON](BC_REPLY_REFRESH_20260905.json)；13:31快照保留，不覆盖。

## 5. 文档与源码交付

- 生产代码：`72a926041867cbdbf2565ab66a14d742c59a0dad`（含FSDP修复5ae809d5）；10项单测及import/compose通过。
- 六次已结束smoke的轻量日志/config/resource/TB及FD脚本/证据已封存并push：`700b6846dbc2fe02398de05c044c8097cc974774`，83个tracked文件，其中75份原产物共387693 bytes（约379KiB），其余为索引/manifest与FD脚本/证据。远端同SHA、13:39本地工作树clean；该提交不改生产源。不上传成功图像、模型、checkpoint或其他用户材料，服务器原产物完整保留。
- [完整逐轮流程与组件排序](BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md#2-每轮准确做什么来自哪里)，[每轮2048样本呈现/2次更新的依据与局限](BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md#4-bc更新量与依据不能混淆三个单位)。
- [BC＋DVAC独立上下文](../01_DVAC_DESIGN.md)仅讨论：V[50]旁路记录、均值1权重作用于未归约FM；高V方向/强度、权重固定或重标定、global-z是否保留位置趋势仍待讨论。未实现/训练，不复制GRPO优势或RLT的Q/reference目标。
