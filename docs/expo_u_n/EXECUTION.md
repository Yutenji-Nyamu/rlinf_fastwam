# EXPO U/Norm 执行记录

2026-10-09，深圳3机chenyiteng，固定host-key通过；只到smoke。

## 启动前锁定

- 21:28精确暂停原RLT6/7，保留CP150完整标记与SHA；两个旧driver退出15、cleanup_error=null；namespace actor=0、6/7计算/图形上下文=0。统一CPU owner2707775，仅保留4/5原槽位。
- 源码：`/data/chenyiteng/projects/expo-u-n-sz3-20261009/source`；操作：`/data/chenyiteng/deployment-20261009/expo-u-n-smoke-v1`。实际基线由SZ2原source直拷；Windows只覆本次差量，发布从3685f54df独立分支。
- 数据：SZ2原clean50完整复制到上述项目`assets/clean50`，53文件、840566780字节、50条/4863帧；训练/评估种子沿原文件。未筛新种子或更换规划器。
- CPU：10项信号/回放/梯度/双trick检查、7项core、10项driver、5项cache通过，共32项。微批梯度测试固定同份epsilon；历史cache测试按其直接脚本入口执行。未增加GPU smoke轮数。
- 新模块Ruff F/I与format通过；原Clean源码保持最小修改。
- 资源：只暴露物理6/7 UUID，模型复制2卡；复用6卡已有EGL profile的DSO标记（mask64），SAPIEN RenderSystem(cuda:0)映射物理6。共享Ray和4/5任务保持原状。

## 最短真实smoke

命令：
```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/expo-u-n-sz3-20261009/source/examples/embodiment/smoke_expo_signals.py --inputs /data/chenyiteng/projects/expo-u-n-sz3-20261009/source/examples/embodiment/config/expo/fm_u_sz3_smoke.json --run /data/chenyiteng/projects/expo-u-n-sz3-20261009/smoke-v1 --max-episodes 3
```

- 共用一次模型加载和真实采集；原ODE10附Norm，选定父候选补U的ODE5。最多按原种子顺序3回合，遇真实成功H50即停止采集。没有可用成功H50即明确失败，不制造成功标记。
- 同一真实动作/观测分别保存U、Norm的历史trace；各一次B64、Q20→FM1→Editor1→alpha1。两个target同时开只用于覆盖；四份配置可分别开FM或Editor。
- 为保证实际覆盖非均匀权重，短测每个batch保留2个真实online窗口，其余仍正常抽样；正式回放没有此覆盖逻辑。
- 预热/正式预算/固定评估不在本次短测内。一次完整保存与同进程load_state_dict检查；不是新进程续训验收。
- 停止条件：两信号更新及保存通过即退出；3回合没有成功H50、非有限值、绑定越界或30分钟限时则停止并记录。
- 输出：`smoke-v1/{events.jsonl,complete.json,failure.json,checkpoint.pt}`；操作回执`smoke-launch.json`、`smoke-finished.json`、`smoke-passed.json`。

## GPU 实测结果

- 单次进程，首个原训练seed131330在57个物理动作成功，提供真实成功H50；没有扩展采集或重复smoke。
- 原ODE10主动作逐元素相等，U补算前后主RNG相等；真实信号均有限且非恒定。SAPIEN显式绑定物理6，计算仅6/7；现场0–3无C/G上下文，4/5原WM进程不变。
- U和Norm各完成一次B64/Q20→FM1→Editor1→alpha1；合计Q40、FM2、Editor2、alpha2。每次约270秒。每种信号的FM与Editor合并在同一次更新作通路覆盖，不是四组独立效果实验。

| 信号 | FM loss | FM grad norm | FM权重范围（含demo） | Editor grad norm | Editor权重范围（含demo） |
|---|---:|---:|---|---:|---|
| U | 0.025915 | 0.266745 | 0.711457–1.587899 | 8.702312 | 0.802625–1.197375 |
| Norm | 0.043831 | 0.214669 | 0.980255–1.462369 | 8.159132 | 1.000000–1.197375 |

- 每次FM均有208个抽样参数发生变化，冻结前缀梯度为0。Norm的FM及Editor各有一个online窗口被dropout恢复为1，未重归一，故整batch均值可略偏离1。
- 22:16正常退出RC0，完整checkpoint保存及同进程状态/计数恢复通过，冻结前缀参数不变；owner cleanup.ok=true，remaining=[]，6/7的C/G上下文均清零。没有启动正式EXPO训练。
- 22:19已把6/7加入原统一owner的独立候补槽：6卡原Clean、7卡原Combo，均CP150；新CPU owner3948394，4/5原WM等待槽未变。旧owner正常退出略超过首个25秒等待，检查退出回执后续接完成，没有强杀或重跑smoke。
- 22:23回卡核验通过：两个槽均RLT_RUNNING；driver6=3948416、driver7=3948417，owner/driver身份匹配，候选namespace分别仅有各自6/7卡上下文，无越界。日志于22:23:29/31确认从各自CP150开始载入；此时处于恢复初始化，尚未据此声称完成新训练轮。
- 精确owner/driver、CP路径、namespace、释放证明与指标见`smoke-evidence.json`、`return-evidence.json`；服务器原始回执保留在本事务目录。

## 发布

实现提交`8a1e96c87`已推送`codex/expo-u-n-20261009`；继承基线`3685f54df`。发布范围为审过的源码、四份配置、短测入口、测试和轻量文档，无凭据/模型/数据集。
