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

## 待完成

GPU smoke结果、精确释放、原逐卡RLT回卡、推送提交在完成后补入。
