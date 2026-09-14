# GPU7 反转组收尾与 Clean 恢复至600轮

用户于09-14授权停止GPU7方向反转RLT、上传主要产物并交付ZIP；Clean继续到累计600轮。先完成恢复准备，再停旧启动新，之后归档。只使用chenyiteng和固定SSH指纹；共享Ray、GPU0–6及其他用户保持。

## 执行结果

- 旧方向组最终采集R434，MA10/20/50为95%/90.625%/89.75%；fixed R400和R425均20/20，最近checkpoint R425。最后采集不代表所有尾部优化tag都到同一轮，精确截止见ZIP内SUMMARY.json。
- 16:58:49.996确认旧namespace与GPU7清空；16:58:50.175新Clean wrapper已启动，间隔约0.18秒。这是进程切换间隔，模型恢复和首轮计算另需时间。
- 17:05:03正式恢复验收通过：第一轮R351采集8/8；恢复前critic计数155970，首轮完成critic520次、actor260次，回放从45299增至45403，ready_for_online=1。没有重新预热。
- 新driver1523072、12个本组actor；trainer实配相对冻结配置diff为空，源码SHA一致，保护进程身份保持，无fatal/非有限指标。

## 恢复范围与配置

原Clean采集曲线到R373，最新完整checkpoint为R350，故从R350恢复，再执行250轮到累计600。原351–373作为旧分支保留；新351以后重新采样，绘图独立展示。恢复包括模型、两个优化器及学习率调度器、target、回放池、RLT训练计数和预热锚点。既有实现未保存环境模拟状态和rollout RNG，回放加载还会重置其采样器/NumPy RNG；本次是正常训练恢复，不是逐位重放。

|项目|本次设置|
|---|---|
|算法与生产源码|Clean，DVAC off；30349428c37a008b95342121c1455debfeb4805e，无源码改动|
|工作树基点|77d0a673055db6210bdcdbf11e5ef317d18b990e，仅附带既有文档|
|交互/训练预算|8env×1，B512/micro256，UTD5，critic:actor=2:1|
|学习率|actor/critic均1e-4，constant|
|任务|adjust_bottle；current causal-AR Stage1；200动作上限、执行C10/模型H50|
|回放/预热|cache80000，原20k采集+30k critic初始化；恢复计数后不重跑|
|原BC/Q课程|7/.05→2.5/.45，保持|
|评估/保存|每25轮fixed20/保存；下次R375|
|实际配置变化|10项运行身份、目录、GPU6→7与resume_dir；算法部分完全相等，种子文件SHA相等|
|停止条件|累计600轮、用户明确中止或不可恢复运行错误；无性能阈值|

CPU准备校验了9份关键恢复源码SHA、严格合同与计数、DCP元数据201项及分片范围、target张量、45299份回放文件与索引及首末样本。初版辅助校验器误将轨迹文件数硬编码为45300；修正为metadata/index/文件集合/trainer实际计数交叉核对后通过。checkpoint和训练源码未作修补；服务器实际校验脚本保存在本地verified中。

## 运行入口

- root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-clean-resume600-20260914`
- run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-resume350to600-phys7-20260914-v1`
- namespace：`RLinf_rlt_clean_resume350to600_phys7_20260914`
- branch：`codex/sz-rlt-clean-resume600-20260914`
- checkpoint：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-control-phys6-fresh600-20260912-v1/current-single-gpu-control-phys6-fresh600-20260912-v1/checkpoints/global_step_350`
- 唯一运维根：`/data/chenyiteng/results/server-maintenance-20260914/rlt-clean-resume`；启动器`ops.py dispatch`已经执行，禁止重放。
- 精确命令与冻结实配：run的`runtime/command.txt`、`runtime/resolved.yaml`；验收`rlt/STARTUP_VERIFIED.json`。

## 归档和可视化

旧方向组[云端归档](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/fefe0f34935887ada5688daa628f6b1233fd44c1/docs/experiments/rlt-dvac-direction-closeout-20260914)已推原分支，远端与本地HEAD均`fefe0f34935887ada5688daa628f6b1233fd44c1`，生产源码无差异。

ZIP：`rlt-dvac-direction-single-gpu600-R434-closeout-20260914.zip`，4,982,005字节，42文件；SHA256 `727ceabb4b37a9f62488a720b3b7a9038398d4dd6967b019f8df48913b651d01`。包含120个指标tag/40746点、CSV/JSON、原TensorBoard events、driver日志、实配/命令、源码标识、checkpoint目录及raw/MA10/20/50/fixed五图。CRC和全部文件SHA及MA重算通过。模型、优化器与回放保持在服务器，不入轻包。

本地全部产物根：`E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-clean-resume-20260914`。

- 旧组ZIP和发布回执：`closeout/rlt_direction/`。
- 新组冻结配置与验收：`verified/rlt/`；新分支发布回执：`publish-new/publish-receipt.json`。
- 最新全服务器快照：`postcutover.json`；简要5图/CSV/ETA与资源摘要：`plots/index.html`、`plots/summary.md`。
- ETA采用自身后期轮间walltime，排首次启动并计入评估/保存开销；恢复初期参考旧Clean后期速度。短新组只作粗估，不把窄轮速区间当可信预测区间。

换组与首轮恢复验证完成后结束启动盯跑，下次按上述新路由只读刷新。
