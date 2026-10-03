# WMRL 云端发布回执

2026-10-03 23:50:44 CST，新OpenDW实现与轻量证据已提交、推送并经`git ls-remote`核对一致。

- 仓库：[Yutenji-Nyamu/rlinf_fastwam](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/robotwin-opendw-wmrl-20261003)
- 分支：`codex/robotwin-opendw-wmrl-20261003`
- 提交：`0d9e90ae4eb6ecb6103ad10ebace5bbef373adcb`
- 77个文件，916,961 bytes；基于`2151a08ee1bd75df1bef0d8190e594bd5c7f7977`。包括规范env适配与runner卸载屏障、单卡/四卡owner与完整CP借还、WorldArena RM接线、资源遥测、可视化/学习审计、配置生成器、研究上下文和真实轻量实验回执。
- 四卡部署代码与CPU测试回执逐SHA核同；运行checkout及已有dirty未改变。本机根目录不是配置了远端的发布仓库，因此使用服务器独立`publication/opendw-v2`，没有批量提交本机无关文件。
- 本轮公开轻量证据包括N8/N16的exit0、RLT归还首轮失败、部分资源/奖励统计、36项CPU检查（29项执行链＋7项分析）及N64两段采样合同、owner启动身份。未将全滤/零梯度、owner启动或CP存在写成有效学习。
- 权重、检查点、reset数据、原始日志/TensorBoard及生成视频未放Git；没有新的独立异地备份回执。

服务器回执：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/publication/opendw-v2-staging/published.json`。发布manifest SHA256为`2142e1053cb553d16ffd40f3f6bfbfa442163e3caa43f2414bcdbc8330b9d8e9`，精确diff SHA为`129c67799455952f6466aacefcfe6b82ee4d0066175ebde22347d975f9cd79be`。

旧LIBERO WMRL也已从本机实时复核：远端`codex/sz3-wan-goal-20260930`仍是`21b5d590a9116d94e38c139a3ec4aa9723abbcc3`，包括此前评估、OOM审计与授权修复/监控。详见[本机及云端盘点](git_inventory_20261003.md)。

Git中的`publication_manifest_v2.json`是提交前的冻结清单，其`committed=false,pushed=false`明确仅描述暂存时刻；实际发布成功以本回执和远端提交为准。
