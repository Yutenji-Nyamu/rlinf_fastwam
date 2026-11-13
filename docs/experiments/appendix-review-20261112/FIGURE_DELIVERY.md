# 绘图材料交付核对（2026-11-13）

- 新增离线图册：`appendix-figure-viewer-complete-20261113.zip`，解压后打开 `index.html`。
- 本图册含训练新旧曲线、仿真逐动作信号、GR-BC、K稳定性、时间和RynnValue图；附当前训练曲线原始标量。
- 168.4 MiB主包含原始补采数组、状态、特征、帧索引、PNG/SVG/PDF/CSV、统计与重绘脚本。
- GR-BC受控干预、四组原生随机化均包含RGB、状态和MCR特征。
- 更早RT2/Dojo/K原始采样、RynnValue和Heun数据见 MATERIALS.md 的既有包。
- 真机10段53窗口及新图保存于原私有数据仓库 data_ur5e_gwam，Release appendix-signals-20261112。
- 代码与说明：本目录 REPORT.md、TIMING.md、MATERIALS.md；采集重绘代码在 tools/appendix_signals。
- 训练曲线以记录轮次保存，新运行独立于历史配置变更段。
