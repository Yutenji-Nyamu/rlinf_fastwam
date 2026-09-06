# π0.5：100轮轻量实验产物

任务：`move_pillbottle_pad`；算法：GRPO；采集：2026-09-05T13:01:40.175520+08:00。

- **已自然结束100轮，exit=0**；step100 checkpoint清单已确认，权重留在服务器。
- 最终训练：**165/256 = 64.45%**；末10轮均值 **66.91%**。
- 固定评估：step100 **19/32 = 59.38%**；最佳首次在step70，**26/32 = 81.25%**。
- 首10轮→末10轮训练均值：41.68% → 66.91%。固定32条评估样本较少，仍有波动；没有同协议step0，不将首轮当SFT基线。
- 本阶段预算：25,600条训练轨迹、200次optimizer调用、640条fixed评估、10代checkpoint。
- 配置：64环境×4轮采样，G8，GB1024/MB32/update2，H50/C50/M10，noise0.5，horizon200；GPU4/5。

## 怎么看

1. 双击 **[dashboard.html](dashboard.html)**：离线交互图，可切换指标、查看具体数据，不需要网络。
2. 手机直接看 **[成功率](01_success.png)**、**[优化](02_optimization.png)**、**[资源](03_resources.png)** 三张PNG。
3. **[training.csv](metrics/training.csv)**：完整1—100轮；**[evaluation.csv](metrics/evaluation.csv)**：20次固定评估；**[all_scalars.csv](metrics/all_scalars.csv)**：全部49项TensorBoard scalar导出；**[resources.csv](metrics/resources.csv)**：2432次资源采样。
4. **[完整配置](source/resolved.yaml)**、**[原始启动命令](source/command.sh)**；`source/`附源码锁等原始文本。
5. `logs/`保存关键driver尾日志、完成状态和错误计数；**[checkpoint清单](checkpoint100_inventory.csv)**仅列step100路径、大小、时间。

图中横轴为已完成训练轮数，不是optimizer调用数。训练每轮256条，固定评估每次32条；真实曲线从step1开始，不虚构step0。资源是运行期间的采样记录，显存峰值可能落在采样间隔内；整机内存包含其他任务影响。

这是 **1—100轮不可变收尾包**；随后接续到200的运行不混入本包。未包含模型/checkpoint本体、视频、数据集、完整Ray日志或TensorBoard二进制。

服务器原始运行目录：

```text
/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1
```
