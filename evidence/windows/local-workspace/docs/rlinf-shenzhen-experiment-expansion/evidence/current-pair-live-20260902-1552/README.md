# 深圳当前两条训练快照（2026-09-02 15:52 CST）

## 结论

- π0.5 clean GRPO：GPU4/5，完整 Step44/100，仍在正常运行；Step44 train success `207/256=80.86%`，MA5 `85.08%`，MA10 `86.25%`，Step40 fixed32 `29/32=90.63%`。KL/clip/grad均有限；最近10步中位耗时约24.75分钟。
- Fast-WAM plain GRPO：GPU6/7，完整 Step14；Step14 train success `38/128=29.69%`，MA5 `32.50%`，MA10 `31.48%`，Step10 fixed32 `14/32=43.75%`。随后在Step15第三次fixed32的环境重置中退出。
- Fast-WAM近因证据是SAPIEN/svulkan2/OIDN先报告`pthread_key_create failed`，随后`invalid handle`，最终EnvGroup触发`PyGILState_Release` fatal。训练标量有限、GPU无OOM、主机RAM/磁盘无压力；因此这是评估渲染线程/句柄路径故障，不是GRPO数值崩溃。最新完整可恢复checkpoint是Step10。

## 服务器

- RAM：2.0 TiB总量，约1.1 TiB available；memory PSI为0；swap约4.8/6.0 GiB。
- 磁盘：`/` 21%（余225 GiB），`/home` 40%（余1.4 TiB），`/data` 52%（余1.6 TiB）；inode均正常。
- GPU：4/5属于π0.5，约70 GiB/card；0--3和6--7空闲。其他用户没有GPU compute任务。
- 网络：Mihomo active；GitHub和Hugging Face经`127.0.0.1:7890`均返回HTTP 200。GitHub直连能建立连接但完整页面在10秒探针内超时，HF直连被reset。

## 图与原始材料

- `figures/01_success_rates.png`：逐步、MA5、MA10和fixed32。
- `figures/02_optimization_signals.png`：KL、clip fraction、gradient norm。
- `figures/03_resource_telemetry.png`：显存、GPU利用率和主机available RAM。
- `figures/interactive_success.html`：桌面交互成功率图。
- `metrics.json`、`summary.json`和`raw/`：绘图所需标量、resolved YAML、driver log和resource CSV。
