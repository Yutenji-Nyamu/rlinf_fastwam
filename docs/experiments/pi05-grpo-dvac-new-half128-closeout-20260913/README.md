# π0.5 GRPO128 · DVAC new 收尾

用户主动中止。最终已记录采集R116，原计划200轮；最新固定评估R115：16/32。最近checkpoint目录R110；目录清单并非模型加载/完整恢复验证。runtime退出码为134。

200轮、64并行×2串行=128条/轮，G8、U2、B512/micro32、LR5e-6；每5轮固定评估32条、每10轮保存；原模型启动、rollout seed42+rank、训练/评估RNG隔离，历史环境实现。两级DVAC new，logV→chunk内+同scene group外层MinMax中心化，alpha_local/chunk=1/1，正负优势都乘，baseline整chunk裁剪保持；没有Prism。

固定评估每5轮32条；采用当前seed42协议，未拼接旧随机流。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[四张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放、图像、视频不入包，仍保留在服务器。本次未删除checkpoint；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-new-half128-seed42-formal200-phys45-20260912-v1`

生产源码：`930e230df94b3a5e2a34bfd4f1fba3d12d0924bd`；归档前HEAD：`354e9ae4216b371f449aeda8067f7667747ea5c1`；分支：`codex/sz-pi05-grpo-dvac-new-half-seed42-20260912`。
