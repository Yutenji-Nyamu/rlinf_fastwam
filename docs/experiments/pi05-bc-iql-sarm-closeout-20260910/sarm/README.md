# π0.5 BC＋RynnValue-SARM · 4/U5 · R100归档

训练已完成100轮，训练进程返回0，R100 checkpoint存在；归档时旧namespace没有ALIVE actor，旧评分器没有活跃身份。本次没有停止进程。

| 项目 | 终点 |
|---|---:|
| 每轮采集/actor更新槽 | 4 / 5 |
| 采集成功率 R100 | 50.00% |
| 采集 MA5 / MA10 | 55.00% / 50.00% |
| 固定32评估 R100 | 40.62% |
| train / wrapper / scorer cleanup 返回码 | 0 / 92 / 1 |

Wrapper92: historical scorer shutdown raced with empty argv; read-only closeout confirms no old live scorer. Original evidence unchanged.

建议先看 [对照与诊断图](analysis/evidence/sarm-iql-diagnosis-20260910/index.html)、[诊断正文](analysis/BC_SARM_IQL_DIAGNOSIS_20260910.md)。对照包含同设置clean BC；这是单seed、固定32场景重复评估。

- `raw/runtime/`：实际配置、argv、合同、driver日志和完成记录。含其他用户身份的保护快照已排除。
- `raw/tensorboard/`：全部原始event与实际配置；`scalars.json/csv`：完整标量序列。
- `source/`、`source-lock.json`：经过核对的方法源码/测试及SHA256。
- `checkpoint-inventory.json`：所有checkpoint文件的路径、大小、inode、mtime等元数据；没有读取或打包大权重。
- `MANIFEST.json`：本包每个文件的SHA256；ZIP另有外部校验回执。

权重、optimizer/replay、成功数据、评分缓存和视频继续保存在服务器。本ZIP用于分析与复核，不是完整恢复备份；没有删除任何实验文件。

服务器输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-bc-sarm-4u5-k2-gpu7-formal100-20260909-v1`。
归档源码HEAD：`d72e7fcf92995de3760efe0c8eba56abfa455462`；个人分支：`codex/sz-pi05-online-bc-rynnvalue-rabc-20260909`。
