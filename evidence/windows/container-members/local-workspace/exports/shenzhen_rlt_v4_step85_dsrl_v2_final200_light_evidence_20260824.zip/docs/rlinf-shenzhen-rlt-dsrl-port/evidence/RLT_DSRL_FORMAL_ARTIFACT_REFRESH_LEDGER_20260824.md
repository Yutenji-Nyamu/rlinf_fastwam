# RLT / DSRL formal 产物与指标刷新流水账（2026-08-24）

## 边界

- 本轮只读刷新深圳服务器上已获授权的 RLT Stage 2 v4 与 DSRL v2 formal。
- 不停止、不重启、不修改训练配置、源码、checkpoint、Ray 或服务器文件。
- 只在 Windows 本地保存轻量日志、TensorBoard event、资源 CSV、派生 CSV、静态 PNG 和说明文档；不下载模型、replay、视频或全量 Ray 日志。

## 操作记录

1. 完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 当前专题路由和专题唯一事实源
   `00_INDEX_AND_MIGRATION_PLAN.md`，锁定 RLT v4、DSRL v2 路径及只读边界。
2. 通过固定 host-key 的 Paramiko 密码路径做普通账号身份探针。第一次误用了管理员密码，认证失败；随后使用用户已提供的普通账号密码重试同一路径，确认 `UID=1003`、host=`admin`。密码只存在于当前 PowerShell 进程。
3. 新增只读现场脚本
   `local_scripts/remote_commands/shenzhen_rlt_dsrl_formal_artifact_refresh_20260824.sh`，只收集 owner、完整 step、错误计数、轻量文件、TensorBoard、checkpoint 目录、产物类别和资源。

后续每次现场命令、下载预算、解析结果和 QA 在本文件继续追加。

4. 首次只读现场确认 RLT v4 完整到 Step 74、DSRL v2 完整到 Step 196；两者均无 traceback、OOM、
   worker crash 或 nonfinite。只下载两个 run 的 driver/resource/resolved/command/TensorBoard event，
   合计不足 3.5 MB。
5. 解析原始 Rich driver 表与 TensorBoard。RLT 的五位数 global transition 在 Rich 表中被截断，改用
   未截断的每-rank mean cache × actor world size 还原；没有猜测截断字符。fixed evaluation 只保留
   真实离散点，不插值。
6. 只读分解 checkpoint。确认 RLT Step25/50/75 均有 DCP metadata/shards、full weights、target、
   replay、两 rank sidecar 和 `complete=true`；DSRL Step65/130/195/200 均为 11 files 的 strict-resume
   分片，Step200 `update_step=104120`。
7. 14:05 CST 终态刷新：DSRL wrapper dead，`exit_code.txt=0`，完整 Step200/200，最后 fixed-12=8/12，
   `global_step_200` 保存完成；GPU6/7 约 5 MiB。RLT wrapper alive，原始文件快照完整到 Step85，
   Step75 fixed20=0/20 且完整保存，`update_step=0`、min-rank replay=6258/10000。
8. 资源 CSV 终止行在 driver 退出后只保留 host available、其余 cgroup/GPU 字段为空。制图 parser 仅做
   一项窄修：空终止字段读为 NaN，保留退出后主存回升样本，不伪造 GPU 数值。随后生成四张 PNG、
   final CSV 和 `summary.json`。
9. 最终图像 QA：四张 PNG 均成功解码，中文字体、图例、坐标、真实 fixed 点和 checkpoint 竖线可读；
   RLT 图明确区分 reference train 与 student fixed，DSRL 图明确标 stochastic fixed-12 和 clip 前梯度。
10. 生成轻量包 `exports/shenzhen_rlt_v4_step85_dsrl_v2_final200_light_evidence_20260824.zip`：
    约 1.1 MB、25 entries，`tar -tf`完整列出；不含 checkpoint、replay、视频或全量 Ray 日志。
11. 14:21 CST 交付前窄探针：RLT PID仍alive，TensorBoard完整到Step94，min-rank replay=6944、
    global total=14034、`actor_switch_rate=0`、`ready_for_online=0`、`update_step=0`；GPU4/5约
    20.9/21.2 GiB。DSRL PID dead、exit0，GPU6/7各约5 MiB；host available约1.93 TiB。

## 关键服务器路径

- RLT Stage 1：
  `/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1`
- RLT Stage 2 v4：
  `/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix`
- DSRL v2：
  `/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2`

## 证据边界

- RLT 仍在运行；本地曲线是 Step85 冻结快照，不自动声称为训练终态。
- DSRL 已自然完成；不把最后一个 8/12 小样本点单独称为退化。
- 未加载 checkpoint 做推理或 resume；本轮只检查结构、manifest、sidecar 和原始训练记录。
- 没有删除、覆盖服务器产物，没有停止或重启任何进程。
