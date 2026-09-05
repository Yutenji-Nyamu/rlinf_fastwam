# GPU6：Sidney π0.5在线成功BC两轮smoke合同

用户已授权独立实现、简查及单卡smoke；不授权正式训练。本合同继承原BC v8容量设置，仅新增Sidney模型/任务适配，不运行DVAC、不用GRPO训练后checkpoint。

## 配置与预算

完整真实组合文件：[PI05_BC_SMOKE_RESOLVED_20260905.yaml](PI05_BC_SMOKE_RESOLVED_20260905.yaml)；逐叶对照：[PI05_BC_RESOLVED_DELTA_20260905.json](PI05_BC_RESOLVED_DELTA_20260905.json)。必须通过服务器validate_cfg和下方检查后启动。

| 项目 | 两轮smoke |
|---|---|
| GPU／环境 | 物理GPU6，train32×1，eval8×4；原shared Ray不重启 |
| 模型／任务 | Sidney原SFT e49e2ab转换资产；move_pillbottle_pad；H200／C50／D14／M10／三相机 |
| 训练 | 冻结VLM，只训expert及相关投影/条件层；原生FM；无图像增强／D0权重0 |
| micro／global／U | 32／1024／10；单卡每Adam累积32个micro |
| 优化器 | LR2.5e-5恒定，AdamW betas0.9/0.95、eps1e-8、wd1e-10、clip1；原生混合精度 |
| 收集预算 | 64条尝试、最多12800个环境action slots、最多256个实际决策query；仅成功episode入累计池 |
| 更新预算 | 池非空时20次Adam、640个micro、20480次chunk呈现；不是20遍池 |
| 评估／保存 | 每轮固定32条，共64条；每轮native＋full＋replay/learner，共两代 |
| 恢复／初始数据 | resume=null、ckpt=null、原SFT、空在线池，绝不续π0／DVAC smoke |

## 精确路径与命令

源树：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc`，分支`codex/sz-pi05-online-bc`，由原BC2467d997建立；启动实际commit记录在runtime/source-head.txt。

输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1`。

```bash
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
run_dir=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH="$robotwin"
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR="$run_dir" RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset CUDA_VISIBLE_DEVICES LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
cd "$root"
timeout --signal=TERM --kill-after=180s 5400 \
 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
 "$root/examples/embodiment/train_embodied_agent.py" \
 --config-name robotwin_adjust_bottle_online_bc_openpi +online_bc_model=pi05_sidney
```

实际wrapper另将stdout/stderr写driver.log、记录PID/起止/退出码；使用继承的GPU6 observer每5秒只读采样。新run使用RLinf自动独立namespace与显式新树工作目录，启动后核验；不修改其他namespace。

## 资源预计、观察与停止条件

- 容量参照：π0 v8同32train/8eval峰69.52GiB，不能直接保证π0.5。占用一张H100 80GB；π0.5真实峰值由本次测量，不假称已有通过数据。
- 预算估计：约40—70分钟、0.7—1.2 GPU小时，90分钟硬上限；π0.5 M10耗时尚未实测。两代checkpoint预计约40GiB量级，预留60GiB写盘空间；实际文件大小验收后回填。主机内存按100—200GiB额外容量预留，只是容量规划，非测得峰值。
- 启动前重新确认GPU6无其他compute、/data及RAM余量；基线/新树HEAD和源文件hash、resolved一致、目标run不存在。
- 观察首次采集、U10、同步、eval、保存和第二轮推进；记录GPU/RAM/FD、FM loss/grad、成功池和checkpoint。明确fatal/OOM/非有限loss/同步保存失败则失败收尾，不自动降并行/batch/U或循环重启；90分钟到期仅终止该run driver，若有残留只核对该run精确owned namespace/PGID后处理。
- 完整两轮exit0、20次Adam、两次fixed32、两代预期文件及replay/learner读回后才算smoke通过。两轮短测不是100轮稳定、方法效果或完整worker恢复验证，不自动接formal。
