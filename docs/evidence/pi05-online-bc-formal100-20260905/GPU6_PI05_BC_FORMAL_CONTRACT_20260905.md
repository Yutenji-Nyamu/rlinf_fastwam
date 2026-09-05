# GPU6 π0.5 online BC正式100轮合同

## 已获授权及锁定配置

用户2026-09-05明确授权启动、沿每10轮保存，确认启动后即报告。完整[resolved YAML](PI05_BC_FORMAL_RESOLVED_20260905.yaml)已在服务器Hydra compose＋validate_cfg通过；[现场及11个叶值差异](PI05_BC_FORMAL_PREFLIGHT_20260905.json)。相对通过的π0.5 smoke只有max_epochs 2→100、eval 1→5、save 1→10、optimizer总步数20→1000和7个名称/输出路径叶变化；不变更任何采集、学习或模型参数。

- 源码：`codex/sz-pi05-online-bc@912bc6907d39a0eec1eb98a6d4c9358e69791924`，clean；生产源码提交`653fe0fb`。不合并GRPO、不改依赖/RoboTwin。
- 起点：`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`，原Sidney SFT；resume_dir/ckpt_path=null，新成功池。不使用smoke或GRPO权重。
- GPU6，32训练环境×1串行，micro32/global1024/U10；LR2.5e-5，constant、warmup0；expert及投影/条件层可训练、VLM冻结；关闭训练图像增强，原生混合精度。
- pillbottle/H200；模型M10/C50/14D、三相机224、state token200、绝对qpos/MEAN_STD，无额外delta。
- 只成功数据、累计池query均匀有放回；demo_weight0，无DVAC、无PPO/GRPO/critic。初始噪声随机的ODE采样，不加入GRPO的flow-SDE探索噪声。
- 固定评估8×4＝32个原Sidney初态，每5轮；保存每10轮，无自动轮换/删除。每轮成功数据追加不等于每轮模型checkpoint。

## 精确命令及输出

服务器输出目录（启动前确认不存在）：

```text
/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1
```

实际命令见同packet `wrapper.sh`；环境配置与smoke一致，物理GPU由RLinf placement指定，不能再全局设置CUDA_VISIBLE_DEVICES=6：

```bash
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH="$robotwin"
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset CUDA_VISIBLE_DEVICES LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
cd "$root"
timeout --signal=TERM --kill-after=180s 172800 "$venv/bin/python" -u \
  "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  +online_bc_model=pi05_sidney \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=1000 \
  runner.logger.experiment_name=pi05-pillbottle-bc-u10-eval8x4-formal100-gpu6
```

通过独立`nohup setsid bash <run>/runtime/wrapper.sh`启动，wrapper记录PID/时间/exit，driver.log在该run内；沿既有BC正式48h硬上限。每5秒只读资源CSV是run-local日志记录，不是助手长期轮询/另建自动化。

## 预算、资源与停止边界

- 3200次训练episode尝试；每条最多4个query，最多12800新query记录。每轮从成功池呈现10240个chunk、320个micro、10个Adam；100轮最多1000Adam/1,024,000样本呈现，空池跳过更新，不保证达最大数。
- fixed32×20＝640评估episode；Step10…100共10代checkpoint，每代基础权重18.907GiB，加当时累计replay/learner；基础权重总189.07GiB，暂留220—250GiB工作空间，回放增长有不确定性。
- 23:36启动前GPU6=11MiB/0%、无compute；RAM available约1317.4GiB，/data可用875.07GiB，/home1246.15GiB。为本run分配空间不等于独占剩余盘，现役Sidney仍继续保存。
- 已通过smoke同并发/同batch峰73.74GiB/79.65GiB；不保证100轮无增长。按两轮耗时估计约24—30h、单GPU24—30GPUh；48h仅已有安全时限，不是完成承诺。
- 正常到100轮结束；致命错误/OOM时不自动降参或重跑；观察到故障只处理本run的明确进程与namespace，不重启共享Ray。启动后确认worker/配置/阶段，无故障即回报，不持续盯守。
- 无删除、迁移、依赖升级；GPU4/5的Sidney及其他用户不动。两轮smoke不等于长程稳定或完整worker恢复验证。
