# GPU7 π0.5在线BC＋DVAC：直接正式100轮合同

## 用户已明确授权

2026-09-05请求基于已跑通π0.5 BC迁入原π0 BC-DVAC增量，范围[0.5,1.5]、同参数正式；随后指定GPU7并明确**不需要smoke**。实际跨日执行至09-06；文件和run名保留任务起始日期20260905，不伪称09-05已启动。

只做服务器CPU单元/接口/配置检查，不安排GPU探针、smoke或试训。原SFT/空池/空标定状态，独立新run，不续GPU6权重、不动GPU6或Sidney/shared Ray/其他用户。

完整配置：[formal resolved](PI05_BC_DVAC_FORMAL_RESOLVED_20260905.yaml)；[逐叶校验](PI05_BC_DVAC_VALIDATION_20260905.json)；[28项CPU回归](PI05_BC_DVAC_TEST_CONFIG_20260905.txt)。源码以启动前Git提交和run/runtime/source-head.txt锁定，基线`6a93605d`、复用DVAC生产delta`736b1416`。

## 相对GPU6正式BC的全部变化

- 7个DVAC叶：enabled=true、tail_steps3、window5、alpha0.125、z_clip2、log_eps1e-12、std_floor1e-6。
- placement物理GPU6→7；独立worktree、新run名称/日志/数据/seed路径共9叶。训练/eval种子文件内容逐字相同，两处Sidney数据适配/原BC主配置/模型配置组逐字相同。
- 合计17叶；采集/模型/学习/评估/保存预算**无差异**。不继承旧π0 DVAC的M4或eval16×2，不升级环境。

方法：同次M10 ODE末3个endpoint，在归一化动作前14维计算每个h的总体方差V[50]；全部有效新query含失败episode贡献log小统计，BC图像/动作仍只成功入池。先用过去最多5轮统计映射本轮，再追加本轮统计。

```text
z = clip((log(V + 1e-12) - past_mean) / max(past_std, 1e-6), -2, 2)
w = stopgrad(1 + 0.125 * (z - valid_mask_weighted_mean_H(z)))
```

保证有效位置均值1、范围[0.5,1.5]，不强行拉满两端；首轮w=1，仍做U10并建立统计。入池后固定w，旧样本不重算、不反复计标定。`[B,50,1]`只乘原生`[B,50,14]`FM监督误差，随后保持原mask/分母/优化器。不开critic、Q/V、reward shaping、GRPO/πRL或新模型forward；DVAC状态随原checkpoint保存。

## 原参数与预算

| 项目 | GPU6 BC与GPU7 DVAC一致 |
|---|---|
| 原模型/任务 | Sidney pi05 e49e2ab native SFT；move_pillbottle_pad/aloha-agilex/H200 |
| 模型协议 | M10/C50/14D，3相机224，state tokens200，绝对qpos/MEAN_STD |
| 采集 | train32×1/轮，100轮＝3200次尝试，最多12800新query/640000名义动作槽 |
| 成功学习 | 累计成功query均匀有放回，demo_weight0；expert及投影/条件层训练、VLM冻结 |
| batch/U | micro32/global1024/U10：每轮320micro/10240呈现/10Adam；全程最多1000Adam、1,024,000呈现 |
| 优化器 | LR2.5e-5 constant、warmup0；AdamW beta .9/.95 eps1e-8 wd1e-10 clip1 |
| 图像/精度 | 训练增强关，必要预处理保留；原生混合精度，无全BF16强制 |
| 评估 | 每5轮8×4，同序32初态；100轮共640评估episode |
| 保存 | 每10轮，共10代，native+full+replay/learner；DVAC另加很小的标定sidecar |
| 资源布局 | 单H100，actor/rollout offload，env不开offload；no_shard FSDP、现有SFT叶wrap；Env FD4096 |
| 时限 | 原正式48h，不自动重启或调参 |

空成功池的轮次跳过优化，所以1000是满数据预算上限。V+w每query约400字节裸数据，12800query约4.88MiB；不保存额外整条去噪链、不增加新forward。信号同模型新run独立标定，不借旧π0/M4的统计。

## 精确命令与输出

```bash
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-dvac
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$robotwin"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment" ASSETS_PATH="$robotwin"
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset CUDA_VISIBLE_DEVICES LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
cd "$root"
timeout --signal=TERM --kill-after=180s 172800 "$venv/bin/python" -u \
  "$root/examples/embodiment/train_embodied_agent.py" \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  +online_bc_model=pi05_sidney +bc_dvac=bounded_half \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=1000 \
  runner.logger.experiment_name=pi05-pillbottle-bc-dvac-u10-w05to15-formal100-gpu7
```

由`nohup setsid bash <ONLINE_BC_RUN_DIR>/runtime/wrapper.sh`单次启动。stdout/stderr为该run/driver.log，wrapper记录PID/source-head/开始结束与exit；沿已有每5秒资源CSV，不建立助手长期heartbeat。

## 资源估算与结束条件

23:52只读GPU7空闲、RAM available1253GiB、/data余875GiB。π0.5 BC此前同并发smoke峰73.74GiB仅作起点估计，不冒充本DVAC实测；本增量不增加模型参数，但长程容量仍由实际运行检验。单run预算约220—250GiB（基础权重189.07GiB＋增长回放/数据），连同GPU6及Sidney后续保存总预留约655—715GiB，当前仍有余量；不以此承诺其他用户不会继续写盘。

沿BC约24—30h/24—30 GPUh粗估，不保证耗时或训练收益；48h为原正式硬上限。完成100轮正常结束；fatal/OOM/结构性错误不自动降低预算或循环重试。此次无DVAC GPU smoke，因此只能报告CPU/接口通过与实际启动，不能宣称新组合已完成端到端/长程验证。启动确认后报告，不持续盯守。
