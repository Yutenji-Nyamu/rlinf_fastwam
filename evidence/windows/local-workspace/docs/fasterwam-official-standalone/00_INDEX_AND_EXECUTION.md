# HUSTVL Faster-WAM official standalone on SZ-H100

日期：2026-09-02

## 目标与来源锁

- 目标：在 SZ-H100 上按官方路径跑通 `hustvl/FasterWAM`（arXiv `2608.04404`）的
  RoboTwin 单任务单 episode 闭环推理。
- 首轮只做 official standalone，不接 RLinf，不改现有 π0.5 / Fast-WAM 训练。
- 源码、checkpoint、stats、RoboTwin revision、resolved Hydra 配置、命令、资源和结果均记录在
  `evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md`。

## 首轮冻结口径

- 模型：官方 RoboTwin release `step_029355.pt` 和同目录 `dataset_stats.json`。
- 模型合同：三相机、14D state/action、H32、M10。
- 环境合同：官方 `demo_randomized`、`unseen`、`replan_steps=28`。
- 运行预算：空闲物理 GPU 之一、`move_stapler_pad`、1 episode。
- 不下载训练 dataset；不执行训练用 SparseActionDiT initialization。

## 停止条件

- 成功：episode 自然结束并产生官方结果/视频与可核对成功标记。
- 失败：明确 traceback、模型/环境合同错误、CUDA OOM 或有界运行超时；保留最小证据后只做一个窄修复。
- 不停止或修改任何其他用户/现有训练进程。

## 终态

- 2026-09-02 official standalone已跑通：GPU3，`move_stapler_pad / demo_randomized / unseen`，
  实际seed100000，`1/1`成功，进程自然exit 0。
- 总wall约2分20秒，其中模型组件加载61.24秒；成功视频14.5秒/145帧，GPU3已释放。
- 可直接查看[成功视频](evidence/official-move1-20260902/episode0_success.mp4)、
  [resolved配置](evidence/official-move1-20260902/eval_config.yaml)和
  [完整实施账](evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md)。
