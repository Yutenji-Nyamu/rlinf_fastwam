# RLT Stage 1 formal evidence

本目录保存 `adjust_bottle` full clean-50、固定 2k endpoint 的小型可版本化证据。大数据、
运行日志、资源 CSV 与 checkpoint 仅保留在 AutoDL 服务器；本文及同目录文件提供路径、hash
与启动合同，不复制大产物。

## 文件

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `dataset_manifest.json` | source revision、converter、50 episode/7,188 frame 与 canonical 文件合同 | `12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c` |
| `source_config.yaml` | 正式 source config | `c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c` |
| `formal_resolved.yaml` | 与 launch 同 env/override 的完整 resolved config | `5aa824fc9ac5cc361dace2b1162b2ef1bdf52adab3c775b8cd2e1ae468dfd67e` |
| `exact_command.txt` | 精确环境变量与启动命令 | `447d0a1b78d715fd55fece4b5a79d89c5899d213831632b36068ffca3ceb07ea` |
| `prelaunch_provenance.tsv` | compose 时的路径与预期 endpoint | `6e36b9601954681c3bb1f01cbcb11705f01797ef06739eb35eb9363cd2f3a532` |
| `run_provenance.tsv` | 实际 launch HEAD、hash、目录与 timeout | `d3f016e88e42ec92b7ee711d38e87173c42e2209879c9186999d0c15e51a1d3d` |
| `early_health.json` | 19:38 一次性早期训练健康快照 | `eb18a1622e53d202f10d9a05e1f64d47d85b8945a0947d3dd45f1cb116dc2f4f` |

## 服务器目录

```text
dataset:
  /root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
dataset manifest:
  /root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
run root:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
runtime/evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
expected endpoint:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
  robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
```

## 早期状态边界

`2026-07-29T19:38:41+08:00` 的固定快照是 step 172/2000、driver PID `650254`，
两卡各 26,447MiB；最近 20 步中位数 0.777s/step，`loss=rlt_loss=1.05`、
`vla_loss=0`、grad norm 1.03、错误计数全 0。它证明 run 已正常进入连续训练，不表示
endpoint 已完成，也不替代完成后的 reconstruction/reload/true-shuffled-zero 验收。
