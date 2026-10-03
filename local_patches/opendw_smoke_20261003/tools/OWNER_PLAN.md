# OpenDW owner plan 合同

部署前用深圳 3 的 RLT Python 做语法、配置和独立 review。下面是字段模板，`<...>` 必须替换为本轮已核验的绝对路径、版本和摘要；模板本身不能执行。

```json
{
  "mode": "smoke",
  "owner_dir": "/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/<unique-owner>",
  "cycle_dir": "/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/gpu4-cycle-v2",
  "repo": "/data/chenyiteng/<reviewed-opendw-rlinf-checkout>",
  "repo_head": "<verified-head>",
  "python": "<exact-python-in-RLT-cycle-plan>",
  "environment_file": "<original-RLT-runtime/environment.json>",
  "physical_gpus": [4],
  "source_sha256": {
    "<absolute-owner-script>": "<sha256>",
    "<absolute-service-script>": "<sha256>",
    "<absolute-reward-script>": "<sha256>",
    "<absolute-opendw-env>": "<sha256>",
    "<absolute-adapter>": "<sha256>",
    "<absolute-modified-runner>": "<sha256>"
  },
  "service": {
    "argv": ["<OpenDW-venv-python>", "-u", "-B", "<opendw_service.py>", "<reviewed-asset-and-port-arguments>"],
    "cwd": "<OpenDW-service-working-directory>",
    "environment": {"PYTHONPATH": "<service-dependencies-paths>"},
    "url": "http://127.0.0.1:<unique-port>",
    "startup_seconds": 1200
  },
  "trials": [
    {
      "key": "n8",
      "num_envs": 8,
      "config": "<absolute-resolved-n8.yaml>",
      "namespace": "opendw_<unique>_n8",
      "timeout_seconds": 2700
    },
    {
      "key": "n16",
      "num_envs": 16,
      "config": "<absolute-resolved-n16.yaml>",
      "namespace": "opendw_<unique>_n16",
      "timeout_seconds": 2700
    }
  ],
  "restore_wait_seconds": 1200
}
```

启动：`<RLT-python> -u -B <opendw_smoke_owner.py> --plan <reviewed-plan.json> owner`。owner 继承原 RLT 的环境；driver 去掉 CUDA/ROCR/HIP masks，在共享 Ray 上验证三角色 logical/physical 都是 4。service 单独设置 `CUDA_VISIBLE_DEVICES=4`，先 CPU-ready，借卡后才由环境 onload。配置的 `runner.logger.log_path` 必须位于 `owner_dir` 内。

owner 自动冻结 plan、config/environment hash、唯一 token、共享 Ray 地址与 namespace。任一档失败或超过 45 分钟不再开始下一档。正常结束及异常均走 `finally`，精确回收已登记 PID/UID/start 与 namespace/job，再核 GPU4 计算和图形上下文归零，写 release，恢复 RLT，最多 20 分钟核恢复首轮。若清理或身份核验失败，保留 `recovery-error.json` 供现场处理，不伪造归还成功。

`result.json` 的 exit 0 只表示程序退出成功，还要结合 placement、服务轨迹、actor/backward/optimizer 和 checkpoint 证据判断 smoke 结果；不把零优势/零有效梯度说成学到了东西。根 agent 根据资源结果选择正式参数。

## 接正式 plan

正式使用新的 owner 目录、已解析新配置和新 cycle。`mode` 设 `formal`，`trials` 仅一项，额外写明确的 `runner_iterations` 与 `timeout_seconds`，顶层 `smoke_evidence` 指向前次成功的 `final.json`。正式配置的 `max_steps/max_epochs` 必须等于写明的迭代数；episode 目前仍要求 C32 的整数倍。代码不会把 smoke 的“一轮”自动当正式预算，也不会自己修改正式参数。

第二次借 GPU4：新 cycle 的 prepare 加 `--previous-cycle <上次cycle>`，明确从其归还后的 `new_run/namespace` 接续，校验新 driver 和恢复首轮。helper 会同时检查新 run 的 checkpoint 和继承的 `resume_dir`，因此不要求等待下一次保存。该链严格限定同一个 GPU4 原任务；扩大到 GPU5–7 须单独绑定真实当前 owner 与配置。

## Review 重点

- `register_actors` 要求新 namespace 和 driver 写出的 job ID 相符，跨 namespace 的活 actor PID 不能被信号清理。
- 借卡期间 SIGTERM 会延迟到精确 stop/receipt 事务结束，随后立即走 finally；SIGKILL 或机器故障仍需依据独占回执恢复。
- `Catalog` 用启动身份、父子链、唯一 token/phase 及 actor 身份登记；信号经 pidfd。禁止进程名匹配和共享 Ray restart。
- `C.finalize_stopped()` 只补全已有 `clean-old-stopped.json` 的 bookkeeping。半途 stop 无完成证据就明确报错，不能冒充已借还。
- 服务 CPU-ready 前不准建立 GPU 上下文；运行中发现本 owner 的 C/G PID 出现在 4 以外即终止本次 trial 并精确清理。
- 当前 cleanup 的共享 Ray named-actor API 沿用既有 helper；若共享 Ray 本身不可响应，归还不能凭空确认，需现场核验。
