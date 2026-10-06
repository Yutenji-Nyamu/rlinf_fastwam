# DSRL6/7独立lease与RLT候补（审核稿）

本目录没有执行入口的隐式副作用。`prepare_lease.py`只准备文件/专用DSO和profile；`lease_owner.py owner`是唯一派发者。GPU6 Clean、GPU7 U；GPU4/5及其原协调器不发信号、不改实配。

## 状态与退出

每卡 `DEV_HOLD → SMOKE → DEV_HOLD` 可以覆盖模型probe、fresh smoke、恢复smoke和microbatch测量；probe/smoke不自动归还。只有 `FORMAL` driver退出、Ray namespace与登记进程树消失、全部C/G释放且GPU健康，才唯一恢复该卡原RLT。正式成功和正式失败均会归还；资源未释放、身份/源码不一致则NEEDS_ATTENTION，保留证据、不reset、不广泛kill、不自动重试。

恢复使用 `stop-attempt.json.capture` 的完整CP；新run、新namespace，原RLT N8/200动作/3000累计轮次及其余算法实配不变。RLT首轮真实回放/更新指标验收后写 `gpuN-rlt-restored.json`，两卡归还写 `lease-returned.json`；owner继续观察RLT至其自然结束并写finished，不启动第二轮恢复。

## CPU检查与准备

在服务器的既有训练Python运行：

```text
python -B -m unittest discover -s tools/dsrl_u_20261006/ops -p test_lease_cpu.py -v
```

测试只检查probe环境隔离、恢复路径替换保持预算、渲染卡位拦截，不初始化CUDA/Ray。源码审核通过后，先用另一个受审stop_handoff事务保存与停止6/7，等待唯一释放回执，再准备输入JSON：

```json
{
  "control_dir": "/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1",
  "stop_attempt": "/data/chenyiteng/deployment-20261006/dsrl-rlt67-handoff-v1/stop-attempt.json",
  "release_receipt": "/data/chenyiteng/deployment-20261006/dsrl-rlt67-handoff-v1/rlt67-released.json",
  "dsrl_repo": "<审过的新工作树>",
  "dsrl_python": "<已验证的训练Python>"
}
```

```text
python -B tools/dsrl_u_20261006/ops/prepare_lease.py --inputs <lease-inputs.json>
python -B tools/dsrl_u_20261006/ops/lease_owner.py --control <control_dir> owner
```

owner应由主代理以已有后台启动工作流启动并记录其PID/start；本工具本身不调用shell、不守护共享Ray。每个control只允许一个owner；另外持有6/7全局任务lease文件锁。prepare不可重放，会拒绝现存control、新RLT run或scope profile。

## 配置与任务请求

factory继承RLT已验证的Ray/venv/assets依赖，移除RLT日志/统计变量及旧native overlay、旧scope bootstrap；将REPO_PATH、EMBODIED_PATH、RLINF_CODE_WORKING_DIR、PYTHONPATH换新源码，模型与norm环境路径换Sidney π0.5。每卡独立DSO/profile/manifest；在control中生成 `gpu6-scope-env.json` / `gpu7-scope-env.json`。

**正式配置生成器必须把该scope_env注入cluster.node_groups对应actor/env/rollout进程。** 只给CPU driver环境不足以保证Ray工作进程继承。新profile只匹配新DSO；runtime保持已有SAPIEN3.0.1 Scene绑定并用独立comm名称，不改其他卡的profile。

由配置生成器先保存该run的 `runtime/resolved.yaml`；每个请求使用全新runtime、request_id和namespace。例如：

```text
python -B tools/dsrl_u_20261006/ops/make_request.py --control <control_dir> --request-id clean-smoke-v1 --gpu 6 --kind smoke --runtime <run>/runtime --namespace dsrl-u-20261006-clean-smoke-v1 --output <request.json>
python -B tools/dsrl_u_20261006/ops/lease_owner.py --control <control_dir> submit --request <request.json>
```

`make_request`固定所有rlinf Python源码、entry、resolved YAML与repo HEAD；submit/owner启动前重核。`--kind formal`才开启完成/异常后的RLT归还。resume-smoke仍使用kind=smoke，恢复来源由root配置生成器提供。

轻量model probe使用kind=probe、repo内entry、独立runtime和必要的 `--probe-arg=...`；不指定namespace。每次子进程单独构造CUDA mask：probe仅本卡，正式driver不设全局mask，让RLinf物理placement负责；probe不会污染owner或后续正式环境。

## 边界

- 本分工只编写本地脚本，尚未运行服务器CPU测试、prepare、owner或任何GPU任务。
- 脚本依赖已有Ray Dashboard、trusted ops/common、SAPIEN3.0.1及cc；缺失即停止。
- CPU owner异常或被停会保留已运行子任务，没有自动猜测资源归还；需读取status/identity并处理。正式driver挂死但未退出也不会仅凭0%利用率归还。
- 图形越卡时仅精确TERM本任务wrapper并停在NEEDS_ATTENTION，由其finally做本job清理；不会主动终止4/5或共享Ray。
- 信号/算法、模型配置与正式预算由root维护，本目录仅负责运行隔离与交接。
