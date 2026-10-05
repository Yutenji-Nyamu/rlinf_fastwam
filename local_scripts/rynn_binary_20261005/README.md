# Rynn 真实二值判定短测

只测一个固定原生评估批次32回合，N16×2、C32/max384、原CP70。采集不是第二套评估器：给私有 `RoboTwinEnv` 加只写文件的旁路，原策略、动作、seed、奖励和done保持。每环境保存一次：首次原生success的K8历史，或无成功执行到384动作的K8历史。标签来自simulator `infos['success']`，不用终止标志或拼图叠字；提前异常退出不记失败。

`install_capture.py --private-env-file PRIVATE_REPO/rlinf/envs/robotwin/robotwin_env.py` 在私有copy安装，原源另存SHA命名备份。原生评估进程和Ray runtime env必须显式传 `RYNN_BINARY_CAPTURE_DIR=/.../capture`；只有这一个独立eval使用。初始化/reset及每个C32完成时记录主图，均匀K8、320×256、无R/B交换，动作时钟和原图尺寸同时留痕。

已核CP保存源码导出 `CP70/actor/model_state_dict/full_weights.pt`。服务器确认该文件存在后，用 `prepare_native_eval.py --formal-config FORMAL_JSON --checkpoint-file FULL_WEIGHTS --output-dir FRESH_EVAL_DIR` 派生配置，再以既有graphic-scope环境启动 `run_native_eval.py --config FRESH_EVAL_DIR/native_eval.json --private-repo PRIVATE_REPO --environment-fragment REVIEWED_JSON --capture-dir CAPTURE_DIR --namespace UNIQUE_NAMESPACE`。graphic-scope片段的PYTHONPATH须先将旧RLinf替换为PRIVATE_REPO；继承HOME、LD_PRELOAD和scope-manifest，driver保持无CUDA mask、worker按placement绑定GPU4。

`run_native_eval.py` 沿官方 `EmbodiedEvalRunner`，仅创建rollout/env，明确将同一环境片段传入Ray runtime env。末尾driver调用 `ray.shutdown()`；worker清理由根既有owner凭精确namespace/job及catalog负责，不调用不存在的WorkerGroup.stop，也不宽泛停Ray。源配置中的训练env被删除，32条结束不会进入训练。CPU检查 `python -B -m unittest test_binary_probe -v` 在服务器运行。

接入原formal owner的precheck时，准备器传 `--config-dir PREPARED_NATIVE --output-dir OWNER/KEY`，只提前创建prepared、不提前创建owner。运行器传 `--receipt-dir OWNER/KEY`，`ray-job.json`与原catalog读取位置一致。原owner的 `OPENDW_SMOKE_OWNER_TOKEN` / `OPENDW_SMOKE_OWNER_PHASE` 必须成对继承，运行器显式复制到Ray runtime env；不生成另一套token或常驻管理层。

采集完成后，RLinf Python执行：

```bash
python merge_native_dataset.py --capture-dir /.../capture --output-dir /.../dataset --expected-count 32
```

随后确认评估GPU进程已由原owner正常释放，在既有Rynn Python运行（以下S为`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`）：

```bash
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID /data/chenyiteng/venvs/rynnvalue-8b-py310/bin/python -u -B rynn_binary_probe.py \
  --physical-gpu 4 --batch-size 16 \
  --numeric-module "$S/rynn-numeric-v1/code/rynn_numeric_probe.py" \
  --service-module "$S/rynn-control-v2/code/rynn_success_service.py" \
  --official-inference /data/chenyiteng/projects/RynnValue-10e0d333/rynn_infer/inference.py \
  --model-path /data/chenyiteng/models/RynnValue-8B-8738c5e4 \
  --manifest-path /data/chenyiteng/models/RynnValue-8B-8738c5e4/manifest.json \
  --cases-json /.../dataset/cases.json --samples-npz /.../dataset/samples.npz \
  --output /.../run/result.json
```

脚本复用已验B16等token长度分桶与官方数值forward；同一32段既读语言Success又读数值末槽。语言缺失/截断重试一次256token，仍未知单独记账。报告TP/FN/FP/TN/unknown、成功/失败数值范围及AUC。只检查一个阈值能否将本批严格分开，不拟合上线阈值、不做差分奖励，也不把原生可分直接解释为OpenDW生成视频可靠。

本目录没有GPU借还代码。根任务沿既有owner处理GPU4借还；RLT低优先，不能在Rynn与后续WM切换之间重新抢占。此处不操作共享Ray、不自建常驻协调层。

## 后续 click_bell 原SFT与小RM复核复用

配置派生接受已准备好的click_bell正式配置，`--original-sft`不加载CP70。仍为原生N16×2/C32/max384，仅一批32回合。运行器加 `--capture-mode reward_native`，会将每回合全长C32观测及对应native success向量保存在同一NPZ/JSON：

- `native_frames`：原生main_images的原尺寸uint8，不重编码、无通道置换。与JSON `action_steps`、`native_success`逐项对应，适合送小RM。
- `native_success`：每块后明确的simulator布尔标签。reset尚无success查询，标None，不能擅自按0计入混淆矩阵。
- `first_success_position` / `reference_success`：首次成功位置和全回合成功；`native_success_at_end`另留末态信号。
- `frames`：为了保持原二值接口，仍留到首次成功或失败终点的K8。每回合只产生一个记录，不额外增加Rynn推理数。

奖励复核可比较全部已标记C32画面的分类器阈值结果，另列首次成功与成功之后的画面。原生任务的success字段可能已锁存（按铃是首次接触后保持成功），应按任务语义解释后续True，不能声称每帧都重新发生一次接触。暂不额外生成专家样本或做动作/奖励改动。

`prepare_bell_reward_samples.py --capture-dir CAPTURE --output-dir FRESH_SAMPLES` 读取固定32回合，按episode字典序固定选最多64图：16个 `first_success`、32个 `near_false`、16个 `post_success_latched`。near_false仅表示首次成功前的最后一个假标签（或失败回合末图），不声称几何上贴近铃铛。首次成功只能定位到C32边界，物理接触可能在这32动作内部发生。

输出samples.npz直接给既有 `bell/probe_reward.py`。post_success_latched在native语义是成功，但当前是否仍在接触没有标签，因此给probe的label=-1，单独保留native真值。最后 `summarize_bell_reward_probe.py --samples-json SAMPLES/samples.json --probe-result RM_RESULT --output FRESH_SUMMARY` 校验NPZ哈希与逐条ID、分别报告首次成功检出和near_false误报。成功后抬手的低分只作辅助观察，不纳入主假阴性。
