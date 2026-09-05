# π0.5在线BC smoke证据包

2026-09-05 22:41:20 CST正常结束（exit0），22:42服务器读回验收通过；运行39分52秒。两轮训练成功8/32→18/32，更新后fixed11/32→18/32，累计26个成功episode／89个query；TensorBoard原始step0/1对应已完成轮1/2，不是更新前Step0评估。20次Adam，FM loss0.008772→0.007208；显存采样峰73.74GiB、Env FD峰882，主机available最低1206.59GiB。两代checkpoint共37.89GiB；训练结束GPU6降至11MiB。

- 训练源码：`653fe0fbc05188eb0ec19077de5c78a00b8230ad`，独立`codex/sz-pi05-online-bc`；由π0 BC `2467d997`迁移Sidney adapter、模型/任务配置与固定32初态表。
- 预算与命令：[合同](GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md)、[完整resolved](PI05_BC_SMOKE_RESOLVED_20260905.yaml)、[对原BC逐叶差异](PI05_BC_RESOLVED_DELTA_20260905.json)。本包只对应两轮GPU6 smoke，不是正式训练。
- CPU检查：`PI05_BC_TEST_CONFIG_FINAL_20260905.txt`记录13项pytest通过，随后旧验证脚本因预期FP64、原生prepare输出FP32而断言；**不是该整条命令全部通过，也不是模型运行故障**。只修验证预期dtype，未改生产精度，单独重验结果见`PI05_BC_DATA_VALIDATION_RETEST_20260905.txt`及`PI05_BC_VALIDATION_20260905.json`。
- 完整smoke结论以`PI05_BC_SMOKE_VERIFICATION_20260905.json`为准：实际采集/更新/同步/评估/保存；读取learner、replay RNG与少量模型tensor，不声称完整worker/optimizer恢复或长程稳定。
- 同包保留`pi05_bc_wrapper_20260905.sh`、原GPU6 observer、真实数据检查脚本和结束验收脚本，便于复核。checkpoint与大日志仅留服务器，不加入Git；无密码或SSH身份材料。
- 短测成功率仅检查链路，不据两轮判定学习收益；未自动转正式，未修改Sidney GRPO、共享Ray或其他用户任务。
