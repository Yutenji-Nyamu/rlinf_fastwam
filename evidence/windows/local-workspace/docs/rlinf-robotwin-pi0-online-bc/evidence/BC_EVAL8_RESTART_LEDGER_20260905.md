# 原BC OOM后：eval8×4独立接续账本

09-05 19:15用户确认原BC已断，并提出评估8×4再放；随后要求GPU6与GPU7并行、不必相互等待。范围：原BC独立树仅评估并发/分批配置、对应测试与证据；GPU6短测通过后从同SFT/空池重开100轮。训练32×1/micro32/global1024/U10/M4等不变；不动GPU7已在跑的DVAC、Sidney/shared Ray/他人。

1. 19:17普通账号只读身份、Git、源hash、GPU/RAM/磁盘探针：原BC HEAD385d4e75 clean，三个本地文件SHA256与部署逐一一致；GPU6仅11MiB无compute，GPU7正在DVAC第二轮；RAM available约1.2TiB，memory PSI0，足够并行。原始`BC_EVAL8_PREFLIGHT_20260905.txt`。
2. 原正式BC在17:44:40第6轮rollout CUDA OOM，完整5轮、无save10 checkpoint；新formal不是续第5轮，也不拿smoke权重/成功池续训。
3. `/data`现余453160865792 bytes≈422.04GiB；`/home`余1338044260352 bytes≈1246.15GiB。/data可容本次smoke，但扣除π0.5后续九代约241.6GiB、DVAC剩余一代约17.3GiB和新BC smoke两代约34.5GiB，不足新BC正式全部十代约172.8GiB。因此已非阻塞询问新BC正式是否写本人/home；不删除/移动旧产物。formal启动需获得该存储选择；CPU配置和GPU6 smoke可继续。
4. 实施计划：基线YAML仅eval.total_num_envs16→8、rollout_epoch2→4、fixed_reset_batch_count2→4；固定32种子已由原通用循环支持，无需新改环境Python。更新原配置断言，并让既有种子回归同时检查16×2和8×4完整有序覆盖32个相同ID。随后服务器11tests＋真实配置组合/逐叶对照；完整GPU6合同展示后并行launch。
5. 用户随后明确选择“先完成smoke，暂不放正式”，覆盖本文件第1段原formal计划；不写/home、不清理。后续仅GPU6两轮smoke和GPU7既有smoke。
6. 两个精确文件已部署；11/11测试9.90s通过，真实validate_cfg与live Ray物理GPU6映射通过。原种子表实读比对：8×4与旧16×2拼接32ID完全同序且无重复，循环返回首批。其余resolved只有必要输出路径/名称差异。source `a8764944`已push原BC独立分支，clean；GPU7树不受影响。证据`BC_EVAL8_TEST_CONFIG_20260905.txt`，完整合同`GPU6_EVAL8_SMOKE_CONTRACT_20260905.md`。
7. 19:27:23独立GPU6 smoke启动，wrapper/PGID1597471、observer1597472；输出`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8`，源锁a8764944。启动前再次验证GPU6无compute、源clean/两个本地文件及resolved与服务器一致、目标run不存在。只启动smoke，不存在自动formal接续。
8. wrapper实际时间19:27:26—20:05:53，38分27秒，exit0；两轮完整采集/20次Adam/两次32条固定评估/两代保存均完成。read-only watch自然结束，GPU6释放至11MiB。没有自动formal接续。
9. 20:14:17在服务器CPU只读验收：训练26/32、23/32；fixed24/32、25/32；FM loss0.023685→0.016830，梯度有限。累计成功26→49 episode、78→147 query；两代learner.update_step10/20，原BC replay均不含DVAC权重。两代各有10390434302B native shard、8065002471B full weights及非空replay/learner，sidecar实际torch.load读取。验收`BC_EVAL8_SMOKE_VERIFICATION_20260905.json` passed。没有生产worker全模型/优化器重启恢复测试。
10. 每5秒采样GPU峰71188MiB＝69.52GiB，原v7 eval16×2峰77.46GiB；训练32/micro32/global1024/U10不变。Env FD最高882，RSS最高62.50GiB，主机RAM available最低1203.40GiB，memory PSI0；无所查fatal/OOM/相机分配错误。两轮显存余量明显增加，但不能据此保证第6轮及100轮长程容量。
11. 20:15仅归档6个轻量配置/测试/验收文件到原BC树`docs/evidence/online-bc-eval8-20260905/`；`git diff --cached --check`通过，提交`2467d997831166b70444b0c99d5198a2d3dfc8f6`已push个人`codex/sz-pi0-online-bc`，HEAD/clean已核验。没有大模型、replay图像或他人存储表进入Git。用户最新约束仍为只完成smoke、暂不formal，不切/home、不删旧产物。
