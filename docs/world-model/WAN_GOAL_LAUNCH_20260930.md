# SZ3 Wan Goal 首次启动计划

2026-09-30建立，2026-10-01按用户最新优先级更新。WM第一、Dojo第二；RLT暂不安排。下列为本次唯一启动路由，实际状态以服务器回执为准。

|阶段|配置|预算|通过条件|
|---|---|---|---|
|OFT smoke|wan_goal_oft_smoke_sz3|4卡，N32/G8，256动作，rollout1，global1024/micro32，2epoch|两轮真实有效GRPO梯度、有效样本及checkpoint权重变化|
|π05 smoke|wan_goal_pi05_headonly_smoke_sz3|4卡，N32/G8，320动作，rollout1，global1280/micro64，2epoch|同上，使用批准的腕图mask、H10/C8/M5|
|π05正式|wan_goal_pi05_headonly_formal_sz3|4卡，N64/G8，320动作，rollout8，global2048/micro128，1000epoch，每40轮保存|从固定SFT重新开始，确认真实训练指标推进后交由服务器owner持续运行|

只借物理GPU4–7。训练在独立Ray实例，端口63843/63844/63845；外层63842仅作序列归属记录。GPU0–3、共享Ray和其他实验保持原状。真实LIBERO自动评测仍关闭，WM内部奖励不当作真环境成功率。

根目录`/data/chenyiteng/projects/wan-goal-sz3`，输出`runs/wan-goal-sz3-20260930-r1/{oft-smoke,pi05-smoke,pi05-formal}`，各阶段control目录保存释放、退出及smoke验收回执。TensorBoard、checkpoint和视频沿各阶段resolved config保存。

控制器命令由冻结配置展开：`<RLT_PY> -u -B <Dojo项目>/scripts/wm-bridge-20260930-v1/continue_pipeline.py --config <当前Dojo配置> --cycle-dir <Dojo项目>/rlt-cycle-sz3-wan-goal-20260930-v1 --preparation-dir <原Dojo run>/prepare-continuation-20260930-wan-goal-v1`。控制器运行已审`wm_sequence.py`；单阶段调用`private_ray_driver.py --repo <对应clone> --config <上表配置名> --log-dir <阶段输出> --port <独立端口>`。

停止条件：smoke达到2epoch即验收，未通过不进入下一阶段；正式达到1000epoch停止，不额外设墙钟预算。WM正常完成或失败后精确清理本批并续原Dojo；原Dojo最终退出后RLT仍保持暂停，待用户另行安排。若无法核实本批GPU释放，保留现场并报告，不抢占或重复恢复。当前准备验收中的pip元数据警告见`WAN_GOAL_ENVIRONMENT_NOTES.md`，真实GPU路径仍由smoke裁决。

切换顺序：旧Dojo退出触发的RLT恢复已发生；完成旧owner退出和归属交接即可，不再把RLT首轮作为WM前置条件。新cycle仅精确暂停这四条RLT→WM序列→保留原6300评估结果继续Dojo；最终仍保留RLT恢复清单，但不自动派发。启动使用`--skip-prior-first-round --defer-rlt-restore`，方法与训练预算不变。
