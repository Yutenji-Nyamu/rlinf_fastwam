# GPU6 π0.5 BC formal100执行账本

## 授权与边界

用户明确授权GPU6正式训练，沿已通过π0.5 smoke的学习预算、原正式eval5/save10；启动确认后报告，不持续盯守。原Sidney SFT、空成功池，100轮、32×1、micro32/global1024/U10。不得续smoke、不动GPU4/5 Sidney/shared Ray/其他用户、不清理或升级依赖、不实施checkpoint轮换。

## 逐操作记录

1. 完整读取根规则/交接、窗口交接与唯一π0.5 BC专题；复核smoke合同、resolved和原π0正式wrapper。确认原正式硬上限48h；smoke 90min不能照搬。
2. 创建本账本与独立正式启动packet脚本；生产源码不改。下一步先普通账号固定host-key只读身份/资源/HEAD刷新，再服务器Hydra compose逐叶核验。完整resolved、合同展示后才启动。
3. 23:36:03—11 CST，`pi05_bc_formal_execute_20260905.py prepare`：身份uid1003；HEAD912bc690/clean、GPU6无compute，新路径不存在，原smoke exit0、原模型大小一致。Hydra compose/validate通过；11个叶变化恰为4个正式调度/总量＋7个命名路径，其他全相同。只连接现有Ray做配置校验并断开本client，未重启集群/创建训练worker。GPU6=11MiB、RAM available1317.4GiB、/data余875.07GiB。完整证据`PI05_BC_FORMAL_PREFLIGHT_20260905.json`和resolved已落盘。
4. 正式合同已建立，精确命令、输出、3200采集/最多1000Adam/640评估/10代checkpoint及48h原正式时限明确；按用户授权准备单次启动，不再重复smoke。
5. 23:38:34 CST单次启动成功：wrapper/PGID2143105，observer2143106。独立新run `pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1`；没有续smoke/修改原模型。远端wrapper `bash -n`通过；启动前再次确认源码clean/GPU6无compute/目标目录不存在。
6. 23:41:58只读启动验收：wrapper及Env2143761/Rollout2143755/Actor2143747均在、uid1003；已进入首轮`Generating Rollout Epochs 0/1`。expert_only=True、可训练693,422,112参数，actor/rollout加载原Sidney norm；无所查fatal/OOM/Traceback等。GPU6现场25.27GiB（仍初始化/采集，非容量峰值）、RAM available1281.4GiB、/data余875.06GiB；无exit/finished。初始Vulkan ICD/FSDP精度建议warning仍与原环境相同，不为此擅改依赖/精度。完整Step1和正式fixed/checkpoint尚未产生，不称效果已提升。
7. 23:44:38启动参数复核：真实driver2143110的cmdline包含M配置组及全部正式覆盖；三worker的CUDA_VISIBLE_DEVICES均为6，REPO_PATH/RLINF_CODE_WORKING_DIR/模型/新run路径逐一一致，均无LD_PRELOAD/Fast shim；Env FD639。shared Ray原321933/322685及Sidney原wrapper602620仍在。没有在driver日志找到可直接解析的完整JSON config（actual_config_matches=null）；不冒充完成独立日志配置逐叶对照。完整配置依据是实际源码同HEAD、实际CLI/env一致与服务器Hydra resolved逐叶校验。证据`PI05_BC_FORMAL_BINDING_20260905.json`。
8. 参数来源已逐项整理至唯一计划§10；准备仅将本次轻量合同/resolved/启动证据/参数来源提交并push既有个人分支。生产源码不改，不等待首轮完成，不新建自动化。
