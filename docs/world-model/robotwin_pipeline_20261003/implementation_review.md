# 首smoke接口独立复核与服务器验证入口

2026-10-03。范围：本地新adapter/env/service与固定Sidney源码的静态review；不作为GPU运行验收。

## 明确结论

1. **C32没有把策略模型改成H32。** 固定2151的`OpenPi0ForRLActionPrediction.sample_actions()`仍保H50 chain，采样`prev_logprobs`截取`action_chunk × action_env_dim`；`default_forward()`重算执行同样截取。新配置actor/rollout均H50、C32、14D，不把OpenDW action expert当策略输出。原生OpenPI输出反归一化后只把前32动作交env。
2. **新的环境注册可走旧动作接口。** `SupportedEnvType.OPENDW_ROBOTWIN`注册后，固定`prepare_actions()`的末尾分支透传raw action；不会套LIBERO的夹爪变换。adapter只裁剪两夹爪0–1，不重复归一化。
3. **返回一个末观察可用。** 旧EnvWorker取`obs_list[-1]`和`infos_list[-1]`，不要求长度32。reward、termination和truncation仍是[B,32]。本smoke不使用critic/外部奖励worker，所以没有value bootstrap需要隐藏的final observation。
4. **初始差分是官方口径。** WorldArena `_reset_metrics()`把prev设0，reset不先评分。新env相同。8未来帧连续概率差分放于动作4/8/…/32，累计=末图概率；并非score32-score(reset)。RM已经sigmoid，服务不能二次sigmoid或round。
5. **成功和回报不是一个值。** 任一预测帧score≥0.9会在chunk末termination；分数随后回落时，累计回报仍是末图score。它继承WorldArena语义，first_success_action单独记录。到smoke32动作同时可有truncation，done为两者并集。0.9未达到也可因连续score差异产生GRPO优势。
6. **RM源码对齐。** 新`opendw_reward.py`与固定WorldArena ResNet18/T5-base/cross-attention、LayerNorm、MLP、sigmoid逐结构一致；严格加载完整state dict；图像内部uint8→[0,1]、双线性224与ImageNet normalize。输入是生成画布的head区域，不是整张含腕拼图。创建ResNet不下载ImageNet权重，因为马上完整严格加载任务权重。
7. **GPU卸载是服务端事实。** env`offload()`同步等待HTTP；runner再等EnvWorker结束。服务锁串行化模型使用，明确更新OpenDW缓存device、移动参数/buffer，清VAE运行缓存后同步/empty_cache，再答复成功。异常由独立owner接收并精准收尾，不能只查Ray的显存。

## 已发现并修正

- 多worker `service_urls`索引此前误将`seed_offset`除以`num_envs`。旧EnvWorker传的是rank/stage索引，不是全局环境槽偏移，会让多个worker选同一URL。adapter作者已改为`seed_offset % len(urls)`并补定点测试；当前GPU4单worker不受旧bug影响。
- runner补丁的旧目录名guard会误拒实际`.../opendw-robotwin-smoke-20261003/rlinf`。已改为精确绝对checkout＋本次独立分支核验，不放宽到任意名含opendw的目录。
- 文档旧句“G8全失败没有学习信号”只适用于二元结果。当前连续RM以回报差异/过滤和mask判断，已纠正。

## reset JPEG颜色

本地RoboTwin相机源`get_picture("Color")[:,:,:3]`直接给RGB，原生vector_env与Sidney观察链不交换通道。已有旧数据转换链连续使用`cv2.imdecode`/`cv2.imencode`无额外交换，保持原数值。

官方当前文档明确存在两种HDF5 JPEG：legacy是RGB数组直接交`cv2.imencode`，应直接`cv2.imdecode`以还原RGB数值；新standard格式含JPEG COM `XPL-RGB1`，由官方`decode_image_bit`识别并返回RGB。实际reset必须按资产类型解码；不能无条件追加BGR2RGB，也不能将旧bytes直接PIL RGB解码。应在轻量receipt记所用HDF5、frame index、JPEG marker和解码分支。[RoboTwin官方数据说明](https://github.com/RoboTwin-Platform/RoboTwin#-getting-data)

## 最小服务器验证

1. 把本次新增adapter/env/service与register patch复制到新独立checkout；核源码SHA、固定2151基点、OpenDW版本与资产。用`prepare_runner_barrier.py`生成/应用精确wait patch；用`build_smoke_config.py`生成本次新名配置与diff。
2. 在服务器RLinf环境、`PYTHONPATH=<本次checkout>`下运行`test_opendw_adapter.py`和`test_opendw_env.py`；两者CPU/mock，无需模型GPU。服务另有HTTP/mock协议检查，由服务作者维护。它们只验证明确接口，不能代表WM图像质量。
3. 服务器Hydra `train_embodied_agent.py --config-path <本次配置目录> --config-name <本次名> --cfg job --resolve`检查真实配置展开。真正`validate_cfg`与placement需结合共享Ray中本次独立namespace及明确物理卡声明；不另起Ray服务、不使用自动资源分配，核N/G/微批整除和只卡4。
4. 显式GPU4 service容量试配，首次完成一条真实外部动作预测、8帧score、同步offload时保存回执。后接N8完整GRPO smoke，所有过程只以唯一owner命令为准。
5. 验收读取本次退出码、所有角色结束、服务峰值/卸载后残余、32动作末状态、score/mask/优势/梯度、CP文件。先报`pipeline_completed`，有真实非零优势与有效梯度才报`learning_signal_verified`。

配置生成器和runner补丁生成器已本地静态AST检查；本文没有执行服务器测试，也不声称GPU smoke已通过。
