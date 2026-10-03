# N64/G8/R8/C32 四卡 overlay

此目录是独立副本，不修改此前单卡已跑通源码。actor/rollout placement 由配置负责：4、5；env placement 6、7；两个HTTP WM服务各只见一张卡，内部仍B1串行。两卡服务可以并行处理各32个环境。

## rank 来源（已查代码，不使用猜测或取模）

RLinf `2151a08ee1bd75df1bef0d8190e594bd5c7f7977` 的 `rlinf/workers/env/env_worker.py:139–143`：`num_envs = total_num_envs // world_size // stage_num`；`:417–430` 的 `_setup_env_and_wrappers` 构造env时传 `seed_offset = self._rank * self.stage_num + stage_id`，`total_num_processes = self._world_size * self.stage_num`。

因此 pipeline_stage_num=1、env world=2、N64：

|env rank/seed_offset|本地env数|全局env索引|端口|WM物理卡|
|---:|---:|---|---:|---:|
|0|32|0–31|18946|6|
|1|32|32–63|18947|7|

配置 `service_urls: ['http://127.0.0.1:18946', 'http://127.0.0.1:18947']`。新env严格要求URLs数等于process数、规范化后的host/port不重复，直接用 `urls[seed_offset]`；修改stage数但仍只有两服务将拒绝启动。旧 `service_url` 单地址仍兼容。每rank保留原来的 `seed=base_seed+seed_offset` 和各环境独立RNG、image/state/score历史；`use_fixed_reset_state_ids` 完全由配置继承，不修改数据采样。

服务日志新增 `env_process_index`、`env_process_count`、`global_env_index`，原 `env_index/reset_id/request_id` 保留。每次R结束仍由原EnvWorker的finish_rollout调用 `update_reset_state_ids()`，R8不需要服务器跨请求隐藏状态。

## GPU映射与卸载

服务原本已接受明确的 physical GPU4–7与CVD单卡，此副本继续保留；CPU初始化仅用nvidia-smi记录物理index/UUID/PCI，不调用CUDA。第一次及后续 `/onload` 在owner完成借卡后，以CUDA Driver API验证唯一可见逻辑0的UUID/PCI等于该物理卡；错误即拒绝加载模型。每个服务有自己的HTTP锁、模型、输出目录和资源监控；同步 `/offload` 保持原逻辑，回执后才确认卸载完成。

服务6示例：`CUDA_VISIBLE_DEVICES=6 <OpenDW Python> -B <overlay>/tools/opendw_service.py --physical-gpu 6 --port 18946 ...`；服务7相同，CVD/physical-gpu改7、port改18947，输出目录必须独立。模型、RM、10步推理、C32、三图、归一化及reward规则均继承旧服务。

部署时将此目录的 `rlinf/envs/world_model/{opendw_robotwin_env,opendw_adapter}.py` 放入新隔离checkout；registry继续使用既有 `opendw_robotwin` 注册。服务运行此目录tools下版本，配套reward/telemetry副本未改变。原checkout、旧服务脚本保持不动。

CPU检查：在服务器原RLinf Python及新隔离checkout PYTHONPATH下运行 `python -B <overlay>/test_multigpu_routes.py`。4项仅核rank分配、拒绝重复/错process数、状态与RNG独立、NVML文本身份与日志字段；不发HTTP、不用GPU、不启动Ray。GPU真实映射留给owner控制下的服务onload和全卡C/G监督。
