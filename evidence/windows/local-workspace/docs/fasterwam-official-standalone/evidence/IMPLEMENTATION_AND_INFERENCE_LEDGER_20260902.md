# Faster-WAM official standalone 实施账本

日期：2026-09-02

## FW-0001 官方材料与授权边界

- 用户授权：阅读官方材料，处理网络、磁盘与本机适配，并按官方流程跑通 Faster-WAM。
- 来源锁：`https://github.com/hustvl/FasterWAM` / arXiv `2608.04404`；明确排除同名 arXiv `2608.02365`。
- 官方发布合同：Python 3.10、PyTorch 2.7.1/CUDA 12.8；RoboTwin checkpoint
  `robotwin/step_029355.pt` 与对应 `dataset_stats.json`；Wan2.2-TI2V-5B 组件由
  `DIFFSYNTH_MODEL_BASE_PATH` 指定。
- 官方评估默认：`demo_randomized`、`unseen`、100 episodes、replan28、M10；本轮只把调度预算缩到
  1 GPU / 1 task / 1 episode，不改变模型或控制合同。
- 明确不做：训练数据下载、SparseActionDiT 训练初始化、RLinf 集成、修改或停止现有任务。

## FW-0002 本地上下文读取

- 已完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 及 WAM official survey 当前事实源。
- 复用原则：源码/模型/结果放 `/data/chenyiteng`，环境放 `/home/chenyiteng`；动态资源以本轮现场为准。

## FW-0003 SZ 只读 preflight

- 时间：2026-09-02 12:49 CST；固定 host-key、进程内密码的 `chenyiteng` 通道，exit 0。
- GPU0--3空闲；GPU4--7为既有 π0.5/Fast-WAM RLinf 训练，本轮不触碰。
- RAM available约931 GiB；`/data`、`/home`、`/`分别余1.7 TiB、1.4 TiB、225 GiB。
- Faster-WAM目标源码/模型/结果目录均尚不存在。
- 可复用但不混写的公共资产：旧 Fast-WAM 独立环境为 Python3.10/Torch2.7.1+cu128；
  `/data/chenyiteng/models/fastwam`已有UMT5、Wan2.2 VAE/tokenizer等公共组件。
- 网络：Mihomo active；HF直连SSL reset，显式Mihomo代理和hf-mirror可达。官方HF优先走临时显式代理，
  不持久化proxy；失败时才切mirror。

## FW-0004 官方源码落盘与执行合同

- 源码目录：`/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official`。
- exact HEAD：`83667817df0d4f823f39d90700e61ea2f432ac45`；`main`、clean；相对最新父提交只增 latency 工具，
  RoboTwin 部署脚本未变。
- 官方 RoboTwin 安装会用 `environments/robotwin/uv.lock` 创建独立
  `.venvs/robotwin`，安装 CuRobo v0.7.8，并链接 vendored RoboTwin policy adapter。
- 官方 manager 固定先跑 clean 再跑 randomized；为了只跑1个 official episode，本轮使用官方
  `experiments/robotwin/eval_robotwin_single.py`，不改模型、环境或控制逻辑。
- 首个 oracle 锁定 `move_stapler_pad / demo_randomized / unseen / seed0 / replan28 / M10`，
  保留官方 `skip_get_obs_within_replan=true`。

## FW-0005 安装与下载实施设计

- 官方 release 锁定 HF revision `6bf9471ced6919a15ab8fded89f7772f5060c44b`；只下载
  `robotwin/step_029355.pt` 与配套 `dataset_stats.json`到
  `/data/chenyiteng/models/fasterwam/release-6bf9471`。
- 下载只在当前进程使用 `127.0.0.1:7890`，不写 shell/profile 级代理。
- 执行官方 `scripts/setup/install_robotwin.sh`，独立环境位于新源码树的
  `.venvs/robotwin`。
- 对三个与现有 RoboTwin 完全同源的公共 assets 使用显式软链接，避免重复下载；
  不复用旧 Fast-WAM checkpoint 或 stats。
- Wan 公共组件按官方 resolver 的精确目录合同复用现有
  `/data/chenyiteng/models/fastwam/diffsynth`；released eval 已设
  `skip_dit_load_from_pretrain=true`，不需要 SparseActionDiT 初始化权重。

## FW-0006 官方权重第一次下载中断

- HF proxy 连接在已接收 `1,504,511,877` bytes 后返回 `IncompleteRead` /
  `ChunkedEncodingError`；这是传输层中断，不是 revision、权重或磁盘问题。
- 保留同一目标中约1.5 GB的 `.incomplete`，不删除、不另建副本；在同一
  official revision 上加入最多6次的有限自动续传，每次间5秒。

## FW-0007 下载路由窄修正

- 官方 HF/Xet 经服务器 Mihomo 的6次有限续传仍均以 `IncompleteRead` 或
  `SSL EOF` 结束；partial 持续增长，说明是长连接稳定性，不是文件不可达。
- 8 MiB range 实测：official+proxy约1.12 MB/s，hf-mirror直连约0.60 MB/s；但前者长连接
  持续中断，因此切换到更稳定的 mirror 续传同一 partial。
- mirror 路由仍锁定 official HF revision，并从 LFS pointer 取得预期 `size` 与
  `sha256`；只在两者均精确匹配后才将 `.incomplete` 原子改名为正式 checkpoint。

## FW-0008 官方 release 完成

- checkpoint：`11,117,757,817` bytes，SHA-256
  `934684f2b60f78d493d14f30dba4554c0c064803f0a7f659aaf2b16e60d6c7ef`。
- stats：`88,715` bytes，SHA-256
  `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。
- mirror 在同一 partial 上一次自然完成，平均约4.71 MB/s；size 与 official LFS SHA
  均通过后原子改名，没有残留 `.incomplete`。

## FW-0009 uv 网络路由收窄

- 官方 installer 初次在全局临时 proxy 下运行；只读确认它未卡死，但长期仅约0.15 MB/s，
  大量 wheel 连接受 Mihomo SSL EOF 影响。
- 锁文件原始 URL 的4 MiB range 实测：PyPI direct约5.1 MB/s且完整；proxy约6.1 MB/s
  但报SSL EOF；PyTorch direct约0.36 MB/s且完整，proxy在5秒内SSL EOF。
- 因此只精确终止本任务自己的 `uv sync`，保留约4.1 GiB uv cache，然后在同一 official
  installer/lock/env 上取消对 uv 的 proxy 注入并续传。

## FW-0010 官方环境主体完成与 CuRobo header 缺口

- 取消代理后，同一 official lock 的 `.venvs/robotwin` 自然完成：Python3.10、
  Torch2.4.1+cu121、NumPy1.26.4及RoboTwin依赖均已落盘；venv约7.4 GiB，uv cache约7.9 GiB。
- 官方 installer 已执行 RoboTwin 环境 patch，并锁定/克隆 CuRobo v0.7.8，exact commit
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。
- 唯一失败点是 CuRobo extension 编译：official venv解释器为`/usr/bin/python3.10`，其
  sysconfig指向`/usr/include/python3.10`，但主机未安装`python3.10-dev`，因而缺少`Python.h`。
  这发生在C++编译期，不是checkpoint、CUDA runtime、模型或仿真故障。
- 服务器已有匹配的Python3.10开发头：
  `/home/chenyiteng/miniforge3/envs/RoboTwin/include/python3.10/Python.h`。
  采用最窄修复：仅在本次 CuRobo build 进程设置该 include path；不装apt、不改系统Python或全局环境。

## FW-0011 CuRobo与独立官方环境完成

- 只重跑官方 installer 的剩余核心命令：CuRobo v0.7.8 editable build；设置
  `TORCH_CUDA_ARCH_LIST=9.0`、`MAX_JOBS=8`，并只给该进程加入已有Python3.10 header。
- build约6分57秒自然完成；`nvidia-curobo==0.7.8`安装成功。随后按官方脚本建立
  `third_party/RoboTwin/policy/fasterwam_policy`链接；三个公共asset链接均已存在，未重复下载。
- import/CUDA探针：NumPy1.26.4、Torch2.4.1+cu121、SAPIEN3.0.0b1、CuRobo0.7.8；
  `torch.cuda.is_available=True`，设备为GPU3的NVIDIA H100 80GB HBM3。
- 官方环境约7.5 GiB，CuRobo source/build约369 MiB；安装结束GPU3回到9 MiB/0%。
- SAPIEN import出现“未找到Vulkan ICD文件、将尝试自行提供”的warning；尚不是失败，进入真实仿真时验证。

## FW-0012 official single-episode闭环成功

- 入口：官方`experiments/robotwin/eval_robotwin_single.py`，没有修改源码；物理GPU3。
- resolved口径：`move_stapler_pad / demo_randomized / unseen / seed=0`
  （RoboTwin实际seed `100000`），1 episode，`replan_steps=28`、`num_inference_steps=10`、
  `one_pass_future_cache`、`skip_get_obs_within_replan=true`、qpos 14D。
- release：`step_029355.pt`及配套stats；模型组件加载61.24秒。日志确认video expert 5.00B、
  action expert 0.56B，checkpoint覆盖随机初始化的release结构。
- 结果：`1/1`、100%，进程自然exit 0；fatal scan为空。总wall约2分20秒。
- 视频：H.264，640x480，145 frames，14.5秒，文件名携带`success-true`；结果txt、完整resolved
  YAML和eval/driver日志均存在。
- 运行后的GPU3为9 MiB/0%；GPU4--7既有训练未触碰。SAPIEN Vulkan ICD和
  `missing pytorch3d`提示在真实仿真中均非致命，不再为无症状warning增加环境改动。
- 服务器结果：
  `/data/chenyiteng/results/fasterwam-standalone/official-move-stapler-pad-random1-20260902-v1`。
  本机轻量证据：`evidence/official-move1-20260902/`，共5个原始产物加manifest，约138 KiB。
