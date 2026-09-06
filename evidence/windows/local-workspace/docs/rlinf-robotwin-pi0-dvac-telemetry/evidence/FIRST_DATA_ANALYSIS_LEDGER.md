# Idea2 DVAC 首轮离线数据分析逐指令流水账

> 日期：2026-08-20（Asia/Shanghai）  
> 原始运行：`idea2_dvac_sft_smoke_2gpu_16env_v1`  
> 范围：只读取既有 16-episode / 64-query telemetry，CPU 离线计算与作图；不改模型、不改 chunk、不启动 simulator 或新评估。

## 1. 论文与本地合同复核

### 1.1 工作区上下文

- 指令：完整读取根目录 `PROJECT_CONTEXT.md`、`HANDOFF.md`，再读取 `HANDOFF.md` 路由的专题文档 `00_INDEX_AND_PLAN.md`；继续完整读取 `01_SIGNAL_AND_DATA_CONTRACT.md`、`02_AUTODL_RUNTIME_GIT_AND_VISUAL_ALIGNMENT.md`、`03_IMPLEMENTATION_RESULT_AND_SMOKE_REVIEW.md`。
- 结果：确认首轮原始产物位于服务器 `outputs/idea2_dvac_sft_smoke_2gpu_16env_v1`；16 episodes、64 queries、每个 episode 4 次 C50 policy query；原始 NPZ 留在服务器，Windows 只接收小型派生表格和图片。
- 决策：本轮将“论文复现”限定为同公式、同语义的诊断图；由于每个 episode 只有 4 个 query 且没有逐控制步 phase 真值，不把稀疏 query 序列写成论文连续时序，也不把 `query_idx`直接等同于 MOVING/OPERATING。

### 1.2 DVAC v1 论文

- 指令：打开并通读官方 arXiv HTML `https://arxiv.org/html/2606.03847v1` 的方法、实验与附录；搜索论文标题、arXiv ID 和作者对应的公开官方代码。
- 结果：
  - Eq. (3)：`z_i = x_i - t_i v_i`；
  - Eq. (4)：对最后 `L` 个 endpoint estimate 使用总体方差（分母 `L`），再对动作维求和得到 `V_s(k)`；
  - Eq. (5)：`V_total = sum_k V_s(k)`；
  - 默认 `alpha=2, m=5, L=5, N_min=1`；附录完整 RoboTwin sweep 实际标注 `N_min=5, N_max=50`；
  - phase 图使用前视与腕视图，由 Seed-2.0-Pro 产生 MOVING/OPERATING 连续区段，再对齐 inference step；不是从方差本身反推 phase；
  - 截至本轮搜索，arXiv v1 页面没有代码链接，搜索未找到可验证的作者官方仓库。因此公式与图的复核以官方论文为唯一一手来源。
- 适配：当前 Pi0 只有 `M=4`，无法照抄 `L=5`；保存的全部 4 个 endpoint 足以离线同时计算 `L=2,3,4`，不需要重跑推理。

### 1.3 用户所附四张图的含义

- 图 1：论文 Figure 1(a)，每次 policy inference 的 `V_total` 随 rollout 推进的标量诊断，并配 phase 帧。
- 图 2：论文 Figure 4，真实 DVAC 在线执行后的 `V_total` 与自适应 `N_exec`；它不是仅靠固定 C50 baseline 就能验证的因果结果。
- 图 3：论文 Figure 7，已有外部 phase 标签后比较 MOVING/OPERATING 的 `log10(V_total)` 分布。
- 图 4：论文 Figure 13，逐 episode 的 inference-step `V_total` 时序并用已有 phase 标签着色。

## 2. 服务器现场与原始数据审计

### 2.1 身份、source lock、资源与进程

- 连接：固定host-key低层Paramiko，密码只进入当前PowerShell进程环境；认证后执行：

```bash
hostname; pwd; id -u; date -Iseconds
cd /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
printf 'HEAD='; git rev-parse HEAD
printf 'STATUS='; git status --porcelain=v1
test -d outputs/idea2_dvac_sft_smoke_2gpu_16env_v1
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo pid,ppid,stat,rss,cmd --sort=-rss | head -n 12
```

- 结果：主机`autodl-container-nekaqbwt43-6ce5babb`、`/root`、UID0；时间
  `2026-08-20T19:49:28+08:00`。source HEAD为
  `61996e15cc7f5a32bd6012b61b20893d94636c82`且tracked clean；原始output存在。两卡均0 MiB/0%，没有
  残留训练/Ray/simulator进程，只有AutoDL基础服务。
- 行为：只读探针；未停止进程、未占GPU。

### 2.2 第一次CSV审计命令的引用问题

- 原指令：直接把带`for f in ...; do ... $f ...; done`的bash片段嵌进PowerShell双引号参数。
- 结果：PowerShell先处理了`$f`，远端收到残缺循环，bash报
  `syntax error: unexpected end of file`。
- 原因：跨PowerShell→Python argparse→bash三层引用，不是服务器数据问题。
- 修复：新建只读命令文件`tmp/idea2_first_analysis_audit.sh`，通过
  `remote_exec_autodl.py run --command-file`原样传输；以后本轮所有多行远端审计均走command file。

### 2.3 原始shard审计

- 实际指令：

```powershell
& <bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_first_analysis_audit.sh
```

- 命令文件执行：`find`列文件与大小、`sha256sum`两份NPZ/四份CSV、`head -n 3`索引；再以服务器
  RLinf venv的Python逐数组打印shape/dtype/finite。
- 结果：
  - `trace_rollout_rank00.npz` 919,602 bytes，SHA256
    `cf4bfd1978c6874efa66d6f2d49136262eae4ee7951cf5efb75a40c9c8f7cc2b`；
  - `trace_rollout_rank01.npz` 920,530 bytes，SHA256
    `c7081042d9087a026ad1a0a5a914d4da58e2fbcaeccf61ec2b728a05f8b8f911`；
  - 每份rank的shape均为`x_chain[32,5,50,14]`、`z_endpoint[32,4,50,14]`、
    `final_model_action/env_action[32,50,14]`、`robot_state[32,14]`、`timesteps[4]`；全部float32且finite；
  - query CSV各32行，episode CSV各8行；字段含真实episode/reset/query/action-slot、三相机路径和MP4映射。

### 2.4 非必要版本探针的引用失败

- 指令：把一长串`python -c 'import numpy,...; print(...)'`再次直接嵌入PowerShell命令参数。
- 结果：本地argparse将内层内容拆成额外参数，报`unrecognized arguments`；没有运行远端Python，也没有
  写入。
- 处理：不再为已能成功import的依赖增加版本探针；正式分析脚本的server `py_compile`和实际imports足以
  覆盖必要前提。

## 3. 分析脚本、运行与两次窄修正

### 3.1 实现

- 新建本地代码副本：[analyze_dvac_first_collection.py](../../../local_scripts/analyze_dvac_first_collection.py)。
- 新增逻辑：
  1. 按rank加载NPZ/CSV，以`trace_row`严格对齐；
  2. 用episode/reset ID many-to-one join outcome并检查query UID唯一；
  3. 计算`L=2/3/4`、`ddof=0`的coordinate variance、`V_L(h)`与`V_total`；
  4. 只写调用方指定、必须原先不存在的独立输出目录；
  5. 生成派生CSV/JSON和论文语义相近、但明确标注query-level边界的PNG。
- 上传目标：`/root/autodl-tmp/idea2_dvac_analysis_scripts/analyze_dvac_first_collection.py`；未改RLinf
  worktree。

### 3.2 v1运行

- 前置命令：

```bash
install -d -m 755 /root/autodl-tmp/idea2_dvac_analysis_scripts
test ! -e /root/autodl-tmp/idea2_dvac_analysis/first_collection_v1_20260820
sha256sum /root/autodl-tmp/idea2_dvac_analysis_scripts/analyze_dvac_first_collection.py
/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile <script>
```

- 正式CPU-only命令：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/idea2_dvac_analysis_scripts/analyze_dvac_first_collection.py \
  --source /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1/dvac_telemetry \
  --output /root/autodl-tmp/idea2_dvac_analysis/first_collection_v1_20260820
```

- 结果：rc=0；64-row query表、3,200-row horizon表、8张PNG和summary生成。全部trace finite、join完整、
  `max_abs(x_chain_last-final_action)=0`。
- 首个重要结果：L2/L3/L4 query级Spearman分别为0.827、0.629、0.808；gripper d6/d13合计占L3
  方差24.6%。

### 3.3 v1后检发现q3是post-success混杂

- 后检命令文件`tmp/idea2_first_analysis_postcheck.sh`执行：
  `sha256sum *`、`wc -l`、PIL逐PNG读取尺寸/像素extrema，再按`query_idx,success_before`聚合。
- 结果：所有PNG可解码且非空；CSV为65/3,201行（含header）。更重要的是：q0/q1/q2的16个query
  全是`success_before=false`；q3中14个为true，仅两个failure仍为false。
- 解释：14个成功episode都在q2 C50执行期间首次达到success；因为smoke设
  `ignore_terminations=true`，仍继续执行q3。若把q3下降解释成自然task phase变化，会把post-success状态混进来。
- 修复：不改v1产物；在脚本中给representative frame和episode曲线显式标pre/post-success，并新增
  `FIG09_RECORDED_SUCCESS_BOUNDARY.png`；输出到全新的v2目录。

### 3.4 v2复测与第二个高信息发现

- server `py_compile`通过；v2脚本SHA256为
  `59807e61f13a741645e5bfb61e78f64cff25be1f4ea20e4aa1829067f0fdd59c`；v2运行rc=0。
- 后检仍为64/3,200 data rows，9张PNG全部可解码；原始NPZ/派生CSV hash不变。
- 视觉检查`64 query×50 h`热图时发现多数row向远端h变亮。这个现象直接影响以后能否把方差用于RL
  权重，值得一个窄复核；不增加新数据和统计树。
- 修复：新增唯一一张`FIG10_HORIZON_POSITION_EFFECT.png`，只对
  `success_before=false`的50个query计算future-h基线；输出到全新的v3目录，保留v1/v2不覆盖。

### 3.5 最终v3运行与后检

- 最终脚本SHA256：
  `4d9ec65b095576f62cb979ae4b7169bc793f9f08742cc8d9e5556dce47e7042b`。
- 命令与v1相同，仅output改为：
  `/root/autodl-tmp/idea2_dvac_analysis/first_collection_v3_20260820`。
- 结果：rc=0；10张PNG、64-row query CSV、3,200-row horizon CSV与summary。
- v3新增结果（50个pre-success queries）：
  - `h`与跨query中位`V_L3(h)` Spearman 0.852；
  - 后半/前半h均值比中位1.274；
  - 78% query后半均值更高；70% query的最大值在后半。
- 后检：10张PNG尺寸从2144×704到2576×2180不等，均能由PIL读取且RGB extrema覆盖0–255；query/
  horizon CSV行数精确；hash已写入summary和本地压缩包。

## 4. 产物传输与本地检查

### 4.1 服务器打包

```bash
cd /root/autodl-tmp/idea2_dvac_analysis/first_collection_v3_20260820
test ! -e first_collection_v3_20260820_artifacts.tar.gz
tar -czf first_collection_v3_20260820_artifacts.tar.gz -- *.png *.csv *.json
sha256sum first_collection_v3_20260820_artifacts.tar.gz
du -h first_collection_v3_20260820_artifacts.tar.gz
```

- 结果：3.5 MiB；SHA256
  `01ec0cab94a28aee064fa2ecf3fac2377200cdf4554e492e84a3a1c63beae4d0`。

### 4.2 SFTP与本地解包

- 用同一Paramiko helper `get`到原先不存在的
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/first_data_analysis_v3/`。
- 本机`Get-FileHash`与服务器SHA完全相同；`tar -xzf`解包后逐文件大小和服务器一致。
- 用`view_image`实际查看FIG01/02/04/05/06/07/08/09/10；文字、图像、色条和数据均可读，没有空白图或
  裁剪关键信息。
- 原始NPZ没有下载；Windows只保存约3.6 MB派生包、CSV、JSON和PNG。
- 结束时清除当前进程`SEETA_SSH_PASSWORD`并关闭会话。

## 5. 视频/逐control recorder只读源码审计

- 当前6帧严格来自`1 reset + 3普通chunk终态 + 最后一chunk真实终态 + 1 auto-reset spill`；
  `200/C50=4`，不是编码器漏帧。
- `/root/autodl-tmp/RoboTwin_RLinf`的VectorEnv强制native `eval_video_log=false/render_freq=0`；本次没有
  第二套RoboTwin native视频。旧`/root/autodl-tmp/RoboTwin/eval_result`的历史视频来自独立evaluator。
- 若要逐control帧，最窄插点是qpos control loop每次`scene.step()`与render/check_success之后；新增
  default-off `control_trace`配置、每env独占writer、frames.csv和TOPP mapping，不改C50轨迹。
- frame不能天然对应唯一整数h：左右臂分别压缩/TOPP并异步推进；应保存两臂各自
  `h_lo/h_hi/fraction`。生成该mapping还必须在现有那一次compression/TOPP调用中旁路保留
  原始h到轨迹sample的关系，不能重复调用带随机常量路径分支的`compress_path()`。本轮只完成设计，
  没有改RoboTwin代码或启动采集。

## 6. 并发审计结论（未运行）

- 32 total env = 16 env/GPU，等于official 128/8的每GPU密度，是下一档合理资源测量。
- 64 total env = 32 env/GPU，是official密度两倍；只应在32实测显示吞吐收益后再考虑。
- 粗外推：32约25–35 GiB/GPU、75–100 GiB cgroup；64约35–55 GiB/GPU、120–170 GiB cgroup；
  仅容量估计，不是阈值或实测事实。
- 现有fixed-seed分区会随env数扩大前缀：16→32保留当前16并新增16个ID，16→64保留当前16并新增48个；
  不是全量重复。若目的是取得完全无重叠的增量episode，优先显式制作两份互斥32-ID shard串行。
  并发主要决定吞吐，不会提高单episode时间粒度。本轮没有启动32/64。

## 6.1 最终只读复核后的措辞修正

- 数值复核确认CSV/JSON中的counts、hash、L相关、维度占比、future-h统计、q0–q3中位数和14/2 outcome
  与正文一致。
- 把“signal包含task-state信息”收紧为“随query/state与noise组合变化”；每状态只有一次noise，不能分离
  state贡献。
- 明确用户附件图2其实是论文Figure 4；另补论文Figure 2的per-h threshold/crossing语义，以及
  crossing时`max(N_min,k*)`、无crossing时`N_max`的规则。
- 修正Figure 13边界：更细帧只改善phase标签对齐；只有增加query/replan频率才会增加`V_total`时序点。

## 7. 最终变更清单

新增：

- `local_scripts/analyze_dvac_first_collection.py`：可复跑的CPU离线分析脚本；
- `docs/rlinf-robotwin-pi0-dvac-telemetry/04_FIRST_DATA_ANALYSIS.md`：结论与教学说明；
- 本流水账；
- `evidence/first_data_analysis_v3/`中的10张PNG、2个CSV、1个JSON与压缩包。

更新：

- `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md`：阶段H改为完成并路由首轮分析；
- 根`HANDOFF.md`：当前停点从“等待分析”改为“首轮分析完成，等待决定phase/recorder/32-ID次序”。

未改：RLinf/RoboTwin模型与runtime source、原始smoke output、H/C/M、action、checkpoint、任何训练代码。
