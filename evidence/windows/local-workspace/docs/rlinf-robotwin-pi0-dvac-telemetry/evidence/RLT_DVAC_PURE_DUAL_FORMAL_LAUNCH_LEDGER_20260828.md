# RLT-DVAC-Pure 双单卡 formal 启动流水

日期：2026-08-28

## F001 本机上下文与实验合同

- 已完整读取根规则、`PROJECT_CONTEXT.md`、`HANDOFF.md`及本专题`00_INDEX_AND_PLAN.md`。
- 用户授权在两张卡分别fresh启动Pure nominal `[0,2]`与`[0,5]`，启动健康后停止盯跑。
- 两条均继承matched-width单卡干净RLT：480 cycles、train8、fixed20、B512/MB256、warmup20k、
  replay80k、eval/save25；仅方法强度、GPU、名称和输出路径不同。

## F002 服务器启动前现场

- 指令：只读检查主机身份、Pure worktree HEAD/dirty、GPU/RAM、OOM事件、磁盘、Ray/训练进程和目标目录。
- 结果：主机`autodl-container-nekaqbwt43-6ce5babb`；两卡均空闲；cgroup约56.8 GiB；
  `oom=0, oom_kill=0`；`/root/autodl-tmp`可用677 GiB；无旧Ray/训练进程；两条目标目录均不存在。
- 源码：Pure worktree干净，初始HEAD=`cb88e9c5...`。

## F003 配置增量与推送

- `[0,2]`沿用已smoke的mean-one Pure映射：`strength=.5`。
- `[0,5]`沿用同一公式并只改为`strength=2.0`；正式名写`s2p0`，避免把mean-one后的实际权重误写成严格边界。
- 新增`s2p0`薄配置及GPU1 placement薄配置；compose均成功。
- `s0p5`与matched-width干净RLT的差异仅为Pure方法字段及run路径；`s2p0`相对`s0p5`仅为
  `strength`、方法输出目录、实验名；GPU1配置再只改三个component placement为1。
- Git：最终HEAD=`a2ae5cbe81049fb7027c43ea85483ed3ffc3ce2f`，已推送
  `personal/codex/rlt-dvac-pure-reference-bc`。首次普通push遇HTTP/2传输失败；仅对该次push使用服务器现有
  network-turbo环境重试后成功，未改训练环境。

## F004 正式启动

- 时间：2026-08-28 00:17 CST。
- 指令：启动一个看见GPU0/1的shared Ray，再启动`s0p5/GPU0`；`s2p0/GPU1`错开120秒冷启动。
- 输出：
  - `/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1`
  - `/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1`
- 资源CSV只读记录GPU/RAM，每5秒采样；没有资源阈值或自动停止行为。
- PID：Ray `274999`，s0p5 wrapper `275914`，s2p0 wrapper `275915`。

## F005 首轮健康检查

- 两条compose均`exit=0`，resolved YAML及SHA256已落盘。
- placement已现场确认：s0p5为GPU0，s2p0为GPU1。
- s0p5已建立actor/rollout/env/critic并进入`Generating Rollout Epochs`；s2p0已启动并建立GPU1 placement，
  正在相同初始化链。
- 现场`oom=0, oom_kill=0`。Curobo缺失traceback是历史RoboTwin的可选planner导入提示；进程继续运行，
  本任务实际使用MPLib。
- 00:25 CST最终启动检查：s0p5/s2p0首轮rollout分别耗时`232.42/223.78`秒，均已写出
  `Global Step 1/480`并进入下一轮；两条driver及2组actor/rollout/env均alive。按用户要求至此停止主动盯跑。

## F006 旧Step475 replay的C10时间线只读提取

- 指令：定位旧matched-width方法Step475的唯一rank0 replay；读取权威`trajectory_id_list`，只扫描最后
  4,000个one-row trajectory，以当前action对齐的done切分episode；全程`CUDA_VISIBLE_DEVICES=''`。
- 结果：重建329条完整student episode，其中成功281、失败48；导出成功/失败各低中高DVAC共6条的CSV、
  NPZ与四张PNG。
- 数据事实：57,957总row低于80k容量；每row含H50三种DVAC及C10动作/reference，但不含图片、reset ID、
  env slot或runner step。
- 观察：87.39% query的h10--49均值高于h0--9，平均差`0.222 log10`；聚合C10曲线没有成功前单调升高。
- 输出：`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_step475_c10_episode_probe_light_20260828_v1/`；
  服务器轻量tar为1,020 KiB，SHA256=`72f6f599a1d2ad1c3f2879e4ecb8d935faf6da9b9bad08d4503fe64ef6651cf1`。
