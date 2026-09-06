# DVAC 逐信号教学重建流水账（2026-08-23）

任务：修复 Typora 公式显示；基于已落盘的 π0 / Fast-WAM telemetry，按 raw DVAC、两通道分解与四项分解逐项重算、制图并编写案例教学。

边界：本轮只读取和分析既有本地副本，不启动新推理、不接触正在运行的 GRPO、不删除或覆盖原始 telemetry。

## 操作记录

### 1. 任务入口与公式初查

- 读取：`PROJECT_CONTEXT.md`、`HANDOFF.md`、当前专题 SSOT `00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
- 检查：`18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md`、`16_DVAC_FIRST_REAL_RESULT_20260822.md`。
- 发现：两份文档仍使用 Typora 不稳定的 `\\[ ... \\]` display-math 定界符；本工作区约定应改为独占一行的 `$$ ... $$`。
- 决策：修复上述既有文档；新教学文档统一使用 `$...$` 行内公式和独占一行的 `$$` 块公式，并在交付前检查旧定界符、未配对 `$`、标题/表格/图片链接。

### 2. 本地数据与字段核对

- 读取根：`evidence/dvac-analysis-all-four-tasks-20260823`。
- 读取表：`query_metrics.csv`、`query_horizon.csv`、`episode_metrics.csv`、`outcome_summary.csv`、
  `L_sensitivity.csv`、`fastwam_action_frame_metrics.csv`、`storyboard_index.csv`、`analysis_summary.json`。
- 核对规模：128 episodes、759 queries、70,592个horizon rows；717个pre-success主分析queries；
  Fast-WAM 11,528个已执行action/frame对齐rows。
- 核对合同：`y=b+r`、`y=mu+P+S_raw+I_raw`、`R=S_std+I_std`；12个命名字段不是12个独立信号。
  教学口径固定为6个观察视图：`y`、`b/P`、`r`、`R`、`S`、`I`。
- 案例选择继续沿用既有“outcome内动作长度接近中位数+固定tie-break”，没有按DVAC挑图。

### 3. 重算与制图

- 首次命令：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'local_scripts/render_shenzhen_dvac_signal_teaching_20260823.py' `
  --analysis-root 'docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-analysis-all-four-tasks-20260823' `
  --output 'docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-signal-teaching-20260823'
```

- 结果：exit 1，bundled Python没有`matplotlib`。这是本地制图依赖问题，不是数据或分析错误。
- 处理：不安装新包、不改系统Python；改用bundled runtime已含的`pandas + numpy + Pillow`，建立
  `local_scripts/render_shenzhen_dvac_signal_teaching_pillow_20260823.py`。删除未使用的matplotlib草稿，
  避免留下一个不可复现入口。
- 成功命令：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'local_scripts/render_shenzhen_dvac_signal_teaching_pillow_20260823.py' `
  --analysis-root 'docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-analysis-all-four-tasks-20260823' `
  --output 'docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-signal-teaching-20260823'
```

- 结果：exit 0；生成10张PNG及4份CSV索引/重算表。PNG合计约2.6 MiB，未下载新数据、未写服务器。
- 图表：信号关系图、raw/L稳健性、Position/scale、r/R outcome、S/I outcome、π0成功失败case、
  Fast-WAM adjust/move/turn/pick action-video时间轴case。
- 第一版raw histogram的两条图例横向拥挤；窄修为step line与纵向图例，重新执行exit 0并视觉核验。

### 4. 文档与Typora修复

- `16_DVAC_FIRST_REAL_RESULT_20260822.md`：4个display-math块由`\\[...\\]`改为`$$...$$`。
- `18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md`：5个display-math块同样修复。
- 新增：`19_DVAC_SIGNAL_BY_SIGNAL_TEACHING_20260823.md`，按6个信号视图固定讲解“来源、单位、可比边界、
  π0表现、Fast-WAM表现、outcome、代表case、帧/时间轴、能说与不能说”。
- 重点边界：π0成功代表case q3明确灰出并标`post-success descriptive only`；Fast-WAM只画已执行
  `h<24`的action/frame；只有move-stapler两条case使用独立人工phase标签。

### 5. QA

- 新教学文档：442行、21,437 bytes、30个本地链接全部存在、9对`$$`展示公式、无`\\[...\\]`、
  无replacement character、扣除display delimiter后行内`$`成对。
- 16/18/19号文档联合检索：旧`\\(...\\)`或`\\[...\\]`数学定界符为0。
- 10张图逐张视觉核验；成功/失败case、action轴、曲线、标题与图片帧均可读。
- 本轮未启动/停止任何服务器进程，未刷新或改变GRPO动态状态。
