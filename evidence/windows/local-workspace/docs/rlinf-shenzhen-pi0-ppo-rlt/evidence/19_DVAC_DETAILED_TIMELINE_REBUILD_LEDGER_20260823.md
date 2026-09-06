# 深圳 π0 / Fast-WAM DVAC 逐 action 时间轴教学重建流水账

日期：2026-08-23。机器标签：`WIN-LOCAL`（文档、离线重算与制图）与 `SZ-H100`（只读原始数据复核和小型代表视频取回）。

## 授权与边界

- 用户明确要求重新读取深圳服务器原始 telemetry、视频与既有分析材料，重算并绘制更细的时间轴教学图。
- 服务器操作限于只读盘点、读取和下载所需小型原件；不改变 GRPO、π0、Fast-WAM 的 source、环境、配置、进程或既有结果。
- 不删除、覆盖服务器产物，不启动新的推理或训练。
- 代表 case 按既有预注册 outcome 内 episode 长度中位数规则保留；新增抽帧点可由 phase 边界、曲线极值和解释目标选择，但不得拿它们替代总体统计。
- 凭据只注入当前 SSH 进程，不写入文件、命令文件或本账本。

## L19-001 任务恢复与现有材料盘点

时间：2026-08-23 14:31 CST。

执行：

1. 完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与本专题唯一事实源 `00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
2. 读取现有 DVAC 计划、结果、教学文档与分析器/制图脚本的结构。
3. 盘点最终五 source 统一分析目录。

结果：

- 最终本地分析根包含 15 个 CSV（83,514,639 bytes）、1 个 JSON、53 个 PNG；没有本地代表 episode MP4。
- 数据合同为 128 episodes、759 queries、70,592 horizon rows；Fast-WAM 有 11,528 个已执行 action/frame 精确对齐行。
- 现有总览图把成功/失败和多种信号挤在一页，不能满足逐 action 帧与曲线点一一对应的教学目标；本轮将保留其统计数据，不复用其版式。

## 后续逐操作记录

## L19-002 DVAC 论文一手材料复核

目标：不用旧聊天或二手摘要决定 $L$ 与图形语法。

读取：

- `https://arxiv.org/abs/2606.03847v1`
- `https://arxiv.org/html/2606.03847v1`
- `https://arxiv.org/pdf/2606.03847v1`

本地保存：

```text
E:\Codex\home\visualizations\2026\08\19\01a01a2f-13d8-7ad1-aa17-ee996384a38a\dvac-paper\2606.03847v1.pdf
```

结果：

- PDF 1,490,579 bytes，SHA-256 `413802B45A8AE3D7524C23A832D57DA554CD0DDA79F19E9290848BFA13E35F59`。
- Eq. (2)--(5) 核清：对最后 $L$ 个 clean endpoint estimates 做 population variance，$V_{total}$ 只是诊断量。
- Appendix 默认 $L=5$；Fast-WAM $M=10$ 可真实取 `z5..z9`，π0 $M=4$ 不可能取 $L=5$。
- 借鉴 Figure 1/4/11 的帧、时间轴、query/chunk 分区与失败反例语法，不把当前 fixed-chunk telemetry 冒充 online adaptive chunking。

## L19-003 深圳服务器原始材料只读盘点

远端命令文件：

```text
local_scripts/remote_commands/shenzhen_dvac_detailed_timeline_inventory_20260823.sh
SHA-256 259D10C1E3A1FE9C96E098D812EA837A21D14A55BB0D3A3223CA4C8F95785902
```

执行入口：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'local_scripts\remote_exec_autodl.py' `
  --host 120.241.223.9 --port 22 --user chenyiteng `
  --host-key-sha256 'qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY' `
  run-script 'local_scripts\remote_commands\shenzhen_dvac_detailed_timeline_inventory_20260823.sh'
Remove-Item Env:\SEETA_SSH_PASSWORD
```

现场结果：

| telemetry root | `du -sh` | files |
|---|---:|---:|
| π0 fixed-64 | 40M | 786 |
| Fast-WAM adjust | 14M | 323 |
| Fast-WAM move | 27M | 651 |
| Fast-WAM turn | 21M | 535 |
| Fast-WAM pick | 27M | 515 |

- 7 条 Fast-WAM 代表 MP4 均存在，合计 3,742,013 bytes。
- 两条 π0 CSV 所列 rank MP4 路径现场不存在；没有用别的视频替代。
- 两条 π0 代表 episode 的 24 张三相机 query 图均存在，合计 1,024,310 bytes。
- 只读 GRPO probe 仍见 4 个 EnvWorker；未停止、修改或向训练进程发信号。

问题与窄修：第一次命令对缺失 π0 MP4 使用 `test -f`，在 `set -e` 下 exit 1，后续信息未输出。修改为明确打印 `VIDEO_MISSING` 并 `continue`，第二次 exit 0。缺失视频是现场事实，不是模型或分析失败。

## L19-004 代表原件下载

下载前 C: free 约 36.52 GiB；目标都是本专题 evidence，小文件不会触及 checkpoint 或缓存。

Fast-WAM 视频脚本：

```text
local_scripts/download_shenzhen_dvac_representative_videos_20260823.ps1
SHA-256 560A60BCE34DCF957820F63292639540BAF5CEEE04906ECD515C1D73E271963F
```

执行：

```powershell
& 'local_scripts\download_shenzhen_dvac_representative_videos_20260823.ps1'
```

结果：7 MP4，3,742,013 bytes，目录：

```text
docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-detailed-action-timeline-20260823/videos
```

π0 query 图脚本：

```text
local_scripts/download_shenzhen_pi0_representative_query_images_20260823.ps1
SHA-256 E3565B9240930B851620F6F53A12F11FACDFD23D6D22D843842D42237F5796BE
```

执行：

```powershell
& 'local_scripts\download_shenzhen_pi0_representative_query_images_20260823.ps1'
```

结果：24 PNG，目录：

```text
docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-detailed-action-timeline-20260823/pi0-query-images
```

两个脚本都先拒绝覆盖现有目标，密码通过 `Read-Host -AsSecureString` 只进入当前进程，SFTP 使用固定 host-key 指纹。

## L19-005 本地视频读取依赖

现有 bundled Python 没有 OpenCV。为了不向 C: 项目环境安装包，依赖单独放到 E:：

```powershell
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pip install --target `
  'E:\Codex\home\visualizations\2026\08\19\01a01a2f-13d8-7ad1-aa17-ee996384a38a\video-deps' `
  --no-deps 'opencv-python-headless==5.0.0.93'
```

结果：下载约 43.8 MB wheel；没有修改服务器环境、RLinf/Fast-WAM 环境或项目依赖文件。

7 个视频均为 H.264、640×480、10 fps；帧数分别为 118、150、400、67、400、107、400，与 episode action/terminal 边界闭合。

## L19-006 新分析与制图器

实现：

```text
local_scripts/render_shenzhen_dvac_detailed_timeline_20260823.py
SHA-256 AE7BCBCA86E9F45DD4CB9CA8F74B4855240956EA79CF5471F8039EAFB4F176BA
```

正式执行命令：

```powershell
$env:PYTHONPATH='E:\Codex\home\visualizations\2026\08\19\01a01a2f-13d8-7ad1-aa17-ee996384a38a\video-deps'
& 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'local_scripts\render_shenzhen_dvac_detailed_timeline_20260823.py'
```

输出合同：

- 9 个代表 case × 5 个单信号图 = 45。
- 5 个 policy/task group × 4 个总体信号图 = 20。
- 5 个位置中心/尺度图。
- 7 个任务、信号、轴、tail、S/I、位置-phase 参考图。
- 合计 77 张最终 figure；另外保存 77 张 marker 原帧。
- Fast-WAM case 主口径 $L=5$；π0 主口径 $L=3$。
- 每张 case 图都使用全 finite min/max 加 padding，不再用 percentile 截断极值。
- normalized progress 与 absolute slot 总体曲线均先在 episode 内聚合，再跨 episode 画 mean ± 1 SD 和 `n_at_risk`。
- outcome 差按 episode-first，bootstrap 10,000 draws。

## L19-007 制图过程中发现的问题与修复

### 1. Fast-WAM episode UID 跨任务重复

问题：`episode0004_reset3` 等编号在多个任务中重复；若只按 suffix 选取，会把 move/pick 等任务混到同一个 case。

修复：每个 case 同时锁 `(policy, task, episode_uid)`。复测精确行数：π0 200/200 actions；Fast-WAM adjust117、move149/400、turn66/400、pick106/400。

### 2. 旧 L=3 数值混入 L=5 图注

问题：pick failure marker 草稿硬编码 q12 `+3.50`、q15 `+4.23`，这是旧 $L=3$。

修复：marker 只写动作语义，图中数值由当前数据动态读取。$L=5$ 为 q12 `+3.321`、q15 `+3.904`。

### 3. slot319 说明图误取 slot239 行

问题：独立 S/I 说明图先按 L 过滤但漏了 action slot，导致右侧 slot319 重复显示 slot239。

修复：行选择改为 `(L, action_slot)` 双条件。复测：

| slot | L | S | I | R |
|---:|---:|---:|---:|---:|
| 239 | 3 | +2.645 | +1.942 | +4.587 |
| 239 | 5 | +3.142 | +1.785 | +4.928 |
| 319 | 3 | +1.620 | -1.557 | +0.063 |
| 319 | 5 | +1.601 | -1.616 | -0.015 |

### 4. Windows 文件名大小写不敏感

问题：最初用 `__r.png` 与 `__R.png` 区分 raw/standardized residual；Windows 把它们视为同一路径，后者覆盖前者。

修复：改成 `__raw_residual.png` 与 `__standardized_residual.png`。正式重画后，先精确列出本轮 36 个旧命名重复文件，再只删除这些生成错误副本；最终 figures 恰为 77，不涉及原始视频、CSV、query 图或用户文件。

### 5. π0 post-success 可见性

问题：旧版只在文字里说 q3 是 post-success，曲线背景仍与普通 query 相似。

修复：q3 区域单独灰出并直接标 `post-success`；仍保留描述性数值，不进入 baseline/outcome 主统计。

## L19-008 最终产物与 QA

主文：

```text
docs/rlinf-shenzhen-pi0-ppo-rlt/21_DVAC_DETAILED_ACTION_TIMELINE_TEACHING_20260823.md
SHA-256 173BBE4A1052B2F2DAFDA42348FCE027BE656C7130584244CD48A75D99C24638
```

完整图册：

```text
docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-detailed-action-timeline-20260823/FIGURE_GALLERY.md
SHA-256 786778050AF7C6F5257650A2BA61EC6F59688569C17161F8F9810C8D987D8246
```

最终 evidence root：

| 类型 | 数量 | bytes |
|---|---:|---:|
| CSV | 7 | 165,538 |
| JSON | 1 | 749 |
| Markdown | 1 | 9,187 |
| MP4 | 7 | 3,742,013 |
| PNG | 178 | 35,069,964 |

PNG 组成：77 figures + 77 marker frames + 24 π0 原始 query images。C: 最终 free 36.41 GiB；E: free 341.33 GiB。

QA：

- 图册 Markdown 精确引用 77 张图，缺失路径 0。
- 主文 18 张核心内嵌图，缺失路径 0；其余全部由图册顺序展示。
- 主文未使用 `\(...\)` / `\[...\]` 旧公式定界；未转义美元符号数为 414（偶数）。
- 四张高风险图已人工查看：move failure S、slot239/319、position-phase aliasing、turn aggregate S；另查看 signal dictionary、pick failure R、π0 success S、tail sensitivity。
- 所有服务器操作只读或 SFTP get；未改变 GRPO、π0、Fast-WAM 的进程、代码、环境或结果。

## L19-009 教学材料完整 ZIP（用户请求）

目标：将最终教学、完整图册、所有被教学文档引用的派生图与小型数据、代表帧/视频、流水账和可复现脚本整理为一个可直接解压阅读的包；不重复收入已经展开的三份 `.tar.gz`，不包含 checkpoint、训练日志、环境或原始大规模 telemetry。

执行前只读盘点：

- 五个派生 evidence 目录共 341 个文件、202,464,553 bytes；其中 PNG 272、CSV 57、MP4 7。
- C: 可用约 36.35 GiB；目标此前不存在。
- 目标：`exports/shenzhen_dvac_teaching_figures_20260823.zip`。

构建命令：

```powershell
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m py_compile local_scripts/build_shenzhen_dvac_teaching_bundle_20260823.py
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/build_shenzhen_dvac_teaching_bundle_20260823.py
```

结果：

```text
source_files=356
members=358
input_bytes=202911803
zip_bytes=73922151
zip_test=OK
```

ZIP 内含 `README_FIRST.md` 与 `MANIFEST.csv`；保留原工作区相对路径。ZIP 结构复核：最终 figures=`77`、marker frames=`77`、代表 MP4=`7`、π0 query PNG=`24`、Markdown=`11`、CSV=`58`、分析/测试/绘图脚本=`6`。主教学文档18个内嵌图片引用和图册77个引用在 ZIP 内均缺失0；未包含重复 `.tar.gz`/ZIP。构建后 C: 仍约36.31 GiB可用。
