# 八条两卡 GRPO 曲线重绘流水账

- 任务：只使用已有本地正式实验日志，统一重画逐步成功率、5 步均值、10 步均值和 fixed-32；不访问或干预服务器训练。
- 纳入：Control、ST `[0,2]`、Prism、旧 Action-Adv、Action-Adv Fix `[0,2]`、ST `[0.5,1.5]`、Action-Adv Fix `[0.5,1.5]`、ST `[0.8,1.2]`。
- 排除：smoke、失败重跑、四卡实验、AutoDL 实验和不同采样预算实验。
- 统一合同：`64 train env × 4 rollout epochs`、G8、B1024/MB32/update2、fixed32/eval5。
- 真实横轴：从各日志的 Step1 到该实验真实最终/最新完整 step；不伪造 Step0，不补齐短实验。
- 制图规则：八条曲线使用高对比颜色，并叠加独立线型和 marker；每个指标单独一张 PNG。
- 生成脚本：`local_scripts/render_sz_eight_2gpu_grpo_separate_20260831.py`。
- 初次渲染发现 Action-Adv Fix `[0,2]` 使用了 Step79 的旧 live 日志；改接已封存的 Step81 终态日志后重新生成全部图、CSV、JSON 与 HTML。
- QA：四张 PNG 均为 `1760×1000` 且成功解码；图例完整显示 8 条实验，最大横轴 Step96，短实验均在真实终点停止；`summary.json` 与图例终点一致。
