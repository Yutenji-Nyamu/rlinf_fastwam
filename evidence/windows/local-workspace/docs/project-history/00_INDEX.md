# AutoDL / RLinf 项目历史索引

本目录保存从根目录移出的历史快照。它们用于追溯，不是当前规则或动态状态；新任务默认只读根目录 `AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md` 和当前专题唯一计划。

## 2026-09-06 用户授权大幅精简

- `context-20260906-before-trim/`：精简前AGENTS、PROJECT_CONTEXT、HANDOFF逐字副本，SHA256已核对一致。
- 根入口改为短规则/习惯/当前路由；连续追问不重复全文读未变化文件，专题按当前问题读章节，09-03窗口交接不再必读。
- 旧文本中的相对路径按各文件原所在目录解释。过程与尺寸见`../server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_LEDGER_20260906.md`。

## 2026-07-28 根目录精简归档

- `AGENTS_SNAPSHOT_20260728.md`：精简前的规则快照，包含 Fast-WAM 专题级执行细节。
- `PROJECT_CONTEXT_SNAPSHOT_20260728.md`：精简前的长期上下文快照，包含服务器路径、环境和 Fast-WAM 具体实现值。
- `HANDOFF_TIMELINE_THROUGH_20260728.md`：精简前约 29 KiB 的累计交接时间线及最后已知快照。

## 2026-08-07 根交接再次精简

- `HANDOFF_SNAPSHOT_20260807_PRE_COMPACTION.md`：本次精简前的完整根交接，保留 DSRL、RLT、
  QAM 累计运行时间线和旧 OGPO 停点。它只用于追溯；根 `HANDOFF.md` 已恢复为短路由、当前
  OGPO 停点与授权边界，避免每个新任务按规则全文加载其他专题历史。

## 2026-09-03 当前窗口交接精简

- `HANDOFF_SNAPSHOT_20260903_PRE_WINDOW_HANDOFF.md`：精简前的完整约95 KB根交接，保留截至
  2026-09-03 20:40 CST的深圳RLinf、Fast-WAM、Sidney pi0.5及服务器管理累计时间线。
- 根`HANDOFF.md`已重新缩为当前活动实验、关键代码锁和专题路由；近期窗口详情放在
  `../window-handoffs/20260903_RLINF_SHENZHEN_WINDOW_HANDOFF.md`。

当前有效入口仍是根目录：

- `../../AGENTS.md`
- `../../PROJECT_CONTEXT.md`
- `../../HANDOFF.md`

## 专题历史

- DSRL 完整参考历史：`../rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md`
- DSRL 当前唯一计划：`../rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
- Fast-WAM 总索引：`../fastwam-robotwin-rlinf-grpo/00_INDEX.md`
- Fast-WAM 唯一实施计划：`../fastwam-robotwin-rlinf-grpo/05_IMPLEMENTATION_PLAN.md`

## 工作区外旧材料

- `C:\Users\86136\Documents\Codex\2026-07-13\n\autodl-rlinf-fastwam-handoff-20260716.md`：早期 Fast-WAM 基线，仅在追溯旧环境或旧决定时按需读取。

归档规则：不删除历史；不把快照重新并入当前入口；若新增归档，必须在本索引注明日期、来源和用途。
