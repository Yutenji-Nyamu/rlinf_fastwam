from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
TOPIC = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt"
EVIDENCE = TOPIC / "evidence"
TARGET = ROOT / "exports" / "shenzhen_dvac_teaching_figures_20260823.zip"

DOCUMENTS = [
    TOPIC / "13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md",
    TOPIC / "15_DVAC_64_ROLLOUT_AND_SIGNAL_ANALYSIS_PLAN_20260822.md",
    TOPIC / "16_DVAC_FIRST_REAL_RESULT_20260822.md",
    TOPIC / "18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md",
    TOPIC / "19_DVAC_SIGNAL_BY_SIGNAL_TEACHING_20260823.md",
    TOPIC / "21_DVAC_DETAILED_ACTION_TIMELINE_TEACHING_20260823.md",
]

LEDGERS = [
    EVIDENCE / "15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md",
    EVIDENCE / "17_DVAC_SIGNAL_TEACHING_REBUILD_LEDGER_20260823.md",
    EVIDENCE / "19_DVAC_DETAILED_TIMELINE_REBUILD_LEDGER_20260823.md",
]

EVIDENCE_DIRS = [
    EVIDENCE / "dvac-analysis-p1-fixed64-20260822",
    EVIDENCE / "dvac-analysis-fastwam-move-stapler-phase-20260823",
    EVIDENCE / "dvac-analysis-all-four-tasks-20260823",
    EVIDENCE / "dvac-signal-teaching-20260823",
    EVIDENCE / "dvac-detailed-action-timeline-20260823",
]

SCRIPTS = [
    ROOT / "local_scripts" / "analyze_dvac_first_collection.py",
    ROOT / "local_scripts" / "analyze_shenzhen_dvac_observation.py",
    ROOT / "local_scripts" / "test_analyze_shenzhen_dvac_observation.py",
    ROOT / "local_scripts" / "render_shenzhen_dvac_cross_task_20260823.py",
    ROOT / "local_scripts" / "render_shenzhen_dvac_signal_teaching_pillow_20260823.py",
    ROOT / "local_scripts" / "render_shenzhen_dvac_detailed_timeline_20260823.py",
]

README = """# 深圳 DVAC 教学、图册与可复核材料包

## 建议阅读顺序

1. `docs/rlinf-shenzhen-pi0-ppo-rlt/21_DVAC_DETAILED_ACTION_TIMELINE_TEACHING_20260823.md`
   - 最终、最细的逐指标、逐任务、逐 case 教学；主入口。
2. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-detailed-action-timeline-20260823/FIGURE_GALLERY.md`
   - 77 张最终图的完整图册。
3. `docs/rlinf-shenzhen-pi0-ppo-rlt/19_DVAC_SIGNAL_BY_SIGNAL_TEACHING_20260823.md`
   - raw y、Position、r、R、S、I 的逐信号解释和第一轮重绘。
4. `docs/rlinf-shenzhen-pi0-ppo-rlt/16_DVAC_FIRST_REAL_RESULT_20260822.md`
   - π0 与 Fast-WAM 首批真实 rollout 的总体统计和结果边界。
5. `docs/rlinf-shenzhen-pi0-ppo-rlt/18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md`
   - 两通道/四项分解的高层解释以及当时服务器背景。

## 包内内容

- 6 份结果、教学与实验口径 Markdown。
- 5 个完整派生证据目录：所有教学 PNG、storyboard、marker 原帧、π0 query 图、7 条代表视频、CSV/JSON。
- 3 份对应流水账。
- 6 个分析、测试与绘图脚本。
- `MANIFEST.csv`：包内每个实际文件的相对路径和原始字节数。

目录结构保留工作区相对路径，因此从上述 Markdown 打开其相对图片链接即可直接阅读。
三份内容重复的 `.tar.gz` 冷归档没有再次塞进 ZIP；其已展开目录完整收入。
模型 checkpoint、训练日志、环境和原始大规模 telemetry 不在本教学包内。

DVAC 论文：<https://arxiv.org/abs/2606.03847v1>
"""


def collect_files() -> list[Path]:
    explicit = [*DOCUMENTS, *LEDGERS, *SCRIPTS]
    missing = [path for path in explicit if not path.is_file()]
    missing += [path for path in EVIDENCE_DIRS if not path.is_dir()]
    if missing:
        raise FileNotFoundError("missing bundle input:\n" + "\n".join(str(path) for path in missing))

    files = list(explicit)
    for directory in EVIDENCE_DIRS:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix().lower())
    arcnames = [path.relative_to(ROOT).as_posix() for path in files]
    if len(arcnames) != len(set(arcnames)):
        raise ValueError("duplicate archive names")
    return files


def manifest_csv(files: list[Path]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["path", "bytes"])
    for path in files:
        writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size])
    return buffer.getvalue()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite: {TARGET}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    input_bytes = sum(path.stat().st_size for path in files)

    with zipfile.ZipFile(TARGET, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README_FIRST.md", README)
        archive.writestr("MANIFEST.csv", manifest_csv(files))
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(TARGET, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        names = archive.namelist()
        if len(names) != len(files) + 2:
            raise RuntimeError(f"unexpected ZIP member count: {len(names)}")

    print(f"target={TARGET}")
    print(f"source_files={len(files)}")
    print(f"members={len(files) + 2}")
    print(f"input_bytes={input_bytes}")
    print(f"zip_bytes={TARGET.stat().st_size}")
    print("zip_test=OK")


if __name__ == "__main__":
    main()

