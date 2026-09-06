from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = (
    ROOT
    / "tmp/rlt_success_bc_matched_width_stopped_raw_20260827_v1_extract"
    / "rlt_success_bc_matched_width_stopped_light_20260827_v1"
)
OUT = (
    ROOT
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence"
    / "rlt_success_bc_matched_width_stopped_final_20260827"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


comparison = load("rlt_comparison", ROOT / "tmp/analyze_rlt_matched_width_v3_live_20260827.py")
comparison.CONTROL = RAW / "control/metrics.log"
comparison.METHOD = RAW / "method/metrics.log"
comparison.OUT = OUT
comparison.main()
shutil.copy2(OUT / "summary.json", OUT / "comparison_summary.json")

diagnostics = load("rlt_diagnostics", ROOT / "tmp/plot_rlt_success_bc_formal_live.py")
diagnostics.CONTROL_LOG = RAW / "control/metrics.log"
diagnostics.METHOD_LOG = RAW / "method/metrics.log"
diagnostics.RESOURCE_CSV = RAW / "pair/paired_resources.csv"
diagnostics.OUT = OUT
diagnostics.main()
shutil.copy2(OUT / "summary.json", OUT / "diagnostics_summary.json")
(OUT / "RLT_CONTROL_VS_SUCCESS_BC_G111.png").replace(
    OUT / "RLT_CONTROL_VS_SUCCESS_BC_STOPPED_FINAL.png"
)
