"""Reuse the accepted reservation / exact owner / fallback handoff."""

import sys
from pathlib import Path

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
R = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/dsrl")
assert (B / "validation.json").is_file()
sys.path.insert(0, str(R))
from tools.dsrl_rlt_warmstart import cutover

cutover.B = B
cutover.R = R
cutover.swap.R = R
cutover.main(["4", "6"])
