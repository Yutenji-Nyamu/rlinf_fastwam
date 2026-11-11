"""Use the accepted exact-owner handoff after 56-env preparation passed."""

import json
import sys
from pathlib import Path

b = Path("/data/chenyiteng/deployment-20261111/cql-pour-56")
r = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/dsrl")
assert json.loads((b / "validation.json").read_text())["config_validation"]
sys.path.insert(0, str(r))
from tools.dsrl_rlt_warmstart import cutover

cutover.B = b
cutover.R = r
cutover.swap.R = r
cutover.main(["6"])
