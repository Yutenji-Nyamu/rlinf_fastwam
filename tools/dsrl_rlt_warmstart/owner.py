# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
p = json.loads((d / "plan.json").read_text())
if p["bench"] == "dojo":
    if p.get("method") == "rlt":
        from tools.dsrl_rlt_warmstart.rlt_dojo_owner import main
    else:
        from tools.dsrl_accel_v2.dojo_owner import main
else:
    from tools.dsrl_rlt_warmstart.rt2_owner import main
main(str(d))
