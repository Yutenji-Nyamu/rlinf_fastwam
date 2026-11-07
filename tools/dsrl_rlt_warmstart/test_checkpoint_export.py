"""Exercise the real best-evaluation callback with DCP-style RLT checkpoint layout."""

import ast
import json
import tempfile
import types
from pathlib import Path

import torch

src = Path(__file__).with_name("rlt_training_entry.py")
tree = ast.parse(src.read_text())
main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main")
fn = next(
    x for x in main.body if isinstance(x, ast.FunctionDef) and x.name == "progress"
)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cfg = types.SimpleNamespace(
        runner=types.SimpleNamespace(
            val_check_interval=20,
            save_interval=20,
            logger=types.SimpleNamespace(log_path=str(root), experiment_name="formal"),
        ),
        rollout=types.SimpleNamespace(
            rlt_feature_model=types.SimpleNamespace(
                model_path="base", openpi=types.SimpleNamespace(rlt_token_path="token")
            )
        ),
    )
    plan = {
        "run": str(root),
        "protected_output": str(root / "protected"),
        "task": "test",
    }
    ns = {"Path": Path, "json": json, "plan": plan, "phase": "formal"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)  # noqa: S102
    weights = {"actor.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    saved = []

    def save():
        cp = root / "formal/checkpoints" / f"global_step_{obj.global_step}" / "actor"
        (cp / "dcp_checkpoint").mkdir(parents=True)
        (cp / "dcp_checkpoint/.metadata").touch()
        (cp / "model_state_dict").mkdir()
        torch.save(weights, cp / "model_state_dict/full_weights.pt")
        saved.append(obj.global_step)

    obj = types.SimpleNamespace(
        cfg=cfg,
        global_step=20,
        max_steps=400,
        update_rollout_weights=lambda: None,
        evaluate=lambda: {"success_once": 0.0},
        metric_logger=types.SimpleNamespace(log=lambda **kw: None),
        _save_checkpoint=save,
    )
    ns["progress"](obj, 19)
    out = root / "protected/global_step_20/rlt-policy.pt"
    assert torch.equal(
        torch.load(out, weights_only=True)["actor.weight"], weights["actor.weight"]
    )
    assert json.loads((root / "best-evaluation.json").read_text())["round"] == 20
    obj.global_step = 40
    ns["progress"](obj, 39)
    assert not (root / "protected/global_step_40").exists()
    obj.global_step = 60
    obj.evaluate = lambda: {"success_once": 0.5}
    ns["progress"](obj, 59)
    assert (root / "protected/global_step_60/rlt-policy.pt").exists()
    print(
        json.dumps(
            {
                "passed": True,
                "saved": saved,
                "cases": ["first-zero-score-best", "equal-score", "improved-score"],
                "layout": "DCP plus compact weights, no local_shard",
            }
        )
    )
