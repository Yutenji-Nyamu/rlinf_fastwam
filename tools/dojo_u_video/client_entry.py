"""Private import overlay; never edit installed Dojo control or video files."""

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Installed Dojo recording anchor changed: {old}")
    return source.replace(old, new, 1)


def patch_env(source):
    return replace_once(
        source,
        "    return EvalEnv(config, app, resume_state=resume_state, **kwargs)",
        "    from recording import instrument\n"
        "    return instrument(EvalEnv(config, app, resume_state=resume_state, **kwargs))",
    )


def patch_main(source):
    source = replace_once(source, "MAX_INPROC_RESTARTS = 3", "MAX_INPROC_RESTARTS = 0")
    source = replace_once(
        source,
        "eval_num = min(int(_env_eval_num), int(eval_num))",
        "eval_num = int(_env_eval_num)",
    )
    return source


def patch_pipeline(source):
    # Reuse the accepted N25 opt-in change; do not silently cap random tasks at 5.
    return replace_once(
        source,
        'limit = scene_cfg.get("clutter_env_limit", 5)',
        'limit = int(os.environ["DOJO_U_NUM_ENVS"])',
    )


class Loader(importlib.abc.Loader):
    def __init__(self, path, patch):
        self.path, self.patch = path, patch

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(
            compile(self.patch(self.path.read_text()), str(self.path), "exec"),
            module.__dict__,
        )


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        targets = {
            "src.eval_client.eval_env": ("src/eval_client/eval_env.py", patch_env),
            "utils.pipeline_utils": ("utils/pipeline_utils.py", patch_pipeline),
        }
        if fullname not in targets:
            return None
        relative, patch = targets[fullname]
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        expected = Path(os.environ["DOJO_U_REPO"]) / relative
        assert Path(spec.origin).resolve() == expected.resolve()
        spec.loader = Loader(expected, patch)
        return spec


if __name__ == "__main__":
    root = Path(os.environ["DOJO_U_REPO"])
    sys.path[:0] = [str(root), str(root / "XPolicyLab")]
    sys.meta_path.insert(0, Finder())
    source = root / "src/eval_client/main.py"
    # Keep argv[0] pointing here; implicit native self-restarts are disabled.
    exec(
        compile(patch_main(source.read_text()), str(source), "exec"),
        dict(__name__="__main__", __file__=str(source)),
    )
