"""Check one final Wan environment by real imports, without loading a model.

Run with that environment's Python: --repo REPO --kind oft|pi05
Optional --receipt writes a new JSON file. Exit 0 means imports passed; exit 2
means an import, source-path check, or receipt write failed. No installer status,
downloads, model construction, CUDA kernels, Ray, simulator, or training runs.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--kind", choices=("oft", "pi05"), required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    # Set before any third-party import. Do not call jax.devices/default_backend
    # or torch.cuda availability/device APIs merely to collect metadata.
    restrictions = {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    os.environ.update(restrictions)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind, "python": sys.executable,
        "python_version": sys.version, "prefix": sys.prefix,
        "environment": restrictions, "imports": [], "errors": [],
        "scope": "Actual module imports and loader symbol presence only; no model/assets load or GPU kernel execution.",
    }
    try:
        repo = args.repo.resolve(strict=True)
        if not (repo / "rlinf").is_dir():
            raise ValueError("--repo must contain the selected RLinf checkout")
        report["repo"] = str(repo)
        sys.path.insert(0, str(repo))

        checks = [
            ("torch", ("torch",), None),
            ("torchvision", ("torchvision",), None),
            ("flash_attn", ("flash-attn",), None),
            ("transformers", ("transformers", "rlinf-transformer-openpi"), None),
            ("tokenizers", ("tokenizers",), None),
            ("rlinf.envs.sim.world_model.backend.wan", ("diffsynth",), "WanBackend"),
        ]
        if args.kind == "pi05":
            checks += [
                ("jax", ("jax",), None),
                ("jaxlib", ("jaxlib",), None),
                ("orbax.checkpoint", ("orbax-checkpoint",), None),
                ("openpi.training.config", ("rlinf-openpi",), None),
                ("rlinf.models.embodiment.openpi", (), "get_model"),
                ("rlinf.models.embodiment.openpi.tasks.rl", (), "Pi0RL"),
            ]
        else:
            checks.append(("rlinf.models.embodiment.openvla_oft.rlinf", (), "get_model"))

        for name, distributions, symbol in checks:
            row = {"module": name, "distributions": {}, "ok": False}
            for distribution in distributions:
                try:
                    row["distributions"][distribution] = metadata.version(distribution)
                except metadata.PackageNotFoundError:
                    # Both transformers distributions may claim one package dir;
                    # record their metadata without requiring both to exist.
                    row["distributions"][distribution] = None
            try:
                # Keep stdout machine-readable despite import-time log messages.
                with contextlib.redirect_stdout(sys.stderr):
                    module = importlib.import_module(name)
                filename = getattr(module, "__file__", None)
                row["path"] = str(Path(filename).resolve()) if filename else None
                row["module_version"] = str(getattr(module, "__version__", "unreported"))
                if name.startswith("rlinf."):
                    if filename is None or not Path(filename).resolve().is_relative_to(repo):
                        raise ValueError("Imported RLinf module is outside --repo")
                if symbol:
                    if not callable(getattr(module, symbol, None)):
                        raise AttributeError(f"Missing callable {name}.{symbol}")
                    row["loader_or_class"] = symbol
                if name == "torch":
                    row["compiled_cuda_version"] = module.version.cuda
                    row["cxx11_abi"] = bool(module._C._GLIBCXX_USE_CXX11_ABI)
                row["ok"] = True
            except (Exception, SystemExit) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=8)
                report["errors"].append(f"{name}: {row['error']}")
            report["imports"].append(row)
    except (Exception, SystemExit) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc(limit=8)

    report["ok"] = not report["errors"]
    report["status"] = "CPU_IMPORTS_PASSED" if report["ok"] else "CPU_IMPORTS_FAILED"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.receipt:
        try:
            with args.receipt.open("x", encoding="utf-8") as output:
                output.write(payload)
        except OSError as exc:
            report["ok"] = False
            report["status"] = "CPU_IMPORTS_FAILED"
            report["errors"].append(f"Receipt: {type(exc).__name__}: {exc}")
            payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
