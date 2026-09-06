#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
CONTROL=$ROOT/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC=$ROOT/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

date --iso-8601=seconds
echo '=== exact run identity ==='
for item in "control:$CONTROL" "dvac:$DVAC"; do
  label=${item%%:*}
  run=${item#*:}
  pid=$(cat "$run/runtime/wrapper.pid")
  echo "--- $label ---"
  echo "run=$run"
  echo "wrapper_pid=$pid alive=$([[ -d /proc/$pid ]] && echo 1 || echo 0)"
  echo "wrapper_cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null || true)"
  echo -n 'wrapper_cmdline='; tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true; echo
  echo 'wrapper_environment:'
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -E '^(CUDA_VISIBLE_DEVICES|RLINF_CODE_WORKING_DIR|RAY_ADDRESS|RAY_NAMESPACE|HYDRA_FULL_ERROR|PWD)=' | sort || true
  echo 'runtime_files:'
  find "$run/runtime" -maxdepth 1 -type f -printf '%f %s\n' | sort
  echo 'resolved_sha256:'
  sha256sum "$run/runtime/resolved.yaml"
  echo 'driver_identity_lines:'
  grep -aE 'RLINF_CODE_WORKING_DIR|Working directory|worktree|Git|commit|python.*main|config-path|config-name|Namespace|namespace' "$run/runtime/driver.log" | head -n 30 || true
done

echo '=== process-derived code roots and git identity ==='
"$PY" - "$CONTROL" "$DVAC" <<'PY'
import os, subprocess, sys

for label, run in (("control", sys.argv[1]), ("dvac", sys.argv[2])):
    pid = int(open(os.path.join(run, "runtime", "wrapper.pid")).read().strip())
    roots = set()
    pids = [pid]
    try:
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
        pids += [int(x) for x in out.split()]
    except subprocess.CalledProcessError:
        pass
    # Also find owned processes whose command line contains the exact run basename.
    basename = os.path.basename(run)
    for ent in os.listdir("/proc"):
        if not ent.isdigit():
            continue
        try:
            cmd = open(f"/proc/{ent}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            if basename in cmd:
                pids.append(int(ent))
        except OSError:
            pass
    for proc in sorted(set(pids)):
        try:
            env = open(f"/proc/{proc}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        vals = {}
        for raw in env:
            if b"=" not in raw:
                continue
            k, v = raw.split(b"=", 1)
            vals[k.decode(errors="replace")] = v.decode(errors="replace")
        root = vals.get("RLINF_CODE_WORKING_DIR")
        if root:
            roots.add(root)
    print(f"{label}: candidate_code_roots={sorted(roots)}")
    for root in sorted(roots):
        def git(*args):
            return subprocess.check_output(["git", "-C", root, *args], text=True, stderr=subprocess.STDOUT).strip()
        try:
            print(f"{label}: root={root}")
            print(f"{label}: head={git('rev-parse','HEAD')}")
            print(f"{label}: branch={git('branch','--show-current')}")
            print(f"{label}: status={git('status','--short')!r}")
            print(f"{label}: top={git('rev-parse','--show-toplevel')}")
        except Exception as exc:
            print(f"{label}: git_error={exc}")
PY

echo '=== complete resolved leaf diff ==='
"$PY" - "$CONTROL/runtime/resolved.yaml" "$DVAC/runtime/resolved.yaml" <<'PY'
import json, sys, yaml

control = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
dvac = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))

def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value[key], path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{idx}]"))
        if not value:
            out[prefix] = value
    else:
        out[prefix] = value
    return out

c = flatten(control)
d = flatten(dvac)
rows = []
for key in sorted(set(c) | set(d)):
    cv = c.get(key, "<MISSING>")
    dv = d.get(key, "<MISSING>")
    if cv != dv:
        rows.append({"path": key, "control": cv, "dvac": dv})
print(f"control_leaves={len(c)} dvac_leaves={len(d)} diff_count={len(rows)}")
for row in rows:
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
PY

echo '=== current mechanism and fatal scalars ==='
"$PY" - "$CONTROL" "$DVAC" <<'PY'
import json, os, re, sys, time

fatal_patterns = {
    "traceback": r"Traceback",
    "oom": r"out of memory|OutOfMemory|OOM-kill|oom_kill",
    "cuda": r"CUDA error|CUDNN_STATUS|NCCL.*error",
    "worker_death": r"worker died|WorkerCrashedError|RayActorError|actor died",
    "vulkan": r"ErrorInitializationFailed|vk::",
    "nonfinite": r"nan detected|nonfinite|\binf\b|\bnan\b",
    "fatal": r"\bfatal\b",
}

keys = [
    "actor/dvac_weight_mean", "actor/dvac_weight_std", "actor/dvac_weight_sq_mean",
    "actor/dvac_weight_ess_fraction", "actor/grad_norm", "actor/approx_kl",
    "actor/clip_fraction", "critic/value_loss", "critic/explained_variance",
]

for label, run in (("control", sys.argv[1]), ("dvac", sys.argv[2])):
    path = os.path.join(run, "runtime", "driver.log")
    text = open(path, encoding="utf-8", errors="replace").read()
    starts = list(re.finditer(r"Global Step:\s+(\d+)/100", text))
    latest = None
    if starts:
        match = starts[-1]
        block = text[match.start():]
        latest = {"step": int(match.group(1))}
        successes = re.findall(r"success_once=([-+0-9.eE]+)", block)
        latest["train_success"] = float(successes[0]) if successes else None
        latest["fixed_success"] = float(successes[1]) if len(successes) > 1 else None
        for key in keys:
            vals = re.findall(re.escape(key) + r"=([-+0-9.eE]+)", block)
            latest[key] = float(vals[-1]) if vals else None
    counts = {name: len(re.findall(pattern, text, re.I)) for name, pattern in fatal_patterns.items()}
    print(json.dumps({
        "label": label,
        "log_bytes": os.path.getsize(path),
        "log_age_seconds": round(time.time() - os.path.getmtime(path), 1),
        "latest": latest,
        "fatal_counts": counts,
    }, ensure_ascii=False, sort_keys=True))
PY
