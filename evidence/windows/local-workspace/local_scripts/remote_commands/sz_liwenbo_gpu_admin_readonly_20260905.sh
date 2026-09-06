#!/usr/bin/env bash
set -eu
id
sudo -S -p '' /usr/bin/python3 -c '
import os, pathlib, subprocess
p=201345
root=pathlib.Path("/proc")/str(p)
print("ADMIN_READONLY_PID",p)
if not root.exists():
    print("PROCESS_ALREADY_EXITED"); raise SystemExit(0)
for f in ("cwd","exe","fd/1","fd/2"):
    try: print(f,os.readlink(root/f))
    except OSError as e: print(f,type(e).__name__)
# Only device-routing variables; never print the full environment.
allow={"CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","NVIDIA_VISIBLE_DEVICES","DISPLAY"}
for row in (root/"environ").read_bytes().split(b"\0"):
    key,_,value=row.partition(b"=")
    if key.decode(errors="replace") in allow: print("DEVICE_ENV",key.decode(),value.decode(errors="replace"))
parent=int(next(x.split()[1] for x in (root/"status").read_text().splitlines() if x.startswith("PPid:")))
print(subprocess.check_output(["ps","-o","user:16,pid,ppid,etime,args","-p",str(parent)],text=True))
# Read only the already identified simulator entry wrapper, not datasets/weights.
script=pathlib.Path("/data/liwenbo/starvla-assets/simulators/RoboDojo/XPolicyLab/policy/starVLA/scripts/run_robodojo_main.py")
for i,line in enumerate(script.read_text().splitlines(),1):
    if any(x in line for x in ("CUDA","device","runpy","sys.argv","main.py")): print("WRAPPER",i,line)
'
