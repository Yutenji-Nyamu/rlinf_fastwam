#!/usr/bin/env bash
set -eu
id
sudo -S -p '' /usr/bin/python3 -c '
import os, pathlib, subprocess
print("TIME",subprocess.check_output(["date","-Is"],text=True).strip())
rows=subprocess.check_output(["ps","-u","liwenbo","-o","pid=,ppid=,etime=,pcpu=,rss=,args="],text=True)
for row in rows.splitlines():
    if "run_robodojo_main.py" not in row and "server_policy.py" not in row: continue
    print("PROCESS",row)
    pid=int(row.split()[0]); root=pathlib.Path("/proc")/str(pid)
    for name in ("cwd","fd/1","fd/2"):
        try: print("PROC",pid,name,os.readlink(root/name))
        except OSError: pass
    allow={"CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","NVIDIA_VISIBLE_DEVICES"}
    try:
        for item in (root/"environ").read_bytes().split(b"\0"):
            k,_,v=item.partition(b"=")
            if k.decode(errors="replace") in allow: print("DEVICE_ENV",pid,k.decode(),v.decode(errors="replace"))
    except FileNotFoundError: pass
base=pathlib.Path("/data/liwenbo/starvla-assets/simulators/RoboDojo")
for rel in ("XPolicyLab/policy/starVLA/scripts/run_robodojo_main.py","src/eval_client/main.py"):
    p=base/rel
    if not p.is_file(): continue
    lines=p.read_text().splitlines(); selected=set()
    for i,line in enumerate(lines):
        if any(s in line for s in ("CUDA_VISIBLE", "active_gpu", "physics_gpu", "multi_gpu", "device_id", "AppLauncher", "SimulationApp")):
            selected.update(range(max(0,i-3),min(len(lines),i+5)))
    print("GPU_ROUTING_SOURCE",p)
    for i in sorted(selected): print(i+1,lines[i])
'
