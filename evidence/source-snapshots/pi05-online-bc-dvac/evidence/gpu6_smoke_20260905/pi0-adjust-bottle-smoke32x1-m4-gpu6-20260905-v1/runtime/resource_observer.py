"""Bounded run-scoped GPU6 / host resource sampling; never controls jobs."""
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

pid=int(sys.argv[1])
out=Path(sys.argv[2])
with (out/'resource.csv').open('x',newline='') as handle:
    writer=csv.writer(handle)
    writer.writerow(['timestamp','wrapper_alive','host_mem_available_kib','swap_free_kib','mem_psi_some_avg10','gpu6_used_mib','gpu6_util_pct','gpu6_temp_c'])
    while True:
        live=Path(f'/proc/{pid}').exists() and not (out/'exit_code.txt').exists()
        mem={x.split(':')[0]:x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()}
        psi=Path('/proc/pressure/memory').read_text().splitlines()[0].split()[1].split('=')[1]
        p=subprocess.run(['nvidia-smi','-i','6','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        gpu=p.stdout.strip().replace(' ','').split(',') if p.returncode==0 else ['','','']
        writer.writerow([datetime.datetime.now().astimezone().isoformat(),int(live),mem['MemAvailable'],mem['SwapFree'],psi,*gpu])
        handle.flush()
        if not live: break
        time.sleep(5)
