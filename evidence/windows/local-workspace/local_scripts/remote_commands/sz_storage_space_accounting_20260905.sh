set -eu
/usr/bin/python3 - <<'PY'
import os,json,datetime,subprocess
result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'mounts':{}}
for p in ('/home','/data'):
    v=os.statvfs(p)
    result['mounts'][p]={'total':v.f_blocks*v.f_frsize,'used':(v.f_blocks-v.f_bfree)*v.f_frsize,'free_including_reserved':v.f_bfree*v.f_frsize,'available_nonroot':v.f_bavail*v.f_frsize,'free_not_available_nonroot':(v.f_bfree-v.f_bavail)*v.f_frsize,'mount':subprocess.check_output(['findmnt','-T',p,'-n','-o','SOURCE,FSTYPE,TARGET'],text=True).strip()}
print(json.dumps(result))
PY
