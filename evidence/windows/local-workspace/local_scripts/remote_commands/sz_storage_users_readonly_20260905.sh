set -eu
id
sudo -S -p '' nice -n 19 ionice -c 3 /usr/bin/python3 -c '
import datetime, json, os, pathlib, pwd, subprocess
print("METADATA_ONLY_STORAGE_START",datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
users={p.pw_name:{"uid":p.pw_uid,"home":p.pw_dir} for p in pwd.getpwall() if 1000<=p.pw_uid<65534}
result={"time_started":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"basis":"du allocated bytes, by top-level user directory; same-filesystem only, no symlink following, not inode-UID quota; metadata only","users":users,"disks":{}}
for mount in ("/home","/data"):
    print("SCANNING",mount,flush=True)
    p=subprocess.run(["du","-x","-B1","--max-depth=1","--",mount],capture_output=True,text=True,timeout=600)
    rows=[]
    for line in p.stdout.splitlines():
        size,path=line.split("\t",1)
        target=pathlib.Path(path)
        stat=target.lstat()
        try:owner=pwd.getpwuid(stat.st_uid).pw_name
        except KeyError:owner=str(stat.st_uid)
        rows.append({"path":path,"allocated_bytes":int(size),"owner":owner,"user_directory":target.name in users and path!=mount})
    result["disks"][mount]={"rc":p.returncode,"rows":sorted(rows,key=lambda r:r["allocated_bytes"],reverse=True),"metadata_errors":p.stderr[:3000],"df":subprocess.check_output(["df","-B1",mount],text=True),"inodes":subprocess.check_output(["df","-i",mount],text=True)}
result["time_finished"]=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
print(json.dumps(result,ensure_ascii=False),flush=True)
'
