"""Run prepared finite collection stages, then return to the original trainer."""
import json,os,sys,time,signal,subprocess
from pathlib import Path

def main(planfile):
    plan=json.loads(Path(planfile).read_text());root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,plan['cleanup_source']);from tools.appendix_signals.pipeline import cleanup
    def stop(*args):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    for stage in plan['stages']:
        tag=f"{plan['id']}/{stage['name']}";start=time.time();p=None;record={}
        try:
            with(root/f"{stage['name']}.log").open('a')as f:
                p=subprocess.Popen(stage['command'],cwd=stage['cwd'],env={**os.environ,**stage['environment'],'APPENDIX_STAGE_ID':tag},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            (root/'status.json').write_text(json.dumps(dict(stage=stage['name'],pid=p.pid,started=start)))
            with(root/f"{stage['name']}-memory.jsonl").open('a')as f:
                while p.poll() is None:
                    if time.time()-start>stage['timeout']:raise TimeoutError(stage['name'])
                    r=subprocess.run(['nvidia-smi','-i',str(plan['gpu']),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);f.write(json.dumps(dict(time=time.time(),reading=r.stdout.strip()))+'\n');f.flush();time.sleep(3)
            record=dict(returncode=p.returncode,complete=Path(stage['complete']).exists(),seconds=time.time()-start)
        except Exception as e:record=dict(error=repr(e),seconds=time.time()-start)
        finally:
            cleanup(tag)
            if p is not None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            (root/f"{stage['name']}-receipt.json").write_text(json.dumps(record,indent=2))
        if 'publish' in stage:
            with(root/f"{stage['name']}-publish.log").open('a')as f:subprocess.Popen(stage['publish'],cwd=stage['cwd'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'complete.json').write_text(json.dumps(dict(time=time.time(),next='original formal training')))
if __name__=='__main__':main(sys.argv[1])
