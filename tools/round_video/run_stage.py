"""Bounded stage in the existing GPU reservation, followed by formal training."""
import json,os,signal,subprocess,sys,time
from pathlib import Path


def main(path,gpu):
    root=Path(path);stage=json.loads((root/'stages.json').read_text())[gpu]
    out=root/f'stage-g{gpu}';out.mkdir(exist_ok=True)
    previous=out/'finished.json'
    if previous.exists():previous.rename(out/f'previous-finished-{int(time.time())}.json')
    tag=f'round-video-1117/g{gpu}'
    env={**os.environ,**stage['environment'],'APPENDIX_STAGE_ID':tag}
    p=None;result={};start=time.time()
    def stop(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    try:
        with(out/'worker.log').open('a')as f:
            p=subprocess.Popen(stage['command'],cwd=stage['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'running.json').write_text(json.dumps(dict(pid=p.pid,started=start,stage=stage['name'])))
        with(out/'gpu-memory.jsonl').open('a')as f:
            while p.poll() is None:
                if time.time()-start>stage['timeout']:raise TimeoutError('Collection deadline reached')
                r=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                f.write(json.dumps(dict(time=time.time(),reading=r.stdout.strip()))+'\n');f.flush();time.sleep(2)
        result=dict(returncode=p.returncode,seconds=time.time()-start,
                    data_complete=Path(stage['completion_file']).is_file())
    except BaseException as e:result=dict(error=repr(e),seconds=time.time()-start)
    finally:
        if p is not None and p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        sys.path.insert(0,stage['cwd'])
        from tools.appendix_signals.pipeline import cleanup
        cleanup(tag)
        (out/'finished.json').write_text(json.dumps(result,indent=2))
    # A publication error cannot hold the GPU reservation.
    try:
        publisher=root/'publish_results.py'
        if publisher.exists() and result.get('returncode') == 0 and result.get('data_complete'):
            log=(out/'publish.log').open('a')
            subprocess.Popen([sys.executable,'-B',str(publisher),gpu],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except Exception as e:(out/'publish-error.txt').write_text(repr(e))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
