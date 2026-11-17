"""Bounded evaluation stage; completion or failure returns to the queued trainer."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);args=ap.parse_args()
    job_path=Path(args.job);job=json.loads(job_path.read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,**job['environment']);deadline=time.monotonic()+job.get('deadline_s',21600)
    child=None;result={}
    def interrupted(*_): raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for stage in job.get('before', []):
            with (out / (stage['name'] + '.log')).open('a') as log:
                child = subprocess.Popen(stage['command'], cwd=stage.get('cwd',job['source']),
                    env={**env,**stage.get('environment',{})}, stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=stage.get('timeout',900))
            except subprocess.TimeoutExpired:
                stop(child);code=124
            write(out/(stage['name']+'-result.json'),dict(returncode=code,time=time.time()))
            child=None
        with(out/'worker.log').open('a')as log:
            child=subprocess.Popen([sys.executable,'-u','-B','-m','tools.teacher_eval.evaluate','--job',str(job_path)],
                 cwd=job['source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'worker.json',dict(pid=child.pid,time=time.time()))
        while child.poll() is None:
            if time.monotonic()>deadline: raise TimeoutError('Evaluation deadline reached; resume formal training')
            time.sleep(2)
        result=dict(returncode=child.returncode,complete=(out/'complete.json').exists(),time=time.time())
    except BaseException as exc:
        result=dict(complete=False,error=repr(exc),time=time.time())
    finally:
        if child is not None:stop(child)
        write(out/'lane-result.json',result)
    # The continuation is unconditional: a failed collection must not strand a card.
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

