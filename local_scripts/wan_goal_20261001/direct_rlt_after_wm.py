"""Mid-run route change: ask the existing owner to skip Dojo work and return RLT.

This CPU guard never starts training or invokes RLT resume. The frozen v5 owner
only supports SIGTERM outside WM, so wait for verified WM release/EVALUATING,
then interrupt its short Dojo launcher. That owner performs the sole cleanup and
RLT restoration. SIGTERM is never sent while formal WM training is running.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

R=Path('/data/chenyiteng/projects/wan-goal-sz3')
P=Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
D=P/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
A=D/'continuation-20261001-wan-goal-v5'
W=R/'runs/wan-goal-sz3-20261001-r5'
G=R/'post-wm-direct-rlt-v5'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,alive,atomic,exclusive,identity,pidfd_open,pidfd_send,read,sha

def exact_owner(expected):
    current=read(D/'active-continuation.json')
    assert current['attempt_dir']==str(A) and current['cycle_dir']==expected['cycle_dir']
    for key in ('pid','uid','start','boot','command_sha256'):assert current[key]==expected[key],key
    return current

def state(phase,**details):
    record={'time':time.time(),'phase':phase,**details}
    atomic(G/'current.json',record)
    with (G/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')

def launch():
    account();expected=read(D/'active-continuation.json')
    assert expected['attempt_dir']==str(A) and alive(expected,command=True)
    assert read(D/'pipeline-current.json')['phase']=='RUNNING_WM'
    assert read(W/'sequence-current.json')['wm_run']==str(W/'pi05-formal')
    assert not G.exists();G.mkdir(mode=0o700)
    request={'time':time.time(),'after_wm':'RLT_DIRECT','resume_dojo':False,
             'owner':expected,'source_sha256':sha(Path(__file__)),
             'authority':'User: WM结束后直接归还RLT，其他不稳定；继续当前WM'}
    exclusive(G/'request.json',request)
    argv=[sys.executable,'-u','-B',str(Path(__file__))]
    exclusive(G/'launch-intent.json',{'time':time.time(),'argv':argv,'owner':expected})
    env=os.environ.copy();env.pop('WAN_GOAL_OWNER_TOKEN',None)
    with (G/'guard.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=R,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        row=identity(child.pid)
    exclusive(G/'guard-identity.json',row)
    print(json.dumps({'request':str(G/'request.json'),'guard_identity':row,'route':'WM -> original four RLT','training_interrupted':False}))

def guard():
    account();request=read(G/'request.json');expected=request['owner']
    assert request['after_wm']=='RLT_DIRECT' and request['resume_dojo'] is False
    assert request['source_sha256']==sha(Path(__file__))
    sent=False;state('ARMED_WM_THEN_RLT',owner=expected,training_action='none')
    try:
        while True:
            current=exact_owner(expected)
            final=A/'pipeline-final.json'
            if not alive(current,command=True):
                result=read(final) if final.is_file() else None
                if result and result['rlt_dispatched'] and result['wm_released'] and not result['error']:
                    state('RLT_RETURN_DISPATCHED',owner_final=str(final),restore_owned_by='existing v5 owner')
                    return
                raise RuntimeError('Owner ended without verified unique RLT restoration; inspect, never launch another restorer')
            phase=read(D/'pipeline-current.json')
            assert phase['continuation']==str(A)
            if phase['phase']=='EVALUATING' and not sent:
                release=read(A/'wm-release.json')
                assert release['all_workers_stopped'] and release['processes_clear'] and release['gpus_released']
                assert release['physical_gpus']==[4,5,6,7]
                intent=A/'skip-dojo-return-rlt-after-wm-20261001.json'
                fd=pidfd_open(current['pid'])
                try:
                    fresh=identity(current['pid'])
                    assert all(fresh[k]==current[k] for k in ('pid','uid','start','boot','command_sha256'))
                    exclusive(intent,{'time':time.time(),'action':'SIGTERM_EXISTING_OWNER_AFTER_VERIFIED_WM_RELEASE',
                        'identity':fresh,'user_route':'WM -> original four RLT','wm_release_sha256':sha(A/'wm-release.json')})
                    pidfd_send(fd,signal.SIGTERM);sent=True
                finally:os.close(fd)
                state('RETURN_REQUESTED_AFTER_WM_RELEASE',intent=str(intent),training_action='none')
            # Poll rapidly only during the final cleanup/return boundary.
            approaching=(W/'pi05-formal/wm-exit.json').is_file() or sent
            time.sleep(.1 if approaching else 5)
    except Exception as error:
        state('DIRECT_RETURN_NEEDS_ATTENTION',error=repr(error),training_action='none; no competing restoration')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    (launch if parser.parse_args().launch else guard)()
