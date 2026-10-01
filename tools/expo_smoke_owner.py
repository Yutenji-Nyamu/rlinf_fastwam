"""One GPU EXPO owner: flock, exact descendants, bounded fresh/resume cleanup.

Groups are diagnostic only. Signals pin registered boot/UID/PID/start with
pidfd. In borrowed mode the outer resource owner must restore the original RLT.
"""
import argparse, datetime, fcntl, hashlib, json, os, re, signal, subprocess, time, traceback, uuid
from pathlib import Path
from expo_process import pidfd_open, pidfd_send, pidfd_probe
ROOT=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001')
UUID='GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d'
PHYSICAL_GPU=4
PY='/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
UID=20001
MARKER='EXPO_SMOKE_OWNER_SCOPE'
PHASE_MARKER='EXPO_SMOKE_OWNER_PHASE'

def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':int(pid),'uid':p.stat().st_uid,'boot_id':boot_id(),
            'start_ticks':int(stat[19]),'state':stat[0],'ppid':int(stat[1]),
            'pgid':int(stat[2]),'sid':int(stat[3])}

def same(expected,current):
    return all(expected.get(k)==current.get(k) for k in ('boot_id','pid','uid','start_ticks'))

def owned(expected):
    try:
        current=identity(expected['pid'])
        return expected.get('uid')==UID and same(expected,current) and current['state'] not in ('Z','X')
    except (FileNotFoundError,ProcessLookupError):return False

def process_env(pid):
    entries=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    return {k.decode():v.decode(errors='replace') for item in entries if b'=' in item
            for k,v in [item.split(b'=',1)]}

def gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                    '--format=csv,noheader'],text=True,timeout=20)
    return [{'pid':int(line.split(',')[0]),'gpu':line.split(',')[1].strip()}
            for line in output.splitlines() if line.strip()]

def gpu_map():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],
                                   text=True,timeout=20)
    return {int(line.split(',')[0]):line.split(',')[1].strip()
            for line in output.splitlines() if line.strip()}

def rlt_check(before,paused):
    if before['boot_id']!=boot_id():raise RuntimeError('Protected RLT boot changed')
    anchors=before.get('rlt_gpu_anchors',[])
    if not anchors:raise RuntimeError('No protected RLT anchors; vacuous proof refused')
    if len({(r['pid'],r['start_ticks']) for r in anchors})!=len(anchors):
        raise RuntimeError('Duplicate protected RLT anchors')
    rows=gpu_processes();resident={(r['pid'],r['gpu']) for r in rows}
    proofs=[]
    for row in anchors:
        expected=dict(row,boot_id=before['boot_id']);alive=owned(expected)
        env_equal=False
        if alive and not paused:
            try:
                env=process_env(row['pid'])
                env_equal=all(env.get(k)==v for k,v in row.get('safe_env',{}).items())
                alive=owned(expected)
            except (FileNotFoundError,ProcessLookupError):alive=False
        proofs.append({'pid':row['pid'],'start_ticks':row['start_ticks'],'gpu':row['gpu'],
                       'alive_exact':alive,'same_gpu':(row['pid'],row['gpu']) in resident,
                       'safe_env_equal':env_equal,
                       'ok':not alive if paused else alive and env_equal and (row['pid'],row['gpu']) in resident})
    if paused:
        stop_path=ROOT/'rlt-cycle/rlt-stopped.json'
        stopped=json.loads(stop_path.read_text())
        if stopped.get('boot_id',before['boot_id'])!=boot_id():raise RuntimeError('RLT stop receipt boot differs')
        if not stopped.get('all_original_drivers_stopped') or not stopped.get('all_original_namespaces_empty'):
            raise RuntimeError('Borrowed RLT stop receipt is incomplete')
        if set(stopped.get('gpus_released',[]))!={4,5,6,7}:raise RuntimeError('RLT stop receipt GPU set differs')
        mapping=gpu_map();protected={mapping[i] for i in (4,5,6,7)}
        if not protected.issubset({row['gpu'] for row in anchors}):
            raise RuntimeError('Recorded RLT anchors do not cover all borrowed cards')
        active_owned=[]
        for row in rows:
            if row['gpu'] in protected:
                try:
                    if identity(row['pid'])['uid']==UID:active_owned.append(row)
                except (FileNotFoundError,ProcessLookupError):pass
        # Called only before launch and after exact EXPO release, not during it.
        ok=all(row['ok'] for row in proofs) and not active_owned
        return {'all_ok':ok,'mode':'borrowed_rlt_paused','anchors':proofs,
                'stop_receipt':str(stop_path),'active_owned_on_4_7':active_owned,
                'recorded_boot':before['boot_id'],'current_boot':boot_id()}
    return {'all_ok':all(row['ok'] for row in proofs),'mode':'original_rlt_live',
            'anchors':proofs,'recorded_boot':before['boot_id'],'current_boot':boot_id()}

class Roster:
    def __init__(self,root,scope,phase,path):
        self.root=root;self.scope=scope;self.phase=phase;self.path=path
        self.rows={(root['pid'],root['start_ticks']):dict(root,proof='Popen exact child/new session')}
    def write(self):
        atomic(self.path,{'scope':self.scope,'phase':self.phase,'root':self.root,
                          'registered':list(self.rows.values())})
    def scan(self):
        changed=False
        for _ in range(2):
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:r=identity(int(p.name))
                except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
                key=(r['pid'],r['start_ticks'])
                if r['state'] in ('Z','X') or key in self.rows:continue
                parents=[x for x in self.rows.values() if x['pid']==r['ppid'] and owned(x)]
                in_session=r['sid']==self.root['sid']
                if not parents and not in_session:continue
                if r['uid']!=UID:raise RuntimeError('Foreign UID in smoke descendant/session')
                if r['boot_id']!=self.root['boot_id'] or r['start_ticks']<self.root['start_ticks']:
                    raise RuntimeError('Descendant boot/start boundary failed')
                proof=None
                if parents:
                    try:fresh=identity(r['pid'])
                    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
                    if same(r,fresh) and any(owned(x) for x in parents):
                        proof='fresh exact registered parent'
                if proof is None and in_session:
                    try:env=process_env(r['pid']);fresh=identity(r['pid'])
                    # Exiting, never-registered processes may temporarily deny
                    # environ reads. They gain no signal authorization here;
                    # the bounded next scan retries and final GPU proof fails
                    # if an unprovable process remains resident.
                    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
                    if (same(r,fresh) and env.get(MARKER)==self.scope
                            and env.get(PHASE_MARKER)==self.phase):
                        proof='unique run/phase marker + original SID/start boundary'
                if proof:
                    self.rows[key]=dict(fresh,proof=proof);changed=True
        if changed:self.write()
    def alive(self):
        return [r for r in self.rows.values() if owned(r)]
    def signal(self,r,sig):
        if not owned(r):return False
        try:fd=pidfd_open(r['pid'])
        except ProcessLookupError:return False
        try:
            if not owned(r):return False
            try:pidfd_send(fd,sig)
            except ProcessLookupError:return False
            return True
        finally:os.close(fd)
    def cleanup(self):
        sent={'term':[],'kill':[]}
        for sig,seconds,label in ((signal.SIGTERM,30,'term'),(signal.SIGKILL,10,'kill')):
            deadline=time.monotonic()+seconds;delivered=set()
            while True:
                self.scan();live=self.alive()
                if not live:return {'ok':True,'signals':sent,'remaining':[]}
                for r in live:
                    key=(r['pid'],r['start_ticks'])
                    if key not in delivered:
                        if self.signal(r,sig):sent[label].append({'pid':r['pid'],'start_ticks':r['start_ticks']})
                        delivered.add(key)
                if time.monotonic()>=deadline:break
                time.sleep(.2)
        remaining=self.alive()
        if remaining:raise RuntimeError('Exact cleanup deadline: '+json.dumps(remaining))
        return {'ok':True,'signals':sent,'remaining':[]}

class OwnerTerminated(RuntimeError):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True)
    p.add_argument('--rlt-paused',action='store_true');a=p.parse_args()
    if os.getuid()!=UID:raise RuntimeError('Unexpected owner UID')
    pidfd_probe()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',a.attempt) or '..' in a.attempt:
        raise ValueError('Attempt must be a simple unique label')
    lockdir=ROOT/'locks';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'expo-smoke-owner.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another EXPO owner holds the task lock')
        owner=ROOT/'runs'/a.attempt;owner.mkdir(parents=True,exist_ok=False)
        scope=a.attempt+'-'+str(uuid.uuid4())
        meta={'owner':identity(os.getpid()),'boot_id':boot_id(),'scope':scope,
              'physical_gpu':PHYSICAL_GPU,'gpu_uuid':UUID,'status':'PREPARING',
              'rlt_pause_required':a.rlt_paused,'rlt_borrowed':a.rlt_paused,
              'rlt_restore_required_by_outer_owner':a.rlt_paused}
        atomic(owner/'owner.json',meta)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,
            PYTHONPATH=str(ROOT/'source')+':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
            LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64',
            VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
        env[MARKER]=scope
        active=None;child=None;error=None;before=None;receipts=[];cleanups=[];rlt_checks=[]
        signal_state={'closing':False,'starting':False,'requested':None}
        def terminate(sig,_frame):
            signal_state['requested']=sig
            if not signal_state['closing'] and not signal_state['starting']:
                raise OwnerTerminated('Owner received signal '+str(sig))
        previous_term=signal.signal(signal.SIGTERM,terminate)
        previous_int=signal.signal(signal.SIGINT,terminate)
        try:
            mapping=gpu_map()
            if mapping.get(PHYSICAL_GPU)!=UUID:raise RuntimeError('Physical GPU/UUID metadata differs')
            if a.rlt_paused and PHYSICAL_GPU not in (4,5,6,7):
                raise RuntimeError('Borrowed run must use authorized physical4-7')
            before=json.loads((ROOT/'rlt-before.json').read_text())
            if hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest()!=before['inputs_sha256']:
                raise RuntimeError('Inputs changed after protected RLT preparation')
            check=rlt_check(before,a.rlt_paused);rlt_checks.append(check)
            atomic(owner/'rlt-before-verified.json',check)
            if not check['all_ok']:raise RuntimeError('Protected RLT before-launch proof failed')
            if any(r['gpu']==UUID for r in gpu_processes()):raise RuntimeError('Reserved GPU has another compute owner')
            checkpoint=None
            for phase in ('fresh','resume'):
                target=owner/phase;target.mkdir()
                command=[PY,'-u','-B',str(ROOT/'source/examples/embodiment/train_expo_ft.py'),
                    '--inputs',str(ROOT/'inputs.json'),'--run',str(target),
                    '--episodes','1','--batch-size','4','--candidate-microbatch','4']
                if checkpoint:command+=['--resume',checkpoint]
                phase_env=dict(env,**{PHASE_MARKER:phase})
                with (target/'command.log').open('wb') as log:
                    # Delay raising a requested termination only across Popen /
                    # identity registration. Do not block SIGTERM in the child.
                    signal_state['starting']=True
                    try:
                        child=subprocess.Popen(command,cwd=ROOT/'source',env=phase_env,stdout=log,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                        ident=identity(child.pid)
                        active=Roster(ident,scope,phase,target/'process-roster.json')
                    finally:signal_state['starting']=False
                    active.write()
                    if signal_state['requested'] is not None:
                        raise OwnerTerminated('Termination requested during child registration')
                    if ident['sid']!=child.pid or ident['ppid']!=os.getpid():
                        raise RuntimeError('Popen child session/parent differs')
                    meta.update(status='RUNNING_'+phase.upper(),child=ident,command=command)
                    atomic(owner/'current.json',meta);started=time.monotonic()
                    while child.poll() is None:
                        active.scan()
                        if time.monotonic()-started>5400:raise TimeoutError('Smoke exceeded 90min diagnosis boundary')
                        time.sleep(.5)
                    active.scan();rc=child.returncode
                    cleaned=active.cleanup();child.wait(timeout=5)
                    cleanups.append(dict(cleaned,phase=phase));atomic(target/'cleanup.json',cleaned)
                    active=None
                if rc!=0:raise RuntimeError(f'{phase} driver exited {rc}; inspect {target}/command.log')
                complete=json.loads((target/'complete.json').read_text())
                if not complete['ok'] or complete['real_update_calls']!=1 or complete['base_updates']<1:
                    raise RuntimeError('Incomplete real learning receipt')
                if phase=='resume' and not complete['resume_verified']:raise RuntimeError('New-process resume not verified')
                checkpoint=complete['checkpoint'];receipts.append(complete)
                atomic(owner/(phase+'-verified.json'),complete)
                if any(r['gpu']==UUID for r in gpu_processes()):raise RuntimeError('Reserved GPU not released')
                check=rlt_check(before,a.rlt_paused);rlt_checks.append(check)
                atomic(target/'rlt-verified.json',check)
                if not check['all_ok']:raise RuntimeError('Protected RLT phase-boundary proof failed')
        except BaseException as failure:
            error={'type':type(failure).__name__,'message':str(failure),'traceback':traceback.format_exc()}
        finally:
            signal_state['closing']=True
            if active is not None:
                try:
                    cleaned=active.cleanup();cleanups.append(dict(cleaned,phase=active.phase))
                    atomic(active.path.parent/'cleanup.json',cleaned)
                    if child is not None:child.wait(timeout=5)
                except BaseException as failure:error={'type':'CLEANUP_FAILED','message':str(failure),'previous':error}
            final_rlt={'all_ok':False,'anchors':[],'error':'No pre-smoke RLT receipt'}
            if before is not None:
                try:final_rlt=rlt_check(before,a.rlt_paused);rlt_checks.append(final_rlt)
                except BaseException as failure:final_rlt={'all_ok':False,'anchors':[],'error':str(failure)}
            try:released=not any(r['gpu']==UUID for r in gpu_processes());gpu_error=None
            except BaseException as failure:released=False;gpu_error=str(failure)
            if signal_state['requested'] is not None and error is None:
                error={'type':'OwnerTerminated','message':'Termination requested during closeout'}
            rlt_ok=bool(rlt_checks) and all(r['all_ok'] for r in rlt_checks) and final_rlt['all_ok']
            passed=error is None and len(receipts)==2 and released and rlt_ok
            if not passed and error is None:
                error={'type':'CLOSEOUT_VERIFICATION_FAILED','message':'Learning/release/RLT boundary proof failed'}
            final={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                'owner':meta['owner'],'scope':scope,'error':error,'receipts':receipts,'cleanups':cleanups,
                'gpu3_released':released if PHYSICAL_GPU==3 else None,
                'reserved_gpu_released':released,'physical_gpu':PHYSICAL_GPU,'gpu_uuid':UUID,
                'gpu_release_error':gpu_error,'rlt_anchors':final_rlt['anchors'],'rlt_checks':rlt_checks,
                'rlt_borrowed':a.rlt_paused,'rlt_boundary_verified':rlt_ok,
                'rlt_unchanged_all_anchors':rlt_ok if not a.rlt_paused else None,
                'rlt_restore_status':'OUTER_OWNER_REQUIRED' if a.rlt_paused else 'NOT_BORROWED',
                'smoke_passed_scope':'learning_and_owned_gpu_release; borrowed_RLT_restore_external' if a.rlt_paused else 'learning_release_and_RLT_continuity',
                'smoke_passed':passed}
            atomic(owner/'final.json',final);meta.update(status='PASSED' if passed else 'FAILED')
            atomic(owner/'current.json',meta);print(json.dumps(final),flush=True)
            signal.signal(signal.SIGTERM,previous_term);signal.signal(signal.SIGINT,previous_int)
        raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
