"""One-shot recovery for the prelaunch snapshot; change only verified ctime pins."""
import ast,copy,hashlib,os,socket,sys,time
from common import *
from migrate import PAYLOAD
sys.path.insert(0,str(SOURCE/'tools'))
from expo_smoke_owner import owned

def main():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    previous=read(CONTROL/'owner.json')['owner']
    assert not owned(previous) and read(CONTROL/'current.json')['status']=='FAILED_BEFORE_BORROW'
    assert not (CONTROL/'rlt-stop-intent.json').exists()
    assert not (CONTROL/'replay-ctime-repair.json').exists()
    for stage in QUEUES:
        assert read(stage/'queue-status.json')['roles']=={'clean':'TRAINING','combo':'TRAINING'}
        assert owned(read(CONTROL/'staged.json')['queues'][str(stage)]['owner'])
    migration=read(CONTROL/'migration.json'); assert migration['ok']
    run=TRAIN/'run'; replay=TRAIN/'replay'; backup=CONTROL/'baseline-20000'
    old=read(replay/'index.json'); new=copy.deepcopy(old); checked=[]
    assert old==read(backup/'replay/index.json')
    for entry in new['online_entries']:
        for name,key in [('path','pin'),('manifest_path','manifest_pin')]:
            path=replay/entry[name]; pin=entry[key]; st=path.stat()
            now=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_uid]
            assert now[:4]+now[5:]==pin['identity'][:4]+pin['identity'][5:]
            assert sha(path)==pin['sha256'] and st.st_size==pin['bytes']
            if now[4]!=pin['identity'][4]:
                assert name=='path' and path.suffix=='.pt'
                linked=backup/'replay'/entry[name]
                assert os.path.samefile(path,linked) and st.st_nlink==2
                checked.append({'path':entry[name],'before_ctime_ns':pin['identity'][4],'after_ctime_ns':now[4],'sha256':pin['sha256']})
                pin['identity'][4]=now[4]
    assert len(checked)==128
    import torch
    torch.set_num_threads(4)
    tree=ast.parse((SOURCE/'examples/embodiment/train_expo_ft.py').read_text())
    node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='digest']
    ns={'torch':torch,'hashlib':hashlib};exec(compile(ast.Module(body=[node],type_ignores=[]),'digest','exec'),ns);digest=ns['digest']
    results={}
    for name in ('checkpoint-latest.pt','checkpoint-last1.pt'):
        saved=torch.load(run/name,map_location='cpu',weights_only=False)
        before={k:digest(saved[k]) for k in PAYLOAD};assert before==saved['hashes']
        assert saved['replay']['online_entries']==old['online_entries']
        previous_replay=copy.deepcopy(saved['replay'])
        saved['replay']['online_entries']=copy.deepcopy(new['online_entries'])
        expected=copy.deepcopy(previous_replay);expected['online_entries']=new['online_entries']
        assert saved['replay']==expected
        after=dict(before,replay=digest(saved['replay']));saved['hashes']=after
        target=CONTROL/(name+'.ctime-repaired')
        with target.open('xb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
        del saved
        verified=torch.load(target,map_location='cpu',weights_only=False)
        assert {k:digest(verified[k]) for k in PAYLOAD}==after
        assert all(before[k]==after[k] for k in PAYLOAD-{'replay'})
        replay_state=verified['replay']
        results[name]={'before':before,'after':after,'bytes':target.stat().st_size};del verified
    atomic(replay/'index.json',new)
    for name in results:(CONTROL/(name+'.ctime-repaired')).replace(run/name)
    cp=read(run/'checkpoint.json');cp.update(hashes=results['checkpoint-latest.pt']['after'],bytes=results['checkpoint-latest.pt']['bytes']);atomic(run/'checkpoint.json',cp)
    atomic(CONTROL/'migration-before-ctime-repair.json',migration)
    for name,row in results.items():migration['checkpoint_results'][name].update(after=row['after'],bytes=row['bytes'])
    migration['unchanged_payloads']=['base','core','rng']
    migration['replay_payloads_unchanged']=True
    migration['replay_metadata_change']='Only 128 verified online artifact ctime identity fields after hardlink snapshot'
    migration['replay_index_sha256']=sha(replay/'index.json');atomic(CONTROL/'migration.json',migration)
    import importlib.util
    spec=importlib.util.spec_from_file_location('formal_replay_check',SOURCE/'rlinf/algorithms/expo_ft/formal_replay.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    inputs=read(TRAIN/'inputs.json')
    restored=mod.FormalReplay(root=replay.resolve(),demo_path=inputs['demo_path'],seed=inputs['formal']['seed'])
    restored.load_state_dict(replay_state)
    assert digest(restored.state_dict())==results['checkpoint-latest.pt']['after']['replay']
    result={'ok':True,'time':time.time(),'previous_owner':previous,'rlt_untouched':True,'payloads_verified':256,
            'ctime_only_changes':checked,'checkpoint_results':results,'strict_replay_restore_verified':True,'cpu_only':not torch.cuda.is_initialized()}
    atomic(CONTROL/'replay-ctime-repair.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('ctime_only_changes','checkpoint_results')}))

if __name__=='__main__':main()
