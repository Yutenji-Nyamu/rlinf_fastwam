import json,os,subprocess,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');p=json.loads((S/'bc/plan.json').read_text());R=Path(p['repo'])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R))
files=['tests/unit_tests/'+x for x in ('test_online_bc_ugrow.py','test_online_bc.py','test_online_bc_dvac.py','test_online_bc_success_limit.py')]
q=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*files],cwd=R,env=env,text=True,capture_output=True,timeout=180)
result={'rc':q.returncode,'stdout':q.stdout[-7000:],'stderr':q.stderr[-2000:]};assert q.returncode==0,result
code='''import tempfile,torch
from pathlib import Path
from rlinf.data.online_bc import SuccessReplay
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC as S
with tempfile.TemporaryDirectory() as tmp:
 r={'ugrow_u':torch.linspace(.1,.5,50),'action_valid_mask':torch.ones(50,14),'action_weights':torch.linspace(.2,2,50),'ugrow_signal_version':torch.tensor(S['version']),'ugrow_steps':torch.tensor(S['steps']),'ugrow_action_dim':torch.tensor(S['action_dim']),'ugrow_epsilon':torch.tensor(S['epsilon'],dtype=torch.float64)}
 p=SuccessReplay(seed=42,archive_path=tmp+'/pool',max_success_chunks=3,signal_spec=dict(S))
 p.add_episodes([[r],[r,r,r],[r,r,r,r]])
 assert p.episodes==2 and len(p)==4 and p.filtered_success_episodes==1
 p.save_checkpoint(tmp+'/cp');a=p.sample(3)['forward_inputs']
 q=SuccessReplay(seed=42,archive_path=tmp+'/restored',max_success_chunks=3,signal_spec=dict(S));q.load_checkpoint(tmp+'/cp');b=q.sample(3)['forward_inputs']
 assert q.episodes==2 and q.filtered_success_episodes==1 and all(torch.equal(a[k],b[k]) for k in a)
 print('U length<=3 admission and complete record/weight/RNG roundtrip: PASS')
'''
q=subprocess.run([sys.executable,'-B','-c',code],cwd=R,env=env,text=True,capture_output=True,timeout=120)
result['combined']={'rc':q.returncode,'stdout':q.stdout[-1500:],'stderr':q.stderr[-1500:]};assert q.returncode==0,result
(S/'cpu-tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
