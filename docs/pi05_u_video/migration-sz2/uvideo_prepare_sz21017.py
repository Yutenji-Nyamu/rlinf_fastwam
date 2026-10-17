import json,os,sys,time,tarfile,shutil,subprocess
from pathlib import Path
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-u-full-video-20261017');O=Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same
assert os.getuid()==20001 and R.is_dir() and not any(R.iterdir()) and not (D/'prepared.json').exists()
plan=read(S/'plan.json');assert all(same(plan['slots'][g]['external_identity'])for g in ['4','5','6','7'])
assert plan['slots']['4']['external_identity']==plan['slots']['5']['external_identity']
assert 'grpo-u-tau2-notrick-pillbottle256' in plan['slots']['4']['external_release']
O.mkdir(parents=True,exist_ok=True);(O/'batches').mkdir(exist_ok=True);(D/'bootstrap').mkdir(exist_ok=True)
for a,dest in [('/data/chenyiteng/u-video-runtime1017.tar.gz',R),('/data/chenyiteng/u-video-migration-configs1017.tar.gz',D/'original-configs')]:
 dest.mkdir(exist_ok=True)
 with tarfile.open(a)as t:
  assert all(not m.name.startswith('/') and '..'not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in t.getmembers());t.extractall(dest)
shutil.copyfile('/data/chenyiteng/u-video-sitecustomize1017.py',D/'bootstrap/sitecustomize.py')
old=D/'original-configs';(O/'configs').mkdir();jobs={4:[],5:[]}
for p in (old/'configs').glob('*.json'):
 if p.name.endswith('_seeds.json'):shutil.copyfile(p,O/'configs'/p.name);continue
 c=read(p);g=c['batch']+4;out=O/'batches'/p.stem;c.update(gpu=g,output=str(out),source_commit='30ee66c09',acceptance=False,migrated_from='sz1 6/7 planned run')
 c['env']['seeds_path']=str(O/'configs'/Path(c['env']['seeds_path']).name);c['env']['task_config']['save_path']=str(out/'env');c['env']['video_cfg']['video_base_dir']=str(out/'unused')
 assert Path(c['model']['model_path']).is_dir() and Path(c['env']['assets_path']).is_dir()
 dest=O/'configs'/p.name;save(dest,c);jobs[g].append(dict(config=str(dest),task=c['task'],batch=c['batch']))
assert all(len(jobs[g])==50 for g in jobs)
for g in [4,5]:
 jobs[g].sort(key=lambda j:(j['task']!=('turn_switch'if g==4 else'adjust_bottle'),j['task']))
 env=read(old/f'environment-g{g+2}.json');env['PYTHONPATH']=str(D/'bootstrap')+':'+str(R)+':'+env['ROBOTWIN_PATH'];env.update(CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g),TMPDIR=str(D/f'tmp-g{g}'),HOME='/home/chenyiteng',NO_PROXY='127.0.0.1,localhost,h100-gpu02',no_proxy='127.0.0.1,localhost,h100-gpu02',VK_DRIVER_FILES='/usr/share/vulkan/icd.d/nvidia_icd.json',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
 Path(env['TMPDIR']).mkdir();assert Path(env['VK_DRIVER_FILES']).is_file();save(O/f'environment-g{g}.json',env)
 save(D/f'lane-g{g}-plan.json',dict(gpu=g,jobs=jobs[g],predecessor=plan['slots'][str(g)]))
m=read(old/'manifest.json');m.update(time=time.time(),source=str(R),output=str(O),source_commit='30ee66c09',physical_gpus=[4,5],plan_before=plan,old_run=str(Path(plan['slots']['4']['external_release']).parent.parent),old_identity=plan['slots']['4']['external_identity'],old_release=plan['slots']['4']['external_release'],order='GRPO tau2 authorized closeout -> U/video -> original per-card RLT',host='sz2');save(D/'prepared.json',m);save(O/'manifest.json',m)
env=read(O/'environment-g4.json');env['CUDA_VISIBLE_DEVICES']='';env.pop('DOJO_GPU_SCOPE',None)
subprocess.run([plan['python'],'-B',str(R/'tools/pi05_u_video/test_recording.py')],env=env,cwd=R,check=True)
save(D/'preflight.json',dict(time=time.time(),passed=True,cpu_controller_test=True,config_count=100,tasks=50,parallelism_per_card=16,source_commit='30ee66c09',real_video_acceptance='pending first B16 batches'))
print(json.dumps(read(D/'preflight.json')))
