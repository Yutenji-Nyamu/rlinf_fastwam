"""CPU dry run of the deferred winner/calibration/config generation path."""
import importlib.util,json,os,tempfile
from pathlib import Path
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-noise-early-c20-20261019')
spec=importlib.util.spec_from_file_location('followup_prepare',R/'tools/pi05_followups/prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='future-c20-',dir='/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-upload-v1')as tmp:
 m.D=Path(tmp)/'transaction';m.D.mkdir();m.BASE=Path(tmp)/'results'
 rank=m.BASE/'pi05-policy-eval/pillbottle-rank';target=m.BASE/'pi05-policy-eval/pillbottle-c20'
 m.ROOTS=dict(m.ROOTS,rank=rank,c20=target)
 for name,k in [('c20_weak',68)]+[(f'c20_{n}',k)for n,k in zip(m.CANDIDATES,[90,94,92,93,70,65])]:
  m.save(rank/f'summaries/{name}.json',{'condition':name,'episodes':128,'successes':k,'success_rate':k/128})
  m.save(rank/f'summaries/{name}-episodes.json',[])
 b=rank/'batches/calibration_fixture';m.save(b/'done.json',{})
 c=m.read(m.OLD/'configs/grpo_u_tau1_cp180_b0.json');c['execute_chunk']=20;m.save(b/'config.json',c)
 m.save(b/'decisions.json',[dict(slot=0,actual_seed=7,query=q,start=20*q,end=20*q+20,
             noise_seed=40+q,u=.01+q*.002,strong=False)for q in range(10)])
 m.prepare_c20()
 selection=m.read(m.D/'c20-selection.json');assert selection['selected']=='grpo_u_tau2_cp160'
 for g in (4,5):
  plan=m.read(m.D/'c20-sweep'/f'lane-g{g}-plan.json');assert len(plan['stages'])==30
  for stage in plan['stages']:
   assert len(stage['configs'])==2
   for path in stage['configs']:
    c=m.read(path);assert c['execute_chunk']==20 and c['max_takeover_queries']is None and c['gpu']==g
    assert 'grpo_u_tau2_cp160'in c['strong_weights'];m.validate_config(c)
 print(json.dumps({'passed':True,'selected':'grpo_u_tau2_cp160','configs':120,'all_c20_full_episode':True}))
