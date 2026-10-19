"""Prepare the user-authorized SZ1 4/5 sequence from accepted exact configs."""
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'tools/pi05_policy_eval'))
from protocol import read,save,validate_config
from threshold_sweep import conditions
from calibration_pool import collect,deduplicate

D=Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v1')
BASE=Path('/data/chenyiteng/results')
OLD=BASE/'pi05-policy-eval/pillbottle-strong-c20-20261017-v3'
UV=BASE/'pi05-u-video/20261017-v1'
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
CARDS=(4,5)
CANDIDATES=('grpo_u_tau1_cp180','grpo_u_tau2_cp160','grpo_u_tau2_cp150',
            'grpo_u_tau3_cp70','clean_c17_cp140','clean_c15_cp160')
ROOTS={
 'noise':BASE/'pi05-noise-video/20261019-sz1-v1',
 'smoke':BASE/'pi05-policy-eval/pillbottle-followup-smoke-20261019-v1',
 'early':BASE/'pi05-policy-eval/pillbottle-early2-20261019-v1',
 'rank':BASE/'pi05-policy-eval/pillbottle-c20-rank-20261019-v1',
 'c20':BASE/'pi05-policy-eval/pillbottle-c20-sweep-20261019-v1',
}


def environment(root,g):
    env=read(OLD/f'environment-g{g+2}.json');old=env['REPO_PATH']
    env={k:v.replace(old,str(R)) if isinstance(v,str) else v for k,v in env.items()}
    env.update(REPO_PATH=str(R),CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g),
               TMPDIR=str(D/f'tmp-g{g}'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    env.pop('RAY_ADDRESS',None)
    save(root/f'environment-g{g}.json',env)


def dirs(root):
    assert not root.exists(),f'Output already exists: {root}'
    for n in ('configs','thresholds','batches','summaries'):(root/n).mkdir(parents=True)
    for g in CARDS:environment(root,g)


def policy_config(root,name,batch,policy,chunk,weights=None,limit=None,threshold=None):
    c=copy.deepcopy(read(OLD/'configs'/f'grpo_u_tau1_cp180_b{batch}.json'))
    g=4+batch%2;out=root/'batches'/f'{name}_b{batch}'
    seedfile=root/'configs'/f'seeds_b{batch}.json'
    if not seedfile.exists():shutil.copy2(c['env']['seeds_path'],seedfile)
    c.update(condition=name,policy=policy,execute_chunk=chunk,gpu=g,output=str(out),
             source_commit=(R/'SOURCE_REVISION').read_text().strip(),takeover_mode='weights',
             record_full_video=True,max_takeover_queries=limit,
             comparison_baseline='early_weak' if chunk==50 else 'c20_weak',
             collect_weak_u=policy=='strong' and chunk==20)
    if weights:c['strong_weights']=str(weights)
    else:c.pop('strong_weights',None)
    if threshold is not None:
        tp=root/'thresholds'/f'{name}.json';save(tp,threshold);c['threshold_path']=str(tp)
    c['env']['seeds_path']=str(seedfile)
    c['env']['task_config']['save_path']=str(out/'env')
    c['env']['video_cfg'].update(save_video=False,video_base_dir=str(out/'unused'))
    validate_config(c);p=root/'configs'/f'{name}_b{batch}.json';save(p,c)
    return str(p)


def make_stages(root,transaction,specs):
    transaction.mkdir(parents=True,exist_ok=True)
    for g in CARDS:
        stages=[]
        for row in specs:
            configs=[policy_config(root,row['condition'],b,row['policy'],row['chunk'],
                     row.get('weights'),row.get('limit'),row.get('threshold')) for b in (g-4,g-2)]
            stages.append({'condition':row['condition'],'configs':configs})
        save(transaction/f'lane-g{g}-plan.json',{'gpu':g,'stages':stages})


def prepare(pool_path):
    assert os.getuid()==1003 and not(D/'prepared.json').exists()
    D.mkdir(parents=True,exist_ok=True)
    plan=read(S/'plan.json')
    for g in ('4','5'):
        assert 'expo-clean-entropy-b64-sz1-g45-60000-1015' in plan['slots'][g]['external_release']
    pool=read(pool_path);assert pool['unique_decisions']>0
    save(D/'early-calibration.json',pool)
    for root in list(ROOTS.values())[:-1]:dirs(root)
    original=read(UV/'manifest.json')
    paths=sorted((UV/'configs').glob('*.json'))
    configs=[read(p)for p in paths if not p.name.endswith('_seeds.json')]
    assert len(configs)==100 and len({c['task']for c in configs})==50
    jobs={g:[] for g in CARDS};O=ROOTS['noise']
    for c in configs:
        g=4+c['batch'];name=f"{c['task']}_b{c['batch']}";out=O/'batches'/name
        seedfile=O/'configs'/f'{name}_seeds.json';shutil.copy2(c['env']['seeds_path'],seedfile)
        c.update(gpu=g,output=str(out),noise_samples=4,acceptance=False,
                 source_commit=(R/'SOURCE_REVISION').read_text().strip(),
                 signal='U10/5 plus four-noise N10 population std/RMS; executed main unchanged')
        c['env']['seeds_path']=str(seedfile);c['env']['task_config']['save_path']=str(out/'env')
        c['env']['video_cfg'].update(save_video=False,video_base_dir=str(out/'unused'))
        p=O/'configs'/f'{name}.json';save(p,c);jobs[g].append({'config':str(p),'task':c['task'],'batch':c['batch']})
    save(O/'manifest.json',dict(tasks=50,episodes=1600,batches=100,episodes_per_task=32,
         per_card_parallelism=16,physical_gpus=list(CARDS),plan_before=plan,source=str(R),
         original=str(UV),noise_samples=4,extra_inference_batch=48,steps_per_sample=10))
    for g in CARDS:
        first='turn_switch' if g==4 else 'adjust_bottle';jobs[g].sort(key=lambda x:(x['task']!=first,x['task']))
        for label,selected in [('noise-smoke',jobs[g][:1]),('noise-full',jobs[g][1:])]:
            save(D/label/f'lane-g{g}-plan.json',{'gpu':g,'jobs':selected,'predecessor':plan['slots'][str(g)]})
    cp=Path('/data/chenyiteng/models/protected-eval/pillbottle/grpo_u_tau1_cp180/full_weights.pt')
    rows=conditions(pool['rows']);specs=[]
    for row in rows:
        name=row['condition'].replace('cp180_','early2_')
        specs.append(dict(condition=name,policy=row['policy'],chunk=50,weights=cp,limit=2,
                          threshold={row['policy']:row['threshold'],'calibration_target':row['target_fraction']}))
    specs=[dict(condition='early_weak',policy='weak',chunk=50),
           dict(condition='early_all',policy='high',chunk=50,weights=cp,limit=2,threshold={'high':-1}),
           dict(condition='early_random',policy='random',chunk=50,weights=cp,limit=2)]+specs
    make_stages(ROOTS['early'],D/'early',specs)
    save(ROOTS['early']/'frozen-thresholds.json',{'rows':rows,'pool':pool,'limit_queries':2})
    rank=[dict(condition='c20_weak',policy='calibration',chunk=20)]
    for name in CANDIDATES:
        weights=Path('/data/chenyiteng/models/protected-eval/pillbottle')/name/'full_weights.pt'
        assert weights.is_file();rank.append(dict(condition='c20_'+name,policy='strong',chunk=20,weights=weights))
    make_stages(ROOTS['rank'],D/'c20-rank',rank)
    median=sorted(r['u']for r in pool['rows'])[len(pool['rows'])//2]
    for g in CARDS:
        p=policy_config(ROOTS['smoke'],'smoke_early' if g==4 else 'smoke_c20',g-4,'high',
                        50 if g==4 else 20,cp,2 if g==4 else None,{'high':median})
        save(D/'switch-smoke'/f'lane-g{g}-plan.json',{'gpu':g,'stages':[{'condition':'smoke','configs':[p],'summarize':False}]})
    q=dict(time=time.time(),source=str(R),transaction=str(D),roots={k:str(v)for k,v in ROOTS.items()},
           plan_before=plan,physical_gpus=list(CARDS),uid=1003,candidates=list(CANDIDATES),
           phase_order=['noise-smoke','switch-smoke','noise-full','early','c20-rank','c20-sweep','RLT'],
           early_conditions=len(specs),rank_conditions=len(rank),c20_sweep_conditions=30)
    save(D/'prepared.json',q);print(json.dumps({k:v for k,v in q.items()if k!='plan_before'}))


def prepare_c20():
    rank=ROOTS['rank'];weak=read(rank/'summaries/c20_weak.json')
    results=[read(rank/f'summaries/c20_{name}.json')for name in CANDIDATES]
    assert all(r['episodes']==128 for r in results+[weak])
    best=max(results,key=lambda r:r['successes'])
    selected=best['condition'].removeprefix('c20_')
    selection={'time':time.time(),'weak':weak,'candidates':results,'selected':selected,
               'tie_breaker':'predeclared candidate order','same_128_seeds_reused':True,
               'selection_is_exploratory':True}
    save(D/'c20-selection.json',selection)
    if best['successes']<=weak['successes']:
        save(D/'c20-not-stronger.json',{'time':time.time(),'reason':'No RL candidate beats original C20 on the paired128; skip dependent takeover sweep.'})
        return
    historical=list((BASE/'pi05-policy-eval').glob('pillbottle-*'))
    pool=deduplicate([collect(historical,20)])
    assert pool['unique_decisions']>0
    O=ROOTS['c20'];dirs(O)
    for suffix in ('.json','-episodes.json'):
        shutil.copy2(rank/f'summaries/c20_weak{suffix}',O/f'summaries/c20_weak{suffix}')
        shutil.copy2(rank/f"summaries/{best['condition']}{suffix}",O/f"summaries/{best['condition']}{suffix}")
    rows=conditions(pool['rows']);weights=Path('/data/chenyiteng/models/protected-eval/pillbottle')/selected/'full_weights.pt'
    specs=[dict(condition=r['condition'].replace('cp180_','c20_'),policy=r['policy'],chunk=20,weights=weights,
                threshold={r['policy']:r['threshold'],'calibration_target':r['target_fraction']})for r in rows]
    make_stages(O,D/'c20-sweep',specs)
    save(O/'frozen-thresholds.json',{'rows':rows,'pool':pool,'strong':str(weights),'limit_queries':None})
    save(D/'c20-prepared.json',{'time':time.time(),'conditions':len(specs),'selected':selected,'unique_calibration_states':pool['unique_decisions']})


if __name__=='__main__':
    if sys.argv[1]=='c20':prepare_c20()
    else:prepare(sys.argv[1])
