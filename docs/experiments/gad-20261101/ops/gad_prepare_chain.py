import copy,json,os,shutil,time
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1')
reference=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67-retention')
oldsignal=Path('/home/chenyiteng/deployment-20261031/pi05-nstep-sz1-g67-v1')
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
oldplan=read(reference/'formal-plan.json')
oldout=Path(read(reference/'prepared.json')['output'])
sd=D/'signal-recovery';sd.mkdir(exist_ok=True)
for gpu in [6,7]:
    old=read(oldsignal/f'lane-g{gpu}-plan.json');jobs=[];skipped=[]
    for job in old['jobs']:
        c=read(job['config']);out=Path(c['output']);valid=out/'validation.json'
        previous_log=oldsignal/(Path(job['config']).stem+'.log')
        if valid.exists() or previous_log.exists():
            skipped.append(dict(task=job['task'],reason='already_validated' if valid.exists() else 'already_attempted_task_error',
                                output=str(out)))
        else:jobs.append(job)
    old.update(jobs=jobs,prepared_time=time.time(),previously_accepted=True,noise_extra_batch=96)
    save(sd/f'lane-g{gpu}-plan.json',old)
    save(sd/f'prior-attempts-g{gpu}.json',skipped)
save(sd/'registered.json',dict(time=time.time(),scope='child lanes owned by GAD pair67'))
for pair,methods,mount in [('45',['endpoint','mfd'],'home'),('67',['cp_v2','onedp'],'data')]:
    pd=D/f'pair{pair}';po=Path(f'/{mount}/chenyiteng/results/pi05-distill/gad-single-g{pair}-20261101')
    pd.mkdir(exist_ok=True);po.mkdir(parents=True,exist_ok=True)
    cards=list(map(int,pair))
    save(pd/'prepared.json',dict(output=str(po),physical_gpus=cards,poll_seconds=.25))
    predecessors=[read('/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45-recover1/identity.json')] if pair=='45' else []
    plan=dict(methods=methods,heldout_seeds=oldplan['heldout_seeds'],predecessor_identities=predecessors,
              smoke_gate=str(D/'smoke-accepted.json'),run_smokes=pair=='67',
              smoke_output='/home/chenyiteng/results/pi05-distill/gad-smoke-20261101',
              signal_recovery=str(sd) if pair=='67' else None,
              checkpoint_reserve_gib=dict(endpoint=8,cp_v2=10,onedp=13,mfd=13))
    save(pd/'formal-plan.json',plan)
    for method in ['endpoint','cp_v2','onedp','mfd']:
        save(pd/f'{method}.json',read(D/'formal-configs'/f'{method}.json'))
    shutil.copy2(reference/'eval-template.json',pd/'eval-template.json')
    shutil.copytree(reference/'seeds',pd/'seeds',dirs_exist_ok=True)
    for gpu,reference_gpu in zip(cards,[6,7]):
        env=read(oldout/f'environment-g{reference_gpu}.json')
        # Match the evaluator's established per-physical-card environment.
        for key in ['CUDA_VISIBLE_DEVICES','EGL_DEVICE_ID','MUJOCO_EGL_DEVICE_ID']:
            if key in env:env[key]=str(gpu)
        if 'DOJO_GPU_SCOPE' in env:env['DOJO_GPU_SCOPE']=str(gpu)
        env.pop('DISTILL_RUN_TOKEN',None)
        save(po/f'environment-g{gpu}.json',env)
print(json.dumps(dict(pair45=read(D/'pair45'/'formal-plan.json'),pair67=read(D/'pair67'/'formal-plan.json'),
                      recovery_remaining={str(g):len(read(sd/f'lane-g{g}-plan.json')['jobs']) for g in [6,7]})))
