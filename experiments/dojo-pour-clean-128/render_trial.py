"""Matched-action crossover for Dojo pouring using the released policy."""
import concurrent.futures, json, sys, time
from multiprocessing.connection import Client
from pathlib import Path
import numpy as np

def rms(x,y):
    delta=np.asarray(x,dtype=np.float32)-np.asarray(y,dtype=np.float32)
    return float(np.sqrt(np.mean(delta**2)))

def main(root):
    import torch
    from omegaconf import OmegaConf
    from openpi.models.model import Observation
    from rlinf.models.embodiment.openpi import get_model
    d=Path(root); t=d/'trial'; key=(d/'rpc/authkey').read_bytes()
    def rpc(g, req):
        with Client(str(t/f'g{g}.sock'),family='AF_UNIX',authkey=key) as c:
            c.send(req)
            if not c.poll(300):raise TimeoutError((g, req['op']))
            res=c.recv()
            if not res['ok']:raise RuntimeError(res.get('error',str(res)[:200]))
            return res['value']
    def paired(req):
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            return list(ex.map(rpc,(5,6),req))
    model=get_model(OmegaConf.load(d/'config/formal.yaml').actor.model).eval().cuda()
    def infer(obs,seed):
        tensors={k:torch.from_numpy(obs[k]) for k in ('states','main_images','wrist_images')}
        tensors.update(task_descriptions=obs['task_descriptions'],extra_view_images=None)
        proc=model.input_transform(model.obs_processor(tensors),transpose=False)
        ob=Observation.from_dict(model.precision_processor(proc))
        noise=np.random.RandomState(seed).normal(size=(len(obs['states']),50,32)).astype('float32')
        torch.manual_seed(seed)
        with torch.inference_mode():
            x=model.sample_actions(ob,noise=torch.from_numpy(noise).cuda(),mode='eval',compute_values=False)
            act=model.output_transform({'actions':x['actions'],'state':ob.state})['actions']
        return act.detach().float().cpu().numpy() if torch.is_tensor(act) else np.asarray(act,np.float32)
    deadline=time.monotonic()+450
    while not all((t/f'g{g}.sock').exists() for g in (5,6)):
        if time.monotonic()>deadline:raise TimeoutError('trial simulators')
        time.sleep(1)
    layouts=json.loads((t/'layouts.json').read_text())
    seq=[];saved={};stats=[];outcomes=[];profile=[];traces=[]
    for run in range(2):
        modes=['reference','candidate'] if run==0 else ['candidate','reference']
        resets=paired([dict(op='reset',mode='eval',layout_ids=layouts,group_size=1,request_id=-800-run)]*2)
        assert all(r['audit']['layout_ids']==layouts and all(r['audit']['registered_success_conditions']) for r in resets)
        obs=[r['obs'] for r in resets];times=[0.,0.];track=[]
        for k in range(8):
            a,b=[infer(o,1947+k) for o in obs]
            metrics=dict(run=run,query=k,actions=rms(a,b),max_action=float(np.max(np.abs(a-b))))
            for keyname in ('states','main_images','wrist_images'):
                metrics[keyname]=rms(obs[0][keyname],obs[1][keyname])
            for j in range(2):
                metrics[f'wrist{j}']=rms(obs[0]['wrist_images'][:,j],obs[1]['wrist_images'][:,j])
            if run==0:
                for name in ('states','main_images','wrist_images'):
                    x=obs[0][name]
                    saved[(k,name)]=x.copy() if name=='states' else x[...,::4,::4,:].copy()
                seq.append(a.copy());saved[k]=a.copy()
            else:
                metrics['repeat_reference_actions']=rms(saved[k],b)
                for name in ('states','main_images','wrist_images'):
                    x=obs[1][name]
                    metrics['repeat_'+name]=rms(saved[(k,name)],x if name=='states' else x[...,::4,::4,:])
                for j in range(2):
                    metrics[f'repeat_wrist{j}']=rms(saved[(k,'wrist_images')][:,j],obs[1]['wrist_images'][:,j,::4,::4,:])
            stats.append(metrics)
            res=paired([dict(op='step',mode='eval',actions=seq[k],trial_skip=(m=='candidate')) for m in modes])
            times=[x+r['trial_seconds'] for x,r in zip(times,res)]
            track.append([[list(map(int,r['counts'])),r['success'].astype(int).tolist(),np.where(r['terminations']|r['truncations'])[1].tolist()] for r in res])
            obs=[r['obs'] for r in res]
            traces.append(dict(run=run,chunk=k+1,seconds=times))
            (t/'progress.json').write_text(json.dumps(traces[-1]))
        outcomes.append([dict(mode=m,seconds=dt,counts=list(map(int,r['counts'])),success=r['success'].astype(int).tolist()) for m,dt,r in zip(modes,times,res)])
        profile.append([r['trial_profile'] for r in res])
        track_equal=all(v[0]==v[1] for v in track)
        outcomes[-1][0]['same_terminal_trace']=track_equal
    a=[p for run in outcomes for p in run if p['mode']=='reference'];b=[p for run in outcomes for p in run if p['mode']=='candidate']
    speed=1-sum(p['seconds'] for p in b)/sum(p['seconds'] for p in a)
    cross=max(x['actions'] for x in stats);repeat=max(x.get('repeat_reference_actions',0) for x in stats)
    # For a small panel require consistent terminal events and the previously used policy tolerance.
    agree=all(r[0]['same_terminal_trace'] for r in outcomes) and all((p['success'],p['counts'])==(a[0]['success'],a[0]['counts']) for p in a+b)
    within_repeat=all(max(q[name] for q in stats)<=max(q.get('repeat_'+name,0) for q in stats)+tolerance for name,tolerance in [('main_images',1.),('wrist0',1.),('wrist1',1.),('states',0.005)])
    accept=bool(within_repeat and any(a[0]['success']) and agree and speed>0 and all(bi['seconds']<ai['seconds'] for ai,bi in zip(a,b)) and cross<0.02 and max(x['max_action'] for x in stats)<0.15 and cross<=repeat+0.002)
    np.savez_compressed(t/'actions.npz',actions=np.stack(seq),layouts=layouts)
    result=dict(layouts=layouts,repeat='crossed GPU assignment and same open-loop actions',timing_excludes='reset, policy inference and RPC',outcomes=outcomes,queries=stats,profile=profile,fraction_saved=speed,repeat_reference_rmse_max=repeat,candidate_rmse_max=cross,accept_train=accept)
    tmp=t/'result.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(t/'result.json')
    print(json.dumps({k:result[k] for k in ('fraction_saved','candidate_rmse_max','repeat_reference_rmse_max','accept_train')}),flush=True)
if __name__=='__main__':main(sys.argv[1])

