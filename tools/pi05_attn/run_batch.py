"""Native B16/H50/M10 inference and full raw attention recording, no training."""
import os,json,time,random,traceback,argparse
from pathlib import Path
import numpy as np
import torch
from observer import AttentionObserver
import cv2
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def arr(x):
    if torch.is_tensor(x):return x.detach().cpu().float().numpy() if x.dtype==torch.bfloat16 else x.detach().cpu().numpy()
    return np.array(x,copy=True)
def rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])
def restore(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])
def same_rng(a,b):
    return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3]))
def infer(model,obs):
    with torch.inference_mode():return model.predict_action_batch(obs,mode='eval',compute_values=False,return_dvac_telemetry=True)
def run(config):
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    c=json.loads(Path(config).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(c['gpu']) and c['num_envs'] in [16,32]
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.backends.cuda.enable_cudnn_sdp(False)
    batch=c['num_envs'];started=time.time();save(out/'started.json',dict(time=started,pid=os.getpid(),gpu=c['gpu']));save(out/'config.json',c)
    model=get_model(OmegaConf.create(c['model'])).cuda().eval()
    assert model.config.num_steps==10 and model.config.action_horizon==50
    observer=AttentionObserver(model);env=None;videos=[];frame_times=[[] for _ in range(batch)];success=np.zeros(batch,bool);counts=np.zeros(batch,int);costs=[];rows=[]
    try:
        ec=OmegaConf.create(c['env']);assert not ec.auto_reset and ec.ignore_terminations
        env=RoboTwinEnv(ec,batch,0,1,None);obs,_=env.reset()
        assert c['env']['video_cfg']['save_video']
        requested=env.reset_state_ids.cpu().tolist();actual=[int(e.task.ep_num) for e in env.venv.envs]
        seeds=[dict(slot=i,requested=int(requested[i]),actual=actual[i],native_retry=int(requested[i])!=actual[i]) for i in range(batch)]
        save(out/'seeds.json',seeds)
        random.seed(c['noise_seed']);np.random.seed(c['noise_seed']);torch.manual_seed(c['noise_seed']);torch.cuda.manual_seed_all(c['noise_seed'])
        # Reused from frozen signal_inference_20261003/run_batch.py, lines 290-305.
        heads=arr(obs['main_images']);height,width=heads.shape[1:3]
        for slot in range(batch):
            path=out/f'episode_{slot:02d}.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),4,(width,height))
            if not writer.isOpened():raise RuntimeError(f'Cannot open video writer {path}')
            videos.append(writer)
        def snapshot(images):
            for slot,image in enumerate(images):
                videos[slot].write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                frame_times[slot].append(time.time())
        snapshot(heads)
        for query in range((c['step_limit']+49)//50):
            active=~success&(counts<c['step_limit'])
            if not active.any():break
            before=counts.copy();obs.setdefault('extra_view_images',None)
            descriptions=list(obs['task_descriptions'])
            obs_arrays={k:arr(obs[k]) for k in ['main_images','wrist_images','states']}
            save(out/'progress.json',dict(time=time.time(),phase='inference',query=query,successes=int(success.sum())))
            base_seconds=None
            if query==0 and c.get('parity',False):
                r0=rng();torch.cuda.synchronize();t=time.perf_counter();base_action,base=infer(model,obs);torch.cuda.synchronize();base_seconds=time.perf_counter()-t;r1=rng()
                restore(r0);torch.cuda.synchronize();bt=time.perf_counter();again_action,again=infer(model,obs);torch.cuda.synchronize();base_repeat_seconds=time.perf_counter()-bt;r2=rng()
                assert np.array_equal(arr(base_action),arr(again_action)) and torch.equal(base['forward_inputs']['chains'],again['forward_inputs']['chains']) and same_rng(r1,r2),'Native replay is not deterministic'
                restore(r0)
            observer.begin(reference=query==0 and c.get('parity',False))
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();action,result=infer(model,obs);torch.cuda.synchronize();infer_seconds=time.perf_counter()-t
            raw,metadata=observer.finish(result,action)
            if base_seconds is not None:
                eq=dict(env_action=np.array_equal(arr(base_action),arr(action)),full_chain=torch.equal(base['forward_inputs']['chains'],result['forward_inputs']['chains']),rng=same_rng(r1,rng()))
                assert all(eq.values()),eq
                from analysis import reconstruct
                _,reference=reconstruct(raw,device='cuda')
                # v1.1 separates exact native reconstruction from the FP32 analysis path.
                capture_exact=max(r['native_precision_max_abs'] for r in reference)==0
                main_ok=max(r['main_max_abs'] for r in reference)<=.02 and max(r['main_max_row_l1'] for r in reference)<=.08
                report=dict(passed=capture_exact and main_ok,protocol='v1.1',query=query,equal=eq,native_repeat_exact=True,native_attention_exact=capture_exact,baseline_seconds=base_seconds,baseline_warm_repeat_seconds=base_repeat_seconds,recording_seconds=infer_seconds,
                    reconstruction=reference,tolerance=dict(native_precision_max_abs=0,fp32_main_max_abs=.02,fp32_main_max_row_l1=.08),
                    legacy_per_head_limit_passed=max(r['max_abs'] for r in reference)<=.02 and max(r['max_row_l1'] for r in reference)<=.08,
                    note='Native dtype reproduction must be exact. FP32 head differences retained; fixed main head-then-layer mean limits. Both baseline calls excluded from execution; see calibration history.')
                save(out/'parity.json',report)
                if not report['passed']:
                    torch.save(raw,out/'parity-failure.pt');np.savez_compressed(out/'parity-failure-obs.npz',**obs_arrays);save(out/'parity-failure.json',dict(metadata,execution='not submitted'));raise AssertionError('Native capture or FP32 main-distribution check failed; raw saved')
                del base,again,base_action,again_action
            t=time.perf_counter();obss,_,_,_,_=env.chunk_step(arr(action));obs=obss[-1];snapshot(arr(obs['main_images']))
            counts=np.array([int(e.task.take_action_cnt) for e in env.venv.envs]);after=np.array([bool(e.task.eval_success) for e in env.venv.envs])
            submitted=active[:,None]&(np.arange(50)[None,:]<(counts-before)[:,None])
            raw.update(active=torch.from_numpy(active),submitted_mask=torch.from_numpy(submitted),executed_mask=torch.from_numpy(submitted&~after[:,None]),executed_mask_valid=torch.from_numpy(np.broadcast_to(~after[:,None],(batch,50)).copy()),
                success_after=torch.from_numpy(after),action_slot_start=torch.from_numpy(before),action_slot_end=torch.from_numpy(counts))
            env_seconds=time.perf_counter()-t;t=time.perf_counter()
            path=out/f'query_{query:03d}.pt';torch.save(raw,path)
            np.savez_compressed(out/f'query_{query:03d}_obs.npz',**obs_arrays)
            metadata.update(pre_frame=query,post_frame=query+1,video_sampling='head camera before/after native chunk, 4 fps preview',query=query,task=c['task'],task_descriptions=descriptions,physical_prefix='unknown for terminal success chunks (native TOPP)',run_manifest='../../manifest.json',branch_id='native_main',actual_seeds=actual,requested_seeds=requested,raw=path.name,observation=f'query_{query:03d}_obs.npz',sample_index=list(range(batch)))
            save(out/f'query_{query:03d}.json',metadata)
            cost=dict(query=query,inference_seconds=infer_seconds,baseline_seconds=base_seconds,qk_copy_sync_seconds=metadata['qk_copy_sync_seconds'],env_seconds=env_seconds,write_seconds=time.perf_counter()-t,raw_bytes=path.stat().st_size,obs_bytes=(out/f'query_{query:03d}_obs.npz').stat().st_size,max_cuda_allocated=torch.cuda.max_memory_allocated())
            costs.append(cost);success|=after;rows.append(dict(query=query,active=int(active.sum()),successes=int(success.sum()),counts=counts.tolist()))
            save(out/'queries.json',rows);save(out/'storage_and_timing.json',costs);save(out/'progress.json',dict(time=time.time(),phase='saved',query=query+1,successes=int(success.sum())))
            del raw,result,action,obs_arrays
        assert not np.any(~success&(counts<c['step_limit']))
        save(out/'episodes.json',[dict(task=c['task'],batch=c['batch'],**seeds[i],video=f'episode_{i:02d}.mp4',frames=len(frame_times[i]),video_sampling='head camera before/after native chunk, 4 fps preview',success=bool(success[i]),submitted=int(counts[i]),termination='success' if success[i] else 'step_limit') for i in range(batch)])
    finally:
        try:
            for writer in videos:writer.release()
            save(out/'frame_times.json',frame_times)
        finally:
            if env is not None:env.venv.env_thread_pool.shutdown(wait=True);env.offload(clear_cache=True)
            observer.close()
    save(out/'done.json',dict(time=time.time(),elapsed_seconds=time.time()-started,queries=len(rows),episodes=batch,successes=int(success.sum()),unique_actual_seeds=len(set(actual)),schema='pi05-attn-raw-v1',video_sampling='accepted 20261003 chunk endpoints'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');a=ap.parse_args()
    try:run(a.config)
    except BaseException:
        c=json.loads(Path(a.config).read_text());p=Path(c['output']);p.mkdir(parents=True,exist_ok=True);save(p/'error.json',dict(time=time.time(),error=traceback.format_exc()));raise
