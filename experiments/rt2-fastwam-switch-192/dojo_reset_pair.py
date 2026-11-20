"""One isolated cold-versus-warm native reset with the same 32 training layouts."""
import json, os, subprocess, sys, time
from pathlib import Path

def maybe_run(cfg, script):
    # The parent is a new simulator generation; it has not initialized Isaac yet.
    if os.environ.get("DOJO_RESET_PAIR_CHILD") or cfg["gpu"] != 5:
        return
    deployment=Path(cfg['output']).parent
    trial=deployment/'reset-pair'
    if not (trial/'armed.json').is_file() or (trial/'started.json').exists():
        return
    plan=json.loads((deployment/'plan.json').read_text())
    root=Path(plan['run'])/'pour-clean-256/checkpoints'
    if not any(root.glob('global_step_*/dojo-complete.json')):
        return
    started=time.time()
    with (trial/'started.json').open('x') as f:json.dump({'time':started},f)
    shadow={**cfg,'output':str(trial/'native')}
    cfg_path=trial/'native.json';cfg_path.write_text(json.dumps(shadow))
    env={**os.environ,'DOJO_RESET_PAIR_CHILD':str(trial),'DOJO_GRPO_NATIVE_CONFIG':str(cfg_path),'DOJO_RESET_PAIR_START':str(started)}
    try:
        with (trial/'simulator.log').open('w') as log:
            ret=subprocess.run([sys.executable,str(script),*sys.argv[1:]],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        result={'exit_code':ret.returncode}
    except (OSError,subprocess.TimeoutExpired) as exc:
        result={'error':str(exc)}
    (trial/'finished.json').write_text(json.dumps({**result,'seconds':time.time()-started}))

def run_in_shadow(loop, directory):
    import numpy as np
    trial=Path(directory)
    request=json.loads((trial/'armed.json').read_text())
    records=[];observations=[]
    for method in ('fresh','reused'):
        begin=time.monotonic()
        out=loop.reset(request)
        audit=out['audit'];obs=out['obs']
        summary={k:np.asarray(v).copy() if k=='states' else np.asarray(v)[...,::8,::8,:].copy() for k,v in obs.items() if k in ('states','main_images','wrist_images')}
        observations.append(summary)
        records.append(dict(method=method,ready_seconds=time.monotonic()-begin,native_reset_seconds=loop.totals['reset_seconds']-sum(r['native_reset_seconds'] for r in records),layouts=audit['layout_ids'],prompts=list(obs['task_descriptions']),registered=audit['registered_success_conditions']))
        if method=='fresh':
            first_ready=time.time()-float(os.environ['DOJO_RESET_PAIR_START'])
            # A single native hold action exercises the controller before the second reset.
            loop.step(dict(mode='train',actions=np.asarray(obs['states'])[:,None,:]))
    differences={}
    for k in observations[0]:
        gap=np.abs(observations[0][k].astype(np.float32)-observations[1][k].astype(np.float32))
        differences[k]={'mean_absolute':float(gap.mean()),'max_absolute':float(gap.max())}
    (trial/'result.json').write_text(json.dumps(dict(records=records,fresh_launch_to_ready=first_ready,observation_difference=differences,same_layouts=records[0]['layouts']==records[1]['layouts'],same_prompts=records[0]['prompts']==records[1]['prompts']),indent=2))

