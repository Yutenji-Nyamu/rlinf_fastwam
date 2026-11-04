"""Final-policy fixed pool and six-checkpoint inference, using accepted lanes."""
import concurrent.futures,csv,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from entropy_analysis import comparison

def main(directory):
 d=Path(directory);p=json.loads((d/'plan.json').read_text());o=Path(p['output']);cards=p.get('gpus',[4,5])
 protected=Path('/data/chenyiteng/models/protected-eval/pillbottle')/Path(p['run']).name
 linked=[]
 for step in p['milestones'][1:]:
  source_weight=Path(p['checkpoint_root'])/f'global_step_{step}/actor/model_state_dict/full_weights.pt'
  target=protected/f'cp{step:03d}'/'full_weights.pt';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():os.link(source_weight,target)
  if not os.path.samefile(source_weight,target):raise RuntimeError('Protected weight path collision')
  linked.append({'step':step,'source':str(source_weight),'protected':str(target),'do_not_delete':True})
 (o/'protected-analysis-weights.json').write_text(json.dumps(linked,indent=2))
 source=Path(p['inference_source']);sys.path.insert(0,str(source/'tools/pi05_distillation'))
 from multitask_eval import cleanup
 active_tokens=set()
 def stop(*_):
  for token in tuple(active_tokens):cleanup(token)
  raise SystemExit(143)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def run(command,gpu,token,log,timeout):
  env=dict(os.environ,**json.loads((d/f'inference-environment-g{gpu}.json').read_text()))
  env.update(CUDA_VISIBLE_DEVICES=str(gpu),DOJO_GPU_SCOPE=str(gpu),POLICY_EVAL_TOKEN=token,PI05_SIGNAL_SOURCE_ROOT=str(source),PYTHONPATH=str(source)+':'+env['ROBOTWIN_PATH'])
  child=None
  active_tokens.add(token)
  try:
   with log.open('x') as f:
    child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    rc=child.wait(timeout=timeout)
   if rc:raise RuntimeError(f'{log.name}: exit {rc}')
  finally:
   cleanup(token);active_tokens.discard(token)
 def collect(gpu):
  cfg=d/f'pool-g{gpu}.json'
  run([p['python'],'-u','-B',str(d/'tools/pi05_u_video/run_batch.py'),str(cfg)],gpu,f'{d.name}-pool-{gpu}',d/f'pool-g{gpu}.log',3600)
  b=Path(json.loads(cfg.read_text())['output']);done=json.loads((b/'done.json').read_text())
  if done['episodes']!=16:raise RuntimeError('Incomplete final-policy pool; stop six-point analysis')
  return b
 with concurrent.futures.ThreadPoolExecutor(2) as ex:batches=list(ex.map(collect,cards))
 def replay_lane(item):
  gpu,batch=item
  for step in p['milestones']:
   out=o/f'checkpoint-{step:03d}'/f'g{gpu}'
   weights=None if step==0 else str(Path(p['checkpoint_root'])/f'global_step_{step}/actor/model_state_dict/full_weights.pt')
   if weights and not Path(weights).is_file():raise FileNotFoundError(weights)
   cfg={'gpu':gpu,'model_source':str(source),'input_batch':str(batch),'output':str(out),'weights':weights,'checkpoint_step':step,'pool_batch':cards.index(gpu)}
   f=d/f'replay-cp{step:03d}-g{gpu}.json';f.write_text(json.dumps(cfg))
   run([p['python'],'-u','-B',str(d/'tools/checkpoint_replay.py'),str(f)],gpu,f'{d.name}-cp{step}-g{gpu}',d/f'replay-cp{step:03d}-g{gpu}.log',1200)
 with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(replay_lane,zip(cards,batches)))
 vectors={};rows=[];keys=None
 for step in p['milestones']:
  data=[]
  for g in cards:data+=json.loads((o/f'checkpoint-{step:03d}'/f'g{g}/positions.json').read_text())
  data.sort(key=lambda r:(r['pool_batch'],r['slot'],r['query'],r['h']))
  key=[(r['pool_batch'],r['slot'],r['query'],r['h'])for r in data]
  if keys is None:keys=key
  assert keys==key and len(keys)>0,'Checkpoint positions differ'
  vectors[step]=np.array([r['entropy']for r in data])
  rows += [dict(checkpoint_step=step,**r)for r in data]
 with (o/'six-checkpoint-positions.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 comparisons=[]
 for step in p['milestones']:
  base=comparison(vectors[0],vectors[step]);final=comparison(vectors[200],vectors[step])
  comparisons.append({'step':step,'top20_overlap_base':base['overlap'],'top20_overlap_final':final['overlap'],'mean_entropy':float(vectors[step].mean()),'entropy_q50_q80_q95':base['final_quantiles']})
 summary={'complete':True,'same_run':p['run'],'milestones':p['milestones'],'pool_policy':'CP200','pool_episodes':32,'positions':len(keys),'checkpoint_comparison':comparisons,'initial_entropy_deciles':comparison(vectors[0],vectors[200])['bins'],'time':time.time(),'definition':'K4 diagonal Gaussian; D14; ddof1; eps1e-6; fixed final-policy reached positions'}
 (o/'six-checkpoint-summary.json').write_text(json.dumps(summary,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
 matrix=np.array([[r['top20_overlap_base'],r['top20_overlap_final']]for r in comparisons])*100
 im=ax[0].imshow(matrix.T,vmin=0,vmax=100,aspect='auto',cmap='Blues')
 ax[0].set(xticks=range(6),xticklabels=p['milestones'],yticks=[0,1],yticklabels=['vs initial','vs final'],xlabel='GRPO training round',title='Top 20% entropy position overlap (%)')
 for i in range(6):
  for j in range(2):ax[0].text(i,j,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white'if matrix[i,j]>65 else 'black')
 fig.colorbar(im,ax=ax[0])
 delta=[r['delta']for r in summary['initial_entropy_deciles']]
 ax[1].bar(range(1,11),delta,color=['#087f8c'if x>=0 else '#d97736'for x in delta]);ax[1].axhline(0,color='#777',lw=.8)
 ax[1].set(xticks=range(1,11),xlabel='Groups defined by initial entropy',ylabel='Mean final minus initial entropy',title='Same positions throughout training')
 ax[2].plot(p['milestones'],[r['mean_entropy']for r in comparisons],'--s',label='Mean',color='#222222')
 for j,(label,color) in enumerate([('Median','#0072B2'),('80th percentile','#D55E00'),('95th percentile','#7B2CBF')]):
  ax[2].plot(p['milestones'],[r['entropy_q50_q80_q95'][j]for r in comparisons],'-o',label=label,color=color)
 ax[2].set(xlabel='GRPO training round',ylabel='Action entropy estimate',title='Fixed final-policy observation pool');ax[2].legend()
 for ext in ['png','svg','pdf']:fig.savefig(o/f'six-checkpoint-entropy.{ext}',dpi=180)
 plt.close(fig)
 print(json.dumps(summary))

if __name__=='__main__':main(sys.argv[1])
