"""Extract selected evidence; raw logs, complete plans and environments stay out."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
def main(label):
 d=json.loads((OUT/(label+'.out')).read_text());files=d['files']
 proofs={}
 for row in d.get('proofs',[]):
  for marker in ['WMRL_TWO_TO_ONE_RESTORE','WMRL_POLICY_BATCH_PROBE','WMRL_POLICY_ACTUAL_BATCH','WMRL_ACTOR_MEMORY']:
   if marker not in row['line']:continue
   data=json.loads(row['line'].split(marker+' ',1)[1])
   if marker=='WMRL_TWO_TO_ONE_RESTORE':
    params=data.pop('optimizer_parameters');data['optimizer_parameter_count']=len(params);data['optimizer_steps']=sorted(set(x['step'] for x in params))
   key=row['file'].split('/')[0]+'/'+marker
   if key not in proofs:proofs[key]={'source':row['file'],'data':data}
 value={'observed_at':d['time'],'task':'lift_pot','source_head':'9ce50c602c5e773e87e3306218fdde0cf167e8a8',
  'layout':{'policy_gpu':4,'wm_gpu':5,'policy_batch':64,'wm_batch':32,'N':64,'R':8,'G':8,'trajectories_per_formal_round':512,'actor_micro':16,'global_batch':2048,'C':32,'max_actions':384,'max_steps':200,'save_and_native_eval_interval':10},
  'checkpoint_resume':{'original_step':10,'optimizer_preservation_cpu_test':'passed','unsaved_old_completed_steps':[11,12,13,14,15]},
  'owner_directory':'/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/runs/lift-two-gpu-b32-v2',
  'state':{k:v for k,v in files.items() if k.endswith('.json') and k in ['owner-identity.json','state.json','startup-smoke-accepted.json','error.json','final.json','recovery-error.json','startup_smoke/result.json','formal/result.json','services/wm5/service-cpu-ready.json']},
  'proofs':proofs,'wm_batches':d.get('wm_batches'),'rlt_gpu6_gpu7':d['rlt'],'gpu_snapshot_csv':d['gpu'],'gpu_context_snapshot':d['contexts'],
  'prior_attempt':'CPU CLI rejected B32; no GPU borrowing; corrected in immutable service-v2',
  'ready_for_publication':True}
 dest=ROOT/'docs/world-model/task_reward_plan_20261006/two_gpu_b32_light.json';dest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'file':str(dest),'observed_at':d['time'],'proof_count':len(value['proofs'])}))
if __name__=='__main__':main(sys.argv[1])
