"""Create a small publication snapshot from fetched, read-only result files."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--status',required=True);a=p.parse_args()
 pilot=read(E/'task-rm-1006-status03.out')['files'];v1=read(E/'task-rm-1006-status04.out')['files'];now=read(E/a.status)
 out=dict(schema_version=1,observed_at=now['time'],source_snapshot=a.status,
  stopped_wm=dict(checkpoint=50,completed_checkpoint_verified=True,user_requested_stop=True,cleanup_complete=True,rlt_return_dispatched=True),
  cpu_tests=dict(total=5,passed=5),
  bottle_pilot=dict(episodes=32,native_successes=15,native_failures=17,source_mode='Existing K8 frames with exact native indices; not full-resolution full-sequence capture',profile=pilot['pilot/profile.json'],report=pilot['pilot/report.json']),
  v1_retry=dict(reason='Inherited RLINF_CODE_WORKING_DIR selected old bell runtime without recorder; corrected to private capture repository in v2',native_valid_episodes=0,final=v1['run/final.json']),
  current=dict(attempt='task-reward-v2',owner_live=now.get('owner_live'),owner_identity=now['files'].get('run/owner-identity.json'),state=now['files'].get('run/state.json'),heartbeat=now['files'].get('run/heartbeat.json'),capture=now['capture'],untouched_rlt_live=now.get('untouched_rlt_live'),final=now['files'].get('run/final.json')),
  lift_pot_training=dict(profile=now['files'].get('lift-pot/profile.json'),report=now['files'].get('lift-pot/report.json'),status=now['files'].get('lift-pot/status.json')),
  gpu_scope=dict(rm=[4],rlt_unchanged=[5,6,7],no_other_user_process_cleanup=True),
  contract=dict(model='Single-task ResNet18 + MLP256',image='main RGB,224x224,ImageNet normalization',output='[B] sigmoid success scores; validation-selected threshold to0/1',native_collection='original SFT; clean; N16x8; C32/H50;384actions',training='global64,measured micro32/64,FP32,AdamW1e-4,GPU image cache,max100epochs,patience15'),
  limitations=['Pilot threshold is unusable despite high test ranking AUC','Native held-out accuracy is not generated-domain accuracy','No new-task GRPO deployment is claimed'])
 target=ROOT/'docs/world-model/task_reward_plan_20261006/light_results.json'
 target.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'written':str(target),'bytes':target.stat().st_size,'phase':out['current']['state']}))
if __name__=='__main__':main()
