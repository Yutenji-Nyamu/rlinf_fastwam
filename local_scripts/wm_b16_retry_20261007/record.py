"""Record the current lean owner without regenerating historical documentation."""
import json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2];D=W/'docs/world-model/task_reward_plan_20261006'
E=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
s=json.loads((E/(sys.argv[1]+'.out')).read_text())
key='runs/lift-two-gpu-b16-lean-20261007-v1/'
owner=s['files'].get(key+'owner-identity.json',{});driver=s['files'].get(key+'formal/driver-identity.json',{})
phase='formal_sampling' if s['wm'].get('batches',0) else 'formal_initialization' if driver else 'cpu_loading'
if key+'final.json' in s['files']: phase=s['files'][key+'final.json']['terminal_status']
light={k:s.get(k) for k in ['time','files','wm','health','rlt','protected','gpu']}
light.update(phase=phase,task='lift_pot',source_head='9ce50c602c5e773e87e3306218fdde0cf167e8a8',
             cpu_tests_passed=6,training_config_equal=True,service_parameters_equal=True)
(D/'two_gpu_b16_lean_light.json').write_text(json.dumps(light,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
route=f"2026-10-07 {s['time'][11:16]} WMRL精简重启：控制lift-two-gpu-b16-lean-20261007-v1，owner {owner.get('pid')}/start {owner.get('start')}，driver {driver.get('pid','待启动')}，phase={phase}，WM已完成{s['wm'].get('batches',0)}批。GPU4/5、B16/N64/R8/512条/200轮/每10轮评估保持，从原π0.5起训；旧轮无完整CP。已移除运行期Ray Dashboard依赖和重复清理，709→473行，6项CPU回归通过；真正退出后才清理并自动归还RLT，6/7原实验不动。当前仅新入口有效，不重放旧owner。详见[精简记录](two_gpu_b16_simplification_20261007.md)。"
for p in [D/'README.md',W/'HANDOFF.md']:
 text=p.read_text(encoding='utf-8'); head,rest=text.split('\n',1)
 paragraphs=[x for x in rest.strip('\n').split('\n\n') if not (x.startswith('2026-10-07 ') and ('WMRL B16正式：' in x or 'WMRL精简重启：' in x))]
 line=route if p.parent==D else route.replace('(two_gpu_b16_simplification_20261007.md)','(docs/world-model/task_reward_plan_20261006/two_gpu_b16_simplification_20261007.md)')
 p.write_text(head+'\n\n'+line+'\n\n'+'\n\n'.join(paragraphs)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'phase':phase,'owner':owner.get('pid'),'driver':driver.get('pid'),'wm_batches':s['wm'].get('batches',0)}))
