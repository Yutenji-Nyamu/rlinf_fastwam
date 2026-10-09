import ast, json, logging, types
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from unittest.mock import patch
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');R=S/'wm-cycle-20261009-v2'
Accelerator(cpu=True)
measured=[]
for coupled in [True,False]:
 tensor=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([tensor],lr=1e-5)
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=835,eta_min=1e-7)
 wrapped=AcceleratedScheduler(schedule,[types.SimpleNamespace(step_was_skipped=False)],step_with_optimizer=coupled,split_batches=False)
 with patch('accelerate.scheduler.AcceleratorState',return_value=types.SimpleNamespace(num_processes=2)):
  opt.step();wrapped.step()
 measured.append(dict(step_with_optimizer=coupled,last_epoch=schedule.last_epoch,lr=schedule.get_last_lr()[0]))
assert [x['last_epoch'] for x in measured]==[2,1]
print('SCHEDULER',json.dumps(measured))
metrics=[json.loads(x) for x in (R/'run/wm000/train/metrics.jsonl').read_text().splitlines()]
print('OLD_LR',metrics[0]['lr'],metrics[-1]['lr'])
assert abs(measured[0]['lr']-metrics[0]['lr'])<1e-12
code=R/'code'
for path in code.glob('*.py'):ast.parse(path.read_text())
source=ast.parse((code/'wm_train.py').read_text())
fn=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
ns={'logger':logging.getLogger('save-contract')};exec(compile(ast.Module(body=[fn],type_ignores=[]),'save','exec'),ns)
upstream=ast.parse((S/'opendw/dexbotic/exp/generative_trainer.py').read_text())
finish=next(n for n in ast.walk(upstream) if isinstance(n,ast.FunctionDef) and n.name=='_finish_training_due_to_max_steps')
exec(compile(ast.Module(body=[finish],type_ignores=[]),'official_finish','exec'),ns)
fake=types.SimpleNamespace(accelerator=types.SimpleNamespace(is_main_process=True,wait_for_everyone=lambda:None),global_step=835,save_final=True,_finish_wandb=lambda:None,_save_weights_checkpoint=lambda tag:'/tmp/'+tag+'.pt')
fake.save_checkpoint=types.MethodType(ns['save_checkpoint'],fake)
for main in [True,False]:
 fake.accelerator.is_main_process=main
 ns['_finish_training_due_to_max_steps'](fake)
print('SAVE_CONTRACT_AND_SYNTAX_PASS')
result=dict(scheduler=measured,old_first_lr=metrics[0]['lr'],old_final_lr=metrics[-1]['lr'],save_contract='pass',syntax='pass')
(R/'recovery-1009/contracts.json').write_text(json.dumps(result,indent=2)+'\n')
print('TRACKED',__import__('subprocess').run(['git','-C',str(S/'publication/wm-cycle-20261009'),'ls-files','local_scripts/wovr_cycle_20261008'],capture_output=True,text=True).stdout)
