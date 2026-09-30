"""CPU-only composition of actual Hydra recipes before borrowing GPUs."""
import json,os
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
r=Path('/data/chenyiteng/projects/wan-goal-sz3')
os.environ.update(WAN_GOAL_RUN_DIR=str(r/'runs/placeholder'),WAN_GOAL_WM_PATH=str(r/'models/wan-goal'),WAN_GOAL_OFT_PATH=str(r/'models/oft-goal'),WAN_GOAL_PI05_PATH=str(r/'models/pi05-libero'))
for repo,name in [('RLinf','wan_goal_oft_smoke_sz3'),('RLinf-pi05','wan_goal_pi05_headonly_smoke_sz3'),('RLinf-pi05','wan_goal_pi05_headonly_formal_sz3')]:
 os.environ['EMBODIED_PATH']=str(r/repo/'examples/embodiment')
 with initialize_config_dir(config_dir=str(r/repo/'examples/embodiment/config'),version_base='1.1'):
  cfg=compose(config_name=name)
  value=OmegaConf.to_container(cfg,resolve=True)
  assert value['cluster']['component_placement']['actor,env,rollout']=='4-7'
  (r/'logs'/f'{name}-composed.json').write_text(json.dumps(value,indent=2))
  print(json.dumps({'config':name,'env_type':value['env']['train']['env_type'],'backend':value['env']['train'].get('backend'),'epochs':value['runner']['max_epochs'],'N':value['env']['train']['total_num_envs'],'C':value['actor']['model']['num_action_chunks'],'batch':value['actor']['global_batch_size']}))
