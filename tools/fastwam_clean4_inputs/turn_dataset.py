"""Reuse the pinned RoboTwin clean50 conversion functions for turn_switch."""
import os,json,zipfile,importlib.util,random
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
ROOT=Path('/data/chenyiteng/datasets/robotwin2')
REV='9dc9299c163db059931898a9f0852098a61155a1'
TASK='turn_switch'
archive=Path(hf_hub_download(repo_id='TianxingChen/RoboTwin2.0',repo_type='dataset',revision=REV,filename=f'dataset/{TASK}/aloha-agilex_clean_50.zip',local_dir=str(ROOT/'source'/REV)))
raw=ROOT/'raw'/REV/TASK/'clean50-20260919'
raw.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(raw)
raw=raw/'aloha-agilex_clean_50'
assert len(list((raw/'data').glob('episode*.hdf5')))==50
tool=ROOT/'tooling/RoboTwin-c3ddfa8b97d5519efa828b075999bd0006778e5e/policy/pi0'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
inter=ROOT/'intermediate'/REV/TASK/'clean50-20260919';assert not inter.exists()
process=module('raw_to_aloha',tool/'scripts/process_data.py');process.data_transform(str(raw),50,str(inter))
dest=ROOT/'canonical';repo_id='turn-switch-aloha-clean50-20260919';assert not (dest/repo_id).exists()
os.environ['HF_LEROBOT_HOME']=str(dest)
converter=module('aloha_to_lerobot',tool/'examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py')
random.seed(0);np.random.seed(0)
files=sorted(inter.rglob('episode_*.hdf5'),key=lambda p:int(p.stem.split('_')[-1]));assert len(files)==50
d=converter.create_empty_dataset(repo_id,robot_type='aloha',mode='image',has_effort=converter.has_effort(files),has_velocity=converter.has_velocity(files),dataset_config=converter.DEFAULT_DATASET_CONFIG)
converter.populate_dataset(d,files,task='turn the switch',episodes=list(range(50)))
info=json.loads((dest/repo_id/'meta/info.json').read_text());assert info['total_episodes']==50
print(json.dumps({'complete':True,'path':str(dest/repo_id),'episodes':info['total_episodes'],'frames':info['total_frames']}),flush=True)
