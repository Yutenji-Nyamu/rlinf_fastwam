"""Reuse the pinned RoboTwin clean50 conversion functions for move_stapler_pad."""
import os,json,zipfile,importlib.util,random,sys,types
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
ROOT=Path('/data/chenyiteng/datasets/robotwin2')
REV='9dc9299c163db059931898a9f0852098a61155a1'
TASK='move_stapler_pad'
archive=ROOT/'source'/REV/f'dataset/{TASK}/aloha-agilex_clean_50.zip'
if not archive.is_file():
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
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
inter=ROOT/'intermediate'/REV/TASK/'clean50-20260919'
if not inter.exists():
 process=module('raw_to_aloha',tool/'scripts/process_data.py');process.data_transform(str(raw),50,str(inter))
dest=ROOT/'canonical';repo_id='move-stapler-pad-aloha-clean50-20260919';assert not (dest/repo_id).exists()
os.environ['HF_LEROBOT_HOME']=str(dest)
source=(tool/'examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py').read_text()
# Installed LeRobot preserves dataset layout; only import and task API moved.
source=source.replace('from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME',f'HF_LEROBOT_HOME = Path({str(dest)!r})')
source=source.replace('from lerobot.common.datasets.lerobot_dataset import LeRobotDataset','from lerobot.datasets.lerobot_dataset import LeRobotDataset')
assert source.count('dataset.add_frame(frame)')==1
source=source.replace('dataset.add_frame(frame)',"dataset.add_frame(frame, task=frame.pop('task'))")
converter=types.ModuleType('aloha_to_lerobot');sys.modules[converter.__name__]=converter;exec(compile(source,'pinned_converter_compat','exec'),converter.__dict__)
random.seed(0);np.random.seed(0)
files=sorted(inter.rglob('episode_*.hdf5'),key=lambda p:int(p.stem.split('_')[-1]));assert len(files)==50
d=converter.create_empty_dataset(repo_id,robot_type='aloha',mode='image',has_effort=converter.has_effort(files),has_velocity=converter.has_velocity(files),dataset_config=converter.DEFAULT_DATASET_CONFIG)
converter.populate_dataset(d,files,task='move the stapler onto the pad',episodes=list(range(50)))
info=json.loads((dest/repo_id/'meta/info.json').read_text());assert info['total_episodes']==50
print(json.dumps({'complete':True,'path':str(dest/repo_id),'episodes':info['total_episodes'],'frames':info['total_frames']}),flush=True)
