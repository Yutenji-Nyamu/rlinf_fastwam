"""CPU contract checks for the FastWAM geometry, replay, and 192-command evaluation."""
import io
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from omegaconf import OmegaConf

from rlinf.algorithms.expo_ft.formal_eval import evaluate
from rlinf.algorithms.expo_ft.formal_replay import C, D, H, FormalReplay, batch_ready
from rlinf.models.embodiment.fastwam.robotwin_adapter import compose_three_camera_image


def image_obs(batch):
    head = torch.arange(17*19*3).to(torch.uint8).reshape(1,17,19,3)
    wrist = torch.arange(13*9*3).to(torch.uint8).reshape(1,1,13,9,3)
    return dict(main_images=head.repeat(batch,1,1,1), wrist_images=wrist.repeat(batch,2,1,1,1),
                states=torch.zeros(batch,14), task_descriptions=['turn the switch']*batch)


class ModelStub:
    device = 'cpu'
    base_updates = 1

    def encode_executed(self, obs, actions):
        return actions

    def critic_observation(self, obs):
        return {'env_obs':obs}

    def sample_normalized(self, obs, num_candidates, generator):
        assert generator.device.type == 'cpu' and num_candidates == 8
        self.queries += 1
        return torch.zeros(4, 8, H, D)

    def decode(self, obs, actions):
        assert actions.shape == (4, C, D)
        return actions


def check_replay(root, backend):
    d=root/'demo'
    (d/'data/chunk-000').mkdir(parents=True)
    (d/'meta').mkdir()
    (d/'meta/tasks.jsonl').write_text('{"task_index":0,"task":"turn the switch"}\n')
    (d/'prepared.json').write_text('{"complete":true,"dataset_revision":"unit","episodes":1,"frames":32}')
    buf=io.BytesIO()
    Image.fromarray(np.zeros((8,12,3), dtype=np.uint8)).save(buf,format='PNG')
    pixels=[{'bytes':buf.getvalue()}]*H
    table=pa.table({'task_index':[0]*H,'action':[[float(i)]*D for i in range(H)],
                    'observation.state':[[0.]*D]*H,
                    **{'observation.images.cam_'+s:pixels for s in ['high','left_wrist','right_wrist']}})
    pq.write_table(table, d/'data/chunk-000/episode_000000.parquet')
    replay=FormalReplay(root/'replay',d)
    assert (C,H,D)==(24,32,14) and (replay.q_windows,replay.fm_windows)==(9,1)
    q=replay.sample(4,backend,'cpu')
    assert q['actions'].shape==(4,24,14) and q['executed_steps'].tolist()==[24]*4
    o,actions,counts=replay.sample_fm(2)
    assert actions.shape==(2,32,14) and actions[0,-1,0]==31 and counts=={'demo':2}
    rgb=image_obs(1)
    prepared=batch_ready(rgb)
    def compose(obs):
        return compose_three_camera_image(obs['main_images'],obs['wrist_images'],device='cpu',dtype=torch.float32)
    assert torch.equal(compose(rgb),compose(prepared))
    checkpoint = replay.state_dict()
    def finish_episode():
        return replay.append_episode([rgb], torch.zeros(1, D), [1.0], True, False,
                                     True, 'online-000000', final_obs=rgb)
    interrupted = finish_episode()
    replay.load_state_dict(checkpoint)
    assert not replay.online_entries
    committed = finish_episode()
    assert interrupted['path'] != committed['path']
    assert (replay.root / interrupted['path']).is_file()


def check_eval(root,backend):
    path=root/'seeds.json'
    path.write_text(json.dumps({'turn_switch':{'success_seeds':list(range(20))}}))
    cfg=dict(_expo_seed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),task_config=dict(task_name='turn_switch',step_lim=192),
             total_num_envs=4,rollout_epoch=5,max_episode_steps=192,max_steps_per_rollout_epoch=192,
             ignore_terminations=True,is_eval=True,video_cfg={})

    class EnvStub:
        cfg=OmegaConf.create({})
        expo_renderer_binding={}
        def reset(self,env_seeds):
            self.seedblock=env_seeds
            self.steps=0
            return image_obs(4),{}
        def step(self,command,auto_reset):
            assert not auto_reset and command.shape==(4,1,D)
            self.steps+=1
            assert self.steps<=192
            success=torch.tensor([s%2==0 and self.steps==192 for s in self.seedblock])
            return self.obs,None,success,torch.full((4,),self.steps==192),{'success':success}

    env=EnvStub()
    env.obs=image_obs(4)
    learner=SimpleNamespace(generator=torch.Generator(),update_calls=1,select_actions=lambda obs,base:{'actions':base[:,0]})
    backend.queries=0
    with patch('rlinf.algorithms.expo_ft.formal_eval.create_robotwin_env',return_value=env), \
         patch('rlinf.algorithms.expo_ft.formal_eval.close_robotwin_env'):
        r=evaluate(backend,learner,cfg,root/'eval',path)
    assert (r['successes'],r['episodes'],backend.queries)==(10,20,5*8)
    assert r['evaluation_physical_actions']==20*192 and all(row['physical_actions']==192 for row in r['rows'])


if __name__=='__main__':
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        model=ModelStub()
        check_replay(root,model)
        check_eval(root,model)
    print('PASS: official RGB equality; Q24/FM32 actual windows; 40 queries, 20 x 192 eval and last-step success')

