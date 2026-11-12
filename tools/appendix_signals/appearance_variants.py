"""Fixed-state RoboTwin appearance probes with official MCR features.

The action replay and physical states stay fixed. Interventions use RoboTwin's
lighting, camera translation, and background-texture mechanisms. All fixed
anchors are retained; ratios below one are retained too.
"""
import argparse,json,sys,os,time,copy
from pathlib import Path
import numpy as np
from tools.appendix_signals.common import cuda_guard,write_json,write_npz
from tools.appendix_signals.rt_appearance import physics_state,views

def main(args):
 torch=cuda_guard(args.gpu);sys.path.insert(0,args.model_source)
 from omegaconf import OmegaConf
 from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
 import sapien
 from PIL import Image
 from torchvision import transforms
 from torchvision.models import resnet50
 out=Path(args.output)/'grbc-variants';out.mkdir(parents=True,exist_ok=True);root=Path(args.data)
 pool=json.loads((root/'rt-pool/queries.json').read_text());nodes=[];errors=[]
 assets=Path(os.environ['ROBOTWIN_ASSETS_PATH'])/'assets/background_texture/unseen'
 textures=sorted(assets.glob('*.png'))
 assert len(textures)>=2,'Official background textures missing'
 fixedtextures=[textures[0],textures[len(textures)//2]]
 variants=['base','warm_light','dim_light','camera_5cm','texture_1','texture_2']
 if args.native_rand:variants=['base']+[f'native_rand_{i}' for i in range(4)]
 for task_name in ['turn_switch','stack_bowls_two']:
  eps=[]
  for row in pool:
   if row['task']==task_name and row['seed'] not in [e['seed'] for e in eps]:eps.append(row)
  eps=eps[:4];source_root=Path(eps[0]['batch']).parent.parent
  cfg=json.loads((source_root/'configs'/f'{task_name}_b0.json').read_text())['env']
  cfg.update(total_num_envs=4,seed=0,auto_reset=False,ignore_terminations=True)
  seeds=out/f'{task_name}_seeds.json';write_json(seeds,{task_name:{'success_seeds':[e['seed'] for e in eps]}})
  cfg['seeds_path']=str(seeds);cfg['task_config']['save_path']=str(out/'native'/task_name);cfg['video_cfg']['video_base_dir']=str(out/'native'/task_name/'unused')
  torch.empty(1,device='cuda');env=RoboTwinEnv(OmegaConf.create(cfg),4,0,1,None)
  try:
   env.reset();actual=[int(s.task.ep_num) for s in env.venv.envs];assert set(actual)=={e['seed'] for e in eps};byseed={e['seed']:e for e in eps}
   desired={seed:{r['query'] for r in pool if r['task']==task_name and r['seed']==seed} for seed in actual}
   source=Path(eps[0]['batch'])
   for q in range(max(max(x) for x in desired.values())+1):
    for slot,sub in enumerate(env.venv.envs):
     seed=actual[slot]
     if q not in desired[seed]:continue
     task=sub.task;before=physics_state(task);ambient=np.array(task.scene.ambient_light).copy();camera=task.cameras.static_camera_list[task.cameras.head_camera_id];pose=camera.entity.get_pose()
     visual_bodies=[]
     for entity in [task.table,task.wall]:
      entity=getattr(entity,'actor',entity)
      body=entity.find_component_by_type(sapien.render.RenderBodyComponent)
      visual_bodies.append((entity,body))
     lights=task.direction_light_lst+task.point_light_lst;light_colors=[np.array(light.get_color()).copy() for light in lights]
     baseline=views(sub.get_obs());images=[];valid=[];specs=[]
     for variant in variants:
      replacements=[]
      spec={}
      try:
       if variant=='warm_light':task.scene.set_ambient_light([.65,.45,.30])
       elif variant=='dim_light':task.scene.set_ambient_light([.20,.20,.20])
       elif variant=='camera_5cm':camera.entity.set_pose(sapien.Pose(pose.p+np.array([.05,0,0]),pose.q))
       elif variant.startswith('native_rand_'):
        draw_seed=202611120+seed+int(variant[-1])*1000;rng=np.random.RandomState(draw_seed)
        colors=[rng.rand(3) for _ in lights]
        for light,color in zip(lights,colors):light.set_color(color)
        wall_id,table_id=int(rng.randint(len(textures))),int(rng.randint(len(textures)))
        wall_id=None if rng.rand()<=.02 else wall_id;table_id=None if rng.rand()<=.02 else table_id
        spec={'rng_seed':draw_seed,'wall_texture':wall_id,'table_texture':table_id,'light_colors':[x.tolist() for x in colors],'ambient':[.5,.5,.5],'camera_displacement':0,'sampling':'native static random_light and random_background rules; demo_randomized clean_background_rate=.02, random_head_camera_dis=0'}
        for (entity,body),texture_id in zip(visual_bodies,[table_id,wall_id]):
         if texture_id is None:continue
         texpath=assets/f'{texture_id}.png';assert texpath.is_file();mat=sapien.render.RenderMaterial();mat.set_base_color_texture(sapien.render.RenderTexture2D(str(texpath)));mat.base_color=[1,1,1,1];mat.metallic=.1;mat.roughness=.3;new=sapien.render.RenderBodyComponent()
         for oldshape in body.render_shapes:
          shape=sapien.render.RenderShapeBox(oldshape.half_size,mat);shape.local_pose=oldshape.local_pose;new.attach(shape)
         entity.remove_component(body);entity.add_component(new);replacements.append((entity,body,new))
       elif variant.startswith('texture_'):
        texture=sapien.render.RenderTexture2D(str(fixedtextures[int(variant[-1])-1]));mat=sapien.render.RenderMaterial();mat.set_base_color_texture(texture);mat.base_color=[1,1,1,1];mat.metallic=.1;mat.roughness=.3
        for entity,body in visual_bodies:
         new=sapien.render.RenderBodyComponent()
         for oldshape in body.render_shapes:
          shape=sapien.render.RenderShapeBox(oldshape.half_size,mat);shape.local_pose=oldshape.local_pose;new.attach(shape)
         entity.remove_component(body);entity.add_component(new);replacements.append((entity,body,new))
       rgb=views(sub.get_obs());np.testing.assert_array_equal(physics_state(task),before)
       images.append(rgb);valid.append(variant);specs.append({'texture':fixedtextures[int(variant[-1])-1].name} if variant.startswith('texture_') else spec)
      except (AttributeError,TypeError,RuntimeError) as e:
       errors.append({'task':task_name,'seed':seed,'query':q,'variant':variant,'error':str(e)})
      finally:
       task.scene.set_ambient_light(ambient);camera.entity.set_pose(pose)
       for light,color in zip(lights,light_colors):light.set_color(color)
       for entity,body,new in replacements:entity.remove_component(new);entity.add_component(body)
     restored=views(sub.get_obs());np.testing.assert_array_equal(physics_state(task),before)
     delta=int(np.max(np.abs(restored.astype(int)-baseline.astype(int))))
     pair=f'{task_name}_{seed}_q{q:03d}';raw=out/'pairs'/f'{pair}.npz';state=np.asarray(sub.get_obs()['state'])
     write_npz(raw,images=np.stack(images),state=state,physics_state=before)
     for i,variant in enumerate(valid):nodes.append({'pair':pair,'task':task_name,'seed':seed,'query':q,'variant':variant,'appearance':i,'raw':'pairs/'+raw.name,'state_exact':True,'restoration_rgb_delta':delta,**specs[i]})
     write_json(out/'nodes.json',nodes);write_json(out/'errors.json',errors)
    with np.load(source/f'query_{q:03d}.npz',allow_pickle=False) as z:actions=np.stack([z['noise4_env_actions'][byseed[seed]['slot'],0] for seed in actual])
    env.chunk_step(torch.from_numpy(actions));print(task_name,q,len(nodes),flush=True)
  finally:env.offload()
 model=resnet50(weights=None);model.fc=torch.nn.Identity();model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False).cuda()
 transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]);encoded=[];grips=[]
 for lo in range(0,len(nodes),24):
  tensors=[]
  for node in nodes[lo:lo+24]:
   with np.load(out/node['raw'],allow_pickle=False) as z:rgb=z['images'][node['appearance']];state=z['state']
   tensors.extend(transform(Image.fromarray(x)) for x in rgb);grips.append([bool(state[6]<.5),bool(state[13]<.5)])
  with torch.inference_mode():encoded.append(model(torch.stack(tensors).cuda()).float().cpu().numpy())
 raw=np.concatenate(encoded).reshape(len(nodes),3,2048);normalized=raw/np.maximum(np.linalg.norm(raw,axis=-1,keepdims=True),1e-8)
 write_npz(out/'embeddings.npz',raw=raw,per_view=normalized,gripper_closed=np.asarray(grips))
 write_json(out/'complete.json',{'nodes':len(nodes),'anchors':len({x['pair'] for x in nodes}),'variants':variants,'errors':len(errors),'time':time.time(),'texture_files':[str(x) for x in fixedtextures],'all_anchors_retained':True,'native_rand':args.native_rand,'official_config_url':'https://robotwin-platform.github.io/doc/usage/configurations.html','protocol':'native static appearance sampling; fixed physical state and gripper gate' if args.native_rand else 'controlled appearance interventions'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--model-source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',type=int,default=4);p.add_argument('--native-rand',action='store_true');main(p.parse_args())
