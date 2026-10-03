import json,os,socket,subprocess,time,xml.etree.ElementTree as ET,hashlib
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02'
R=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001');D=R/'gpu4567-fix-20261003';D.mkdir(exist_ok=True)
P=Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/expo-gpu4567-20261003.json')
def snapshot(pid=None):
 root=ET.fromstring(subprocess.check_output(['nvidia-smi','-x','-q'],text=True));rows=[]
 for index,g in enumerate(root.findall('gpu')):
  rows.append({'index':index,'uuid':g.findtext('uuid'),'minor':int(g.findtext('minor_number')),'processes':[{v.tag:v.text for v in p} for p in g.findall('processes/process_info') if pid is None or p.findtext('pid')==str(pid)]})
 return rows
before=snapshot();assert [g['minor'] for g in before]==list(range(8))
ids=[g['uuid'] for g in before if g['index'] in [4,5,6,7]]
profile={'rules':[{'pattern':{'feature':'commname','matches':'expo-gpu4567'},'profile':['EGLVisibleDGPUDevices',sum(1<<g['minor'] for g in before if g['index'] in [4,5,6,7])]}]}
assert not P.exists(), 'One-shot profile install: inspect before retry'
P.parent.mkdir(parents=True,exist_ok=True)
P.write_text(json.dumps(profile,indent=2)+'\n');os.chmod(P,0o600)
source=r'''import ctypes,json,os,time
from pathlib import Path
libc=ctypes.CDLL(None);assert libc.prctl(15,b'expo-gpu4567',0,0,0)==0
assert Path('/proc/self/comm').read_text().strip()=='expo-gpu4567'
def stage(name,**kw):
 print(json.dumps({'stage':name,'pid':os.getpid(),'time':time.time(),**kw}),flush=True);time.sleep(2)
stage('before_import')
import sapien
stage('sapien_imported',version=sapien.__version__)
r=sapien.render.RenderSystem('cuda:0')
stage('render_system',description=str(r))
scene=sapien.Scene([sapien.physx.PhysxCpuSystem(),r]);scene.set_timestep(1/240)
scene.set_ambient_light([.5,.5,.5]);scene.add_directional_light([0,1,-1],[1,1,1])
b=scene.create_actor_builder();b.add_box_visual(half_size=[.1,.1,.1]);a=b.build_static(name='probe-box');a.set_pose(sapien.Pose([1,0,0]))
cam=scene.add_camera('probe-camera',64,64,1.0,.01,10)
stage('scene_created')
for _ in range(3):scene.step()
scene.update_render();cam.take_picture();color=cam.get_picture('Color')
import numpy as np
assert color.shape==(64,64,4) and np.isfinite(color).all() and float(color[:,:,:3].max())>0
stage('render_verified',shape=list(color.shape),rgb_max=float(color[:,:,:3].max()),rgb_std=float(color[:,:,:3].std()))
'''
script=D/'profile-probe.py';assert not script.exists();script.write_text(source)
py='/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(ids),PYTHONDONTWRITEBYTECODE='1',__GL_APPLICATION_PROFILE='1',__GL_APPLICATION_PROFILE_LOG='1')
env.pop('DISPLAY',None)
log=D/'profile-probe.log';err=D/'profile-probe.err';samples=[];abort=None
with log.open('x') as f,err.open('x') as e:
 p=subprocess.Popen([py,'-u','-B',str(script)],env=env,stdout=f,stderr=e,start_new_session=True)
 start=time.monotonic()
 try:
  while p.poll() is None:
   snap=snapshot(p.pid);samples.append({'time':time.time(),'gpus':snap})
   if any(g['processes'] for g in snap[:4]):abort='probe PID touched GPU0-3';break
   used=sum(int(v['used_memory'].split()[0]) for g in snap for v in g['processes'] if v.get('used_memory','').endswith('MiB'))
   if used>2048:abort='probe VRAM > 2 GiB';break
   if time.monotonic()-start>120:abort='probe timeout';break
   time.sleep(.25)
 finally:
  if p.poll() is None:
   p.terminate()
   try:p.wait(10)
   except subprocess.TimeoutExpired:p.kill();p.wait(10)
 code=p.returncode
after=snapshot(p.pid)
result={'profile':str(P),'profile_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'mask':240,'allowed_uuids':ids,'pid':p.pid,'returncode':code,'abort':abort,'stdout':log.read_text(),'stderr':err.read_text()[-15000:],'samples':samples,'released':not any(g['processes'] for g in after)}
(D/'profile-probe-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
