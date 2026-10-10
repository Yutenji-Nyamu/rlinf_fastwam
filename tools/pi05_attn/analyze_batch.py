"""Reconstruct once per saved query; generate complete episode-indexed atlas."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import cv2
from analysis import reconstruct,GROUPS
def run(config):
 c=json.loads(Path(config).read_text());out=Path(c['output']);done=json.loads((out/'done.json').read_text());cost=[];atlas=[];errors=[]
 (out/'details').mkdir(exist_ok=True);(out/'frames').mkdir(exist_ok=True)
 episodes=json.loads((out/'episodes.json').read_text())
 # Reuse the complete decode and frame-count check from the accepted 2026-10-03 validator.
 video_check_start=time.perf_counter();decoded_videos=[]
 frame_times=json.loads((out/'frame_times.json').read_text())
 for e in episodes:
  cap=cv2.VideoCapture(str(out/e['video']));decoded=0
  try:
   assert cap.isOpened(),('cannot open video',e['video'])
   while True:
    ok,frame=cap.read()
    if not ok:break
    assert frame is not None and frame.ndim==3
    decoded+=1
  finally:cap.release()
  assert decoded==e['frames']==len(frame_times[e['slot']])==done['queries']+1
  decoded_videos.append(dict(slot=e['slot'],file=e['video'],decoded_frames=decoded))
 video_check_seconds=time.perf_counter()-video_check_start
 torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
 paths=sorted(out.glob('query_*.pt'));assert len(paths)==done['queries'] and [p.stem for p in paths]==[f'query_{i:03d}' for i in range(done['queries'])], 'Incomplete raw query sequence'
 for path in paths:
  t=time.perf_counter();raw=torch.load(path,map_location='cpu',weights_only=False);meta=json.loads(path.with_suffix('.json').read_text())
  assert raw['q_action'].shape[:5]==(c['num_envs'],10,3,8,50) and raw['k_prefix'].shape[1:3]==(3,1)
  scores,refs=reconstruct(raw,device);np.savez_compressed(path.with_name(path.stem+'_scores.npz'),**scores)
  active=raw['active'].numpy();obs=np.load(path.with_name(path.stem+'_obs.npz'))
  for slot in np.where(active)[0]:
   row=dict(query=meta['query'],slot=int(slot),requested_seed=episodes[slot]['requested'],actual_seed=episodes[slot]['actual'],native_retry=episodes[slot]['native_retry'],success=bool(raw['success_after'][slot]),executed_known=not bool(raw['success_after'][slot]),submitted=raw['submitted_mask'][slot].tolist())
   for key in ['E_obs',*[f'E_{g}' for g in GROUPS],*[f'M_{g}' for g in GROUPS],'D_span','B_span','I_anchor','I_future',*[f'B_{g}' for g in GROUPS]]:
    row[key]=[round(float(v),6) if np.isfinite(v) else None for v in scores[key][slot,-1]]
   row['action_amplitude']=scores['action_amplitude'][slot].tolist();row['gripper_left']=raw['env_action'][slot,:,6].float().tolist();row['gripper_right']=raw['env_action'][slot,:,13].float().tolist();row['self_read']=scores['self_read'][slot,-1].tolist();row['M_other']=scores['M_other'][slot,-1].tolist();row['source_token_counts']={g:int(scores['N_'+g][slot,-1,0]) for g in GROUPS}
   row['dv']=[float(v) for v in scores['dv'][slot]];row['layer_entropy']=np.round(scores['layer_entropy'][slot,-1],6).tolist()
   row['denoise_visual_entropy']=np.round(scores['E_visual'][slot],6).tolist()
   atlas.append(row)
   stem=f'q{meta["query"]:03d}_s{slot:02d}'
   details=dict(aa=scores['action_attention_1_5_10'][slot,-1].tolist(),layer=scores['layer_entropy'][slot,-1,:,:,0].tolist(),denoise=scores['E_visual'][slot].tolist(),head=scores['head_entropy'][slot,-1,-1,:,:,0].tolist())
   (out/'details'/(stem+'.json')).write_text(json.dumps(details).replace('NaN','null'))
   for camera,im in [('head',obs['main_images'][slot]),('left',obs['wrist_images'][slot,0]),('right',obs['wrist_images'][slot,1])]:
    cv2.imwrite(str(out/'frames'/(stem+'_'+camera+'.jpg')),cv2.cvtColor(cv2.resize(im,(320,240)),cv2.COLOR_RGB2BGR))
  cost.append(dict(query=meta['query'],seconds=time.perf_counter()-t,score_bytes=path.with_name(path.stem+'_scores.npz').stat().st_size))
  obs.close();del raw,scores
 (out/'atlas-data.json').write_text(json.dumps(atlas,allow_nan=True,separators=(',',':')))
 # Missing entropy diagnostics in JSON use null, never a fabricated zero.
 text=(out/'atlas-data.json').read_text().replace('NaN','null');(out/'atlas-data.json').write_text(text)
 episodes=json.loads((out/'episodes.json').read_text());seeds=[r['actual'] for r in episodes]
 report=dict(passed=True,video_passed=True,video_frames=sum(v['decoded_frames'] for v in decoded_videos),decoded_videos=decoded_videos,video_check_seconds=video_check_seconds,time=time.time(),queries=len(cost),episodes=len(episodes),successes=done['successes'],unique_actual_seeds=len(set(seeds)),seed_duplicate_warning=len(set(seeds))<c['num_envs'],offline=cost,interpretation='Recording and offline reconstruction only; no training evidence')
 (out/'validation.json').write_text(json.dumps(report,indent=2))
 template=Path(__file__).with_name('atlas.html').read_text()
 (out/'index.html').write_text(template.replace('__TITLE__',c['task']+' · batch '+str(c['batch'])))
 print(json.dumps(report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('config');config=ap.parse_args().config
 try:run(config)
 except BaseException:
  c=json.loads(Path(config).read_text());(Path(c['output'])/'error.json').write_text(json.dumps(dict(time=time.time(),phase='offline',error=traceback.format_exc())));raise
