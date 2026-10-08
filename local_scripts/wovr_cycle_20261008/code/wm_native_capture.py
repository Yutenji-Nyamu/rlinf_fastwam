"""Opt-in synchronized capture of the unchanged RoboTwin chunk controller.

Video slots are controller-progress eighths, NOT fictitious per-command physics
steps. Requested C32 actions and every interpolated control target are separate.
"""
import inspect
import json
import os
from pathlib import Path
import textwrap
import time
import uuid
import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")


def snapshot(task):
    obs = task.get_obs()
    return ([np.asarray(obs["observation"][k]["rgb"], dtype=np.uint8).copy() for k in CAMERAS],
            np.asarray(obs["joint_action"]["vector"], dtype=np.float32).copy(),
            bool(task.check_success()))


class Capture:
    def __init__(self, sub):
        self.sub, self.task = sub, sub.task
        root = Path(os.environ["WM_NATIVE_CAPTURE_DIR"])
        self.path = root / (str(os.getpid()) + "_" + str(sub.env_id) + "_" + uuid.uuid4().hex[:12])
        self.path.mkdir(parents=True)
        self.chunks, self.count, self.finished = [], 0, False
        self.started = time.time()
        self.meta = dict(schema=1, task=sub.task_name, requested_seed=int(sub.env_seed),
            actual_seed=int(self.task.ep_num), instruction=sub.instruction,
            policy=os.environ.get("WM_CAPTURE_POLICY", "original_pi05"),
            complete=False, success=False, horizon=int(self.task.step_lim),
            video_timebase="normalized TOPP controller progress; eighths within each requested C32",
            action_timebase="requested policy C32; controller_commands records all executed targets",
            camera_order=list(CAMERAS))
        self.save_meta()

    def save_meta(self):
        value = dict(self.meta, chunks=self.chunks, elapsed_seconds=time.time()-self.started)
        tmp=self.path/'episode.json.tmp'; tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(self.path/'episode.json')

    def begin(self, actions):
        self.request = np.asarray(actions, dtype=np.float32).copy()
        assert self.request.shape == (32,14), self.request.shape
        self.frames=[snapshot(self.task)]
        self.progress=[0.0]
        self.controller=[]
        self.controller_progress=[]
        self.next_slot=1
        self.chunk_start=time.perf_counter()

    def tick(self, left, right):
        progress=min(float(left),float(right))
        q=np.asarray(self.task.robot.get_left_arm_jointState()+self.task.robot.get_right_arm_jointState(),dtype=np.float32)
        self.controller.append(q.copy()); self.controller_progress.append([left,right])
        if progress+1e-9 >= self.next_slot/8:
            frame=snapshot(self.task)
            while self.next_slot<=8 and progress+1e-9 >= self.next_slot/8:
                self.frames.append(frame); self.progress.append(progress); self.next_slot+=1

    def end(self, result):
        terminal=snapshot(self.task)
        success=bool(np.asarray(result['info'].get('success',False)).any())
        self.meta['success'] = bool(self.meta['success'] or success)
        real_slots=len(self.frames)
        # Always retain the terminal contact frame; following entries are padding.
        if len(self.frames)<9:
            self.frames.append(terminal); self.progress.append(self.controller_progress[-1][0] if self.controller_progress else 0.)
            real_slots=len(self.frames)
        else:
            self.frames[-1]=terminal
        while len(self.frames)<9:
            self.frames.append(terminal); self.progress.append(self.progress[-1])
        image_is_pad=np.arange(9)>=real_slots
        name=f'chunk_{self.count:03d}.npz'
        np.savez(self.path/name,
            head=np.stack([x[0][0] for x in self.frames]),
            left=np.stack([x[0][1] for x in self.frames]),
            right=np.stack([x[0][2] for x in self.frames]),
            state=np.stack([x[1] for x in self.frames]),
            success=np.asarray([x[2] for x in self.frames],dtype=bool),
            action=self.request, image_is_pad=image_is_pad,
            controller_commands=np.asarray(self.controller,dtype=np.float32).reshape(-1,14),
            controller_progress=np.asarray(self.controller_progress,dtype=np.float32).reshape(-1,2),
            frame_progress=np.asarray(self.progress,dtype=np.float32))
        self.chunks.append(dict(file=name, native_seconds=time.perf_counter()-self.chunk_start,
            executed_controller_steps=len(self.controller), valid_frames=real_slots,
            requested_action_start=self.count*32, success=success))
        self.count+=1
        self.meta['requested_actions']=self.count*32
        self.finished=success or bool(np.asarray(result['truncated']).any()) or self.count*32>=self.meta['horizon']
        self.meta['complete']=self.finished
        self.save_meta()


def install():
    if not os.environ.get('WM_NATIVE_CAPTURE_DIR'):
        return
    from envs._base_task import Base_Task
    from robotwin.envs.vector_env import SubEnv, VectorEnv
    if getattr(SubEnv,'_wm_capture_installed',False):
        return
    # Extra synchronized camera capture takes 82-135 s at N64; the inherited
    # 120 s Future deadline cut off healthy trajectories. Recording only.
    vector_step=VectorEnv.step
    step_source=textwrap.dedent(inspect.getsource(vector_step))
    assert step_source.count('future.result(timeout=120)')==1
    step_source=step_source.replace('future.result(timeout=120)','future.result(timeout=600)')
    scope={};exec(compile(step_source,inspect.getfile(vector_step),'exec'),vector_step.__globals__,scope)
    VectorEnv.step=scope['step']
    original=Base_Task.gen_sparse_reward_data
    source=textwrap.dedent(inspect.getsource(original))
    needle='        steps_executed += 1'
    assert source.count(needle)==1, 'Native controller changed; update the one recording hook'
    source=source.replace(needle,needle+'\n        if getattr(self, "_wm_capture", None) is not None:\n            self._wm_capture.tick(now_left_id / left_n_step, now_right_id / right_n_step)')
    namespace={}; exec(compile(source,inspect.getfile(original),'exec'),original.__globals__,namespace)
    Base_Task.gen_sparse_reward_data=namespace['gen_sparse_reward_data']
    reset,step=SubEnv.reset,SubEnv.step
    def recording_reset(self,*args,**kwargs):
        result=reset(self,*args,**kwargs)
        self.task._wm_capture=None
        return result
    def recording_step(self,actions):
        if getattr(self.task,'eval_success',False):
            return step(self,actions)
        cap=getattr(self.task,'_wm_capture',None)
        if cap is None:
            cap=Capture(self); self.task._wm_capture=cap
        cap.begin(actions)
        result=step(self,actions)
        cap.end(result)
        return result
    SubEnv.reset,SubEnv.step=recording_reset,recording_step
    SubEnv._wm_capture_installed=True
