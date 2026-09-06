#!/usr/bin/env python3

import warnings
import os
from termcolor import colored

import pickle
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder


warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
    
        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )
        if self.cfg.suite.task_make_fn.benchmarking:
            self.env, self.task_descriptions,self.init_states = hydra.utils.call(self.cfg.suite.task_make_fn) 
        else:
            self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)
            self.init_states = None
        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        # SAVE:add record flag
        self.record =  cfg.record
        
        if self.record:
            if not cfg.env_aug:
                self.play_seed = np.arange(100,101+self.cfg.suite.num_eval_episodes).tolist()
            else:
                print("using seed",cfg.record_seed)
                self.play_seed = np.random.randint(0,cfg.record_seed,size=(self.cfg.suite.num_eval_episodes)).tolist()
            self.success_num_save_dict = {i:0 for i in self.task_descriptions}
            self.success_num_save_dict['total_success'] = 0 
            if not os.path.exists(self.cfg.record_path):
                os.makedirs(self.cfg.record_path)
            
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def record_action(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        # recording 
        total_success = 0
        for env_idx in range(self.envs_till_idx):
            # save per environment 
            data= {}
            data['task_emb'] = np.zeros(384) 
            data['observations'] = []
            data['actions'] = []
            data['states'] = []
            print(f"recording env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

            success_save = 0
            while eval_until_episode(episode):
                # randomly initialized the environment
                self.env[env_idx].seed(self.play_seed[episode])
                time_step = self.env[env_idx].reset() 

                if "libero" in self.cfg.suite.name:
                    state = self.env[env_idx].get_sim_state()

                elif self.cfg.suite.name == "metaworld":
                    state = []

                self.agent.buffer_reset()
                step = 0
                # save_per_episode 
                
                obs_save = {}
                obs_save['pixels']= [] 
                obs_save['pixels_egocentric'] = []
                obs_save['joint_states'] = []
                obs_save['eef_states'] = []
                obs_save['gripper_states'] = []
                actions = []
                
                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx) 
                else:
                    prompt = None

                if episode == 0: 
                    data['task_emb'] = prompt['task_emb']
                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    # append if success
                    if "libero" in self.cfg.suite.name:
                        obs_save['pixels'].append(time_step.observation['pixels'].transpose(1,2,0).copy())
                        obs_save['pixels_egocentric'].append(time_step.observation['pixels_egocentric'].transpose(1,2,0).copy())
                        obs_save['joint_states'].append(time_step.observation['joint_states'])
                        obs_save['eef_states'].append(time_step.observation['eef_states'])
                        obs_save['gripper_states'].append(time_step.observation['gripper_states'])

                    elif self.cfg.suite.name == "metaworld":
                        obs_save['pixels'].append(time_step.observation['pixels'].transpose(1,2,0).copy())
                        state.append(time_step.observation['features'].copy())
                    actions.append(action)
                    time_step = self.env[env_idx].step(action)
                    # self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                    if self.cfg.suite.name == "calvin" and time_step.reward == 1:
                        self.agent.buffer_reset()
                assert len(actions) == len(obs_save['pixels'])

                if time_step.observation["goal_achieved"]:
                    for key,value in obs_save.items():
                        if len(obs_save[key]) == 0:
                            continue
                        obs_save[key] = np.stack(value,axis=0)
                    data['observations'].append(obs_save)
                    data['actions'].append(np.stack(actions,axis=0))
                    if isinstance(state,list):
                        assert len(state) > 0
                        state = np.stack(state,axis=0)
                    data['states'].append(state)
                    # data['']
                    success_save  += 1 
                    total_success += 1 
                episode += 1

            self.success_num_save_dict[self.task_descriptions[env_idx]] = success_save
            file_name = self.task_descriptions[env_idx] + '.pkl'
            save_data_path = os.path.join(self.cfg.record_path,file_name)
            with open(save_data_path,"wb") as f:
                pickle.dump(data,f)
            print("saving to",save_data_path)
        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)
        self.success_num_save_dict['total_success'] = total_success
        self.success_num_save_dict['play_list'] = self.play_seed
        with open(f"{self.work_dir}/save_result.txt", "w") as file:
            for key, value in self.success_num_save_dict.items():
                file.write(f"{key}: {value}\n")

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        if "ema" in "ema" in str(snapshots['bc']).split("/")[-2]:
            agent_payload = {key:value.state_dict() for key,value in agent_payload.items()}
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_record")
def main(cfg):
    from record import WorkspaceIL as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)
    workspace.record_action()


if __name__ == "__main__":
    main()
