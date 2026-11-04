"""Full-noise + U retains all coordinates, U provenance, and strict resume."""
import pytest,torch
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.dsrl_transition import DSRLTransitionReplayBuffer,project_dsrl_trajectory
from rlinf.algorithms.dsrl_ugrow import make_signal_spec,build_dsrl_u_weights

def test_full_u_projection_and_resume(tmp_path):
    spec=make_signal_spec(chunk_length=20)
    noise=torch.randn(1,2,50,32)
    done=torch.zeros(1,2,20,dtype=torch.bool)
    obs=dict(main_images=torch.zeros(1,2,64,64,3,dtype=torch.uint8),states=torch.zeros(1,2,14))
    scores=torch.linspace(.01,.8,40).reshape(1,2,20)
    trajectory=Trajectory(actions=noise,rewards=torch.zeros(1,2,20),terminations=done,truncations=done,dones=done,curr_obs=obs,next_obs=obs,forward_inputs=dict(dsrl_u=scores,dsrl_u_valid=torch.ones_like(scores,dtype=torch.bool),dsrl_u_policy_step=torch.zeros(1,2,1,dtype=torch.int64),dsrl_u_phase=torch.zeros(1,2,1,dtype=torch.int64)))
    batch=project_dsrl_trajectory(trajectory,action_horizon=50,latent_dim=1600,state_dim=14,num_action_chunks=20,gamma=.999,u_spec=spec,full_noise=True)
    assert torch.equal(batch['actions'],noise.reshape(2,1600).bfloat16())
    assert torch.equal(batch['dsrl_u'],scores.reshape(2,20))
    kw=dict(capacity=8,rank=0,world_size=1,seed=42,u_spec=spec)
    replay=DSRLTransitionReplayBuffer(schema_version=4,**kw);replay.add_batch(batch);replay.save_checkpoint(str(tmp_path))
    restored=DSRLTransitionReplayBuffer(schema_version=4,**kw);restored.load_checkpoint(str(tmp_path))
    assert restored.get_stats()['resident_transitions']==2
    with pytest.raises(ValueError):DSRLTransitionReplayBuffer(schema_version=2,**kw).load_checkpoint(str(tmp_path))
    with pytest.raises(ValueError):DSRLTransitionReplayBuffer(schema_version=3,**kw)
    weight,_=build_dsrl_u_weights(batch['dsrl_u'],batch['dsrl_u_valid'],temperature=2.5)
    assert weight.shape==(2,1) and torch.allclose(weight.mean(),torch.tensor(1.))
