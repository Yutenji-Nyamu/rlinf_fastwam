"""CPU checks for prefix selection and independent replay draws per update."""
from types import SimpleNamespace
import torch
from rlinf.models.embodiment.modules.dsrl_na import last_valid_prefix, append_vla_features
from rlinf.workers.actor.fsdp_dsrl_na_policy_worker import EmbodiedDSRLNAFSDPPolicy


def test_prefix_selection():
    prefix=torch.arange(2*5*2048).reshape(2,5,2048).float().requires_grad_()
    mask=torch.tensor([[1,1,0,1,0],[1,0,0,0,0]],dtype=torch.bool)
    out=last_valid_prefix(prefix,mask)
    torch.testing.assert_close(out,torch.stack([prefix[0,3],prefix[1,0]]))
    assert not out.requires_grad
    m=SimpleNamespace(config=SimpleNamespace(dsrl_vla_features=True))
    combined=append_vla_features(m,{'vla_features':out},torch.zeros(2,64))
    torch.testing.assert_close(combined[:,64:],out)


def test_each_update_draws_replay():
    class Replay:
        calls=0
        def sample(self,n):
            self.calls+=1
            rows=torch.full((n,1),float(self.calls))
            return dict(actions=rows,env_actions=rows,curr_vla={'rows':rows},next_vla={'rows':rows},
                        curr_obs={'states':rows},next_obs={'states':rows})
    class Model:
        def eval(self):pass
        def train(self):pass
    w=SimpleNamespace(replay_buffer=Replay(),cfg=SimpleNamespace(actor=SimpleNamespace(global_batch_size=4)),
        device='cpu',model=Model(),na_replay_decode_error={},enable_drq=False,
        _decode=lambda packed,noise:packed['rows'])
    for i,kind in enumerate(['action_q','action_q','noise_q','noise_q','actor'],1):
        out=EmbodiedDSRLNAFSDPPolicy._fresh_batch(w,True,kind)
        assert out['curr_obs']['states'].eq(i).all()
        if kind=='action_q':assert out['next_env_actions'].eq(i).all()
        if kind=='noise_q':assert out['distill_actions'].eq(i).all()
    assert w.replay_buffer.calls==5


def test_state_token_refresh():
    import numpy as np
    from openpi.models.tokenizer import PaligemmaTokenizer
    from rlinf.models.embodiment.modules.dsrl_na import model_operation
    tokenizer=PaligemmaTokenizer(200)
    instruction='Turn the switch.'
    old,oldmask=tokenizer.tokenize(instruction,np.zeros(32))
    expected,expectedmask=tokenizer.tokenize(instruction,np.ones(32))
    assert not np.array_equal(old,expected)
    def processor(raw):
        return {'prompt':raw['task_descriptions'],'state':raw['states']}
    def transform(raw,transpose):
        if 'prompt' in raw:
            pairs=[tokenizer.tokenize(t,s.numpy()) for t,s in zip(raw['prompt'],raw['state'])]
            toks=torch.from_numpy(np.stack([x[0] for x in pairs]))
            mask=torch.from_numpy(np.stack([x[1] for x in pairs]))
        else:toks,mask=raw['tokenized_prompt'],raw['tokenized_prompt_mask']
        return {'image':{'base':torch.zeros(1,4,4,3,dtype=torch.uint8)},
            'tokenized_prompt':toks,'tokenized_prompt_mask':mask,'state':raw['state']}
    model=SimpleNamespace(config=SimpleNamespace(dsrl_na_enabled=True),obs_processor=processor,input_transform=transform)
    value=model_operation(model,'pack',dict(states=torch.ones(1,32),task_descriptions=[instruction]),
        tokens=dict(tokenized_prompt=torch.from_numpy(old[None]),tokenized_prompt_mask=torch.from_numpy(oldmask[None])))
    np.testing.assert_array_equal(value['tokenized_prompt'].numpy()[0],expected)
    np.testing.assert_array_equal(value['tokenized_prompt_mask'].numpy()[0],expectedmask)
    print('changed next-state token positions',int((old!=expected).sum()))


if __name__=='__main__':
    torch.set_num_threads(2)
    for test in [test_prefix_selection,test_each_update_draws_replay,test_state_token_refresh]:
        test();print(test.__name__,'PASS',flush=True)
