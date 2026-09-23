import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch

path = Path(__file__).parents[2] / 'rlinf/algorithms/rlt/dv_observation.py'
spec = importlib.util.spec_from_file_location('observe', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_concentration_extremes():
    flat = module.concentration(torch.ones(10))
    assert abs(flat['top20_mass']-.2) < 1e-12
    assert abs(flat['ess_fraction']-1) < 1e-12
    spike = module.concentration([1]+[0]*9)
    assert spike['top20_mass']==1 and abs(spike['ess_fraction']-.1)<1e-12
    assert spike['entropy_normalized']==0


def test_collection_preserves_inputs_and_rng(tmp_path):
    def row(value, done, success):
        return SimpleNamespace(curr_obs={'teacher_dvac_v':torch.full((1,1,3,50),value),'episode_success':torch.tensor([success])}, dones=torch.tensor([done]))
    rows=[row(1.,False,True),row(3.,True,True),row(2.,True,False)]
    rng=torch.get_rng_state().clone();before=[r.curr_obs['teacher_dvac_v'].clone() for r in rows]
    m,episodes=module.summarize_collected(rows)
    assert m['dv_observe/all/mean']==2 and m['dv_observe/all/chunk_count']==3
    assert m['dv_observe/success/chunk_count']==2 and len(episodes)==2
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,r.curr_obs['teacher_dvac_v']) for a,r in zip(before,rows))
    module.record_collected(rows,output_dir=tmp_path,round_number=1,rank=0)
    assert (tmp_path/'dv_observations/rank_0000.jsonl').is_file()


def test_invalid_and_empty():
    rows=[SimpleNamespace(curr_obs={},dones=torch.tensor([True])),SimpleNamespace(curr_obs={'teacher_dvac_v':torch.full((1,1,3,50),float('nan'))},dones=torch.tensor([True]))]
    m,r=module.summarize_collected(rows)
    assert not r and m['dv_observe/missing_chunks']==1 and m['dv_observe/invalid_chunks']==1
    assert all(math.isfinite(v) for v in m.values())
