"""Focused CPU regressions for noise statistics and early/C20 gates."""
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from protocol import choose, execution_lengths, validate_config
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/pi05_signals'))
from signal_math import noise_disagreement,ugrow_10_vs_5
sys.path.insert(0,str(ROOT/'tools/pi05_u_video'))
from noise_samples import numpy_exact,repeated_observation
from calibration_pool import collect,deduplicate
from protocol import save


class FollowupTests(unittest.TestCase):
    def test_noise_archive_preserves_native_float64_actions(self):
        import torch
        action=torch.tensor([1.123456789012345],dtype=torch.float64)
        actual=numpy_exact(action)
        self.assertEqual(actual.dtype,np.dtype('float64'))
        self.assertEqual(actual[0],action.item())

    def test_multinoise_reduces_to_existing_u_for_two_samples(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(3,50,14));b=rng.normal(size=a.shape)
        np.testing.assert_allclose(noise_disagreement(np.stack([a,b],axis=1)).score,
                                   ugrow_10_vs_5(a,b).score,atol=1e-15,rtol=1e-14)

    def test_noise_signal_zero_identical_and_unit_balanced_signs(self):
        a=np.ones((2,4,50,14));np.testing.assert_array_equal(noise_disagreement(a).score,0)
        a[:,2:]*=-1;np.testing.assert_allclose(noise_disagreement(a).score,1,atol=2e-8)
        a.fill(0);np.testing.assert_array_equal(noise_disagreement(a).score,0)
        a[0,0,0,0]=np.nan;self.assertFalse(noise_disagreement(a).valid[0,0])

    def test_extra_batch_keeps_observation_slot_alignment(self):
        import torch
        obs={'states':torch.tensor([[10.],[20.]]),'images':np.array([[1],[2]]),
             'descriptions':['one','two'],'extra':None}
        r=repeated_observation(obs,3)
        self.assertEqual(r['states'].flatten().tolist(),[10,20]*3)
        self.assertEqual(r['descriptions'],['one','two']*3)
        np.testing.assert_array_equal(r['images'],np.array([[1],[2]]*3))
        self.assertIsNone(r['extra'])

    def test_current_first_two_can_switch_and_later_always_weak(self):
        active=np.array([True,True,False]);u=np.array([0.1,0.9,1.])
        for policy in ('high','low','random'):
            for q in range(4):
                selected=choose(policy,u,active,{'high':.5,'low':.5},np.array([.1,.2,.1]),
                                query=q,max_takeover_queries=2)
                self.assertFalse(selected[2])
                if q>=2:self.assertFalse(selected.any())
                else:self.assertTrue(selected.any())

    def test_c20_all_policies_share_fixed_twenty_action_schedule(self):
        c=dict(num_envs=32,step_limit=200,num_steps=10,execute_chunk=20,
               model={'openpi':{'action_chunk':50}},policy='high',takeover_mode='weights')
        validate_config(c)
        lengths=execution_lengths(np.array([True,False]),np.array([True,True]),
                                  np.array([180,180]),chunk=20,limit=200)
        self.assertEqual(lengths.tolist(),[20,20])
        c['takeover_mode']='c20'
        with self.assertRaises(AssertionError):validate_config(c)

    def test_calibration_filters_chunk_and_early_window_and_deduplicates(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)
            c=dict(num_steps=10,execute_chunk=50,takeover_mode='weights',
                   model={'model_path':'original','openpi':{'action_chunk':50}},
                   env={'task_config':{'task_name':'move_pillbottle_pad'}})
            ds=[dict(slot=0,actual_seed=1,query=q,start=q*50,end=(q+1)*50,
                     noise_seed=100+q,strong=False,u=.1*q)for q in range(4)]
            for name in ('copy_a','copy_b','wrong_chunk'):
                b=root/'batches'/name;save(b/'done.json',{})
                save(b/'config.json',dict(c,execute_chunk=20)if name=='wrong_chunk'else c)
                save(b/'decisions.json',ds)
            pool=collect([root],50,2);unique=deduplicate([pool])
            self.assertEqual(unique['raw_decisions'],4)
            self.assertEqual(unique['unique_decisions'],2)
            self.assertEqual({r['query']for r in unique['rows']},{0,1})

if __name__=='__main__':unittest.main()
