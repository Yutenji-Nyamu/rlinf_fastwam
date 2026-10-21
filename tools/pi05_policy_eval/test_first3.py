"""Verify the new third-chunk boundary without changing native execution."""
import unittest
import numpy as np
from protocol import choose, validate_config

class First3Tests(unittest.TestCase):
    def test_third_chunk_allowed_fourth_forbidden_for_both_rules(self):
        u=np.array([.1,.9]); active=np.array([True,True])
        for rule, expected in [('high',[False,True]), ('low',[True,False])]:
            for q in range(10):
                got=choose(rule,u,active,{rule:.5},None,query=q,max_takeover_queries=3)
                self.assertEqual(got.tolist(),expected if q<3 else [False,False])
        validate_config(dict(num_envs=32,step_limit=200,num_steps=10,execute_chunk=20,
            model={'openpi':{'action_chunk':50}},policy='high',max_takeover_queries=3))

if __name__=='__main__':unittest.main()
