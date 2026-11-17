"""CPU contracts for the entropy mapper and weighted FM reduction."""
import unittest
import csv,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tools.round_video.round_cost import entropy_weights
from rlinf.data.online_bc import masked_fm_loss

class Contracts(unittest.TestCase):
    def test_entropy_and_bounded_weights(self):
        torch.manual_seed(17)
        h,w=entropy_weights(torch.randn(3,8,50,14))
        self.assertEqual(h.shape,(3,50));self.assertTrue(torch.isfinite(h).all())
        self.assertGreaterEqual(float(w.min()),0);self.assertLessEqual(float(w.max()),5)

    def test_masked_fm_padding(self):
        loss=torch.ones(2,50,14);mask=torch.ones_like(loss,dtype=torch.bool);mask[:,20:]=False
        loss[:,20:]=1000
        self.assertEqual(float(masked_fm_loss(loss,mask,torch.ones(2,50))),1)

    def test_continuous_frame_action_query_mapping(self):
        from tools.round_video.capture import install
        class Writer:
            def __init__(self,*_):self.closed=False
            def isOpened(self):return True
            def write(self,_):pass
            def release(self):self.closed=True
        class Loop:
            def observations(self):return self.last_obs
            def step(self,request):
                for _ in range(3):self.env.take_action_cnt[0]+=1;self.observations()
            def serve(self):self.step({});self.step({})
        install(Loop)
        with tempfile.TemporaryDirectory()as d:
            loop=Loop();loop.cfg=dict(action_video_output=d,action_video_slots=[0])
            loop.env=SimpleNamespace(take_action_cnt=[0]);loop.last_obs={'main_images':np.zeros((1,8,8,3),np.uint8)}
            fake=SimpleNamespace(VideoWriter=Writer,VideoWriter_fourcc=lambda *_:0,cvtColor=lambda x,_:x,COLOR_RGB2BGR=0)
            with patch.dict(sys.modules,cv2=fake):loop.serve()
            rows=list(csv.DictReader((Path(d)/'D-B00-frames.csv').open()))
            self.assertEqual([(int(r['frame']),int(r['action_step']),int(r['query']))for r in rows],[(i,i+1,i//3)for i in range(6)])
            self.assertTrue(loop._video_writers[0].closed)

if __name__=='__main__':unittest.main()
