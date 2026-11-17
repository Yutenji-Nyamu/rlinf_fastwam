"""CPU contracts for the entropy mapper and weighted FM reduction."""
import unittest
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

if __name__=='__main__':unittest.main()
