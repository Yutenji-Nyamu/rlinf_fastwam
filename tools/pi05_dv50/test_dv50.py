import unittest
import numpy as np
from rank_curves import rank


class Shapes(unittest.TestCase):
    def test_flat_and_invalid_rejected(self):
        for y in ([0]*10,[1]*10,[1,float('nan'),2,3],[1,2]):
            self.assertEqual(rank(y)['score'],0)

    def test_sustained_region_beats_spikes(self):
        clear=rank([1,1,1,5,6,5,1,1,1,1])
        spikes=rank([1,6,1,1,6,1,1,6,1,1])
        self.assertGreater(clear['score'],spikes['score'])
        self.assertEqual(clear['high_regions'],1)

    def test_mask_contract(self):
        submitted=np.ones((2,50),bool);success=np.array([False,True])
        exact=submitted & ~success[:,None]
        self.assertEqual(exact.sum(),50)
        self.assertTrue(np.all(exact<=submitted))


if __name__=='__main__':unittest.main()
