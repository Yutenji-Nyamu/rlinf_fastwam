import unittest
import numpy as np
from teacher_followup import gate
class Contracts(unittest.TestCase):
    def test_first_three_only_and_active(self):
        x=np.arange(10.);active=np.ones(10,bool);active[-1]=False
        self.assertEqual(gate(x,'high',4.5,2,active).sum(),4)
        self.assertFalse(gate(x,'high',4.5,3,active).any())
    def test_deciles_and_complement(self):
        x=np.arange(100.);active=np.ones(100,bool)
        for p in range(10,100,10):
            self.assertEqual(gate(x,'high',np.quantile(x,1-p/100),0,active).sum(),p)
            self.assertEqual(gate(x,'low',np.quantile(x,p/100),0,active).sum(),p)
    def test_rmse_and_equal_bins(self):
        delta=np.full((2,20,14),2.);np.testing.assert_array_equal(np.sqrt((delta**2).mean(-1)),2)
        self.assertEqual([len(x)for x in np.array_split(np.arange(40),10)],[4]*10)
if __name__=='__main__':unittest.main()
