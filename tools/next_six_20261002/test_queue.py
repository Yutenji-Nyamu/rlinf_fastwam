"""Targeted CPU-only checks for foreign cutovers and borrowed Stage1 GPUs."""
import unittest
from unittest.mock import patch
import ops, queue_owner as q

class GuardTests(unittest.TestCase):
 def test_foreign_target_rejected_before_gpu_query(self):
  plan={'old_runs':[{'run':'old','namespace':'expected','gpus':[7]}]}
  with patch.object(ops,'checked',return_value=plan),patch.object(ops,'gpu_pids',side_effect=AssertionError('GPU query reached')):
   with self.assertRaisesRegex(AssertionError,'not a frozen old target'):
    ops.stop_old([{'run':'foreign','namespace':'expected','gpus':[7],'identity':{}}],'combo-')
 def test_pid_reuse_is_not_live(self):
  with patch.object(ops,'proc',return_value={'pid':123,'uid':1003,'start':21,'state':'S'}):
   self.assertFalse(ops.same({'pid':123,'uid':1003,'start':20}))
 def test_running_eval_cannot_be_bypassed_by_free_gpu(self):
  plan={'stage1_gate':{'owner':{'pid':10},'terminal_files':['/missing']},'runs':{'stage1-full':{'gpus':[2]}}}
  with patch.object(ops,'same',return_value=True),patch.object(ops,'gpu_pids',side_effect=AssertionError('GPU query reached')):
   self.assertEqual(q.stage1_gate(plan),(False,'WAITING_OFFICIAL_EVAL'))
 def test_boot_change_stops_gate_before_receipt(self):
  with self.assertRaisesRegex(AssertionError,'Host rebooted'):
   q.gate({'gates':{'clean':{}},'boot_id':'not-the-current-boot'},'clean')

if __name__=='__main__':unittest.main()
