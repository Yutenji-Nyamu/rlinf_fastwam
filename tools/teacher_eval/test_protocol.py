import unittest
from tools.teacher_eval.protocol import episode_schedule,batches,paired_comparison

class PairedProtocolTest(unittest.TestCase):
    def test_full_schedule_at_48_parallel(self):
        schedule=episode_schedule(list(range(32)),4)
        self.assertEqual([len(b)for b in batches(schedule,48)],[48,48,32])
        self.assertEqual(len({x['episode_id']for x in schedule}),128)
        self.assertTrue(all(sum(x['scene_seed']==s for x in schedule)==4 for s in range(32)))
    def test_changed_scene_is_not_paired(self):
        row=dict(episode_id=0,scene_seed=5,repeat=0,accepted_seed=5,instruction='switch',chunk_steps=20,success=False)
        with self.assertRaisesRegex(ValueError,'accepted_seed'):
            paired_comparison([row],[dict(row,accepted_seed=6,success=True)])
    def test_actual_paired_gain(self):
        rows=[dict(episode_id=i,scene_seed=i,repeat=0,accepted_seed=i,instruction='switch',chunk_steps=10,success=i==0)for i in range(4)]
        result=paired_comparison(rows,[dict(x,success=x['episode_id']in(0,2,3))for x in rows])
        self.assertEqual(result['teacher_only_success'],2);self.assertEqual(result['difference'],.5)
    def test_chunk_mismatch_is_not_paired(self):
        row=dict(episode_id=0,scene_seed=5,repeat=0,accepted_seed=5,instruction='switch',chunk_steps=20,success=False)
        with self.assertRaisesRegex(ValueError,'chunk_steps'):
            paired_comparison([row],[dict(row,chunk_steps=10)])

if __name__=='__main__':unittest.main()
