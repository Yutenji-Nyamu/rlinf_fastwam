"""CPU checks for exact device-metadata migration and formal method invariants."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import torch
from common import change_devices, jhash

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'examples/embodiment'))
from train_expo_formal import validate_inputs
from train_expo_ft import digest

def saved_state():
    base = {'parallel':{'devices':4,'candidate_microbatch':8},'optimizer':{'lr':2.5e-5}}
    return dict(base=dict(contract=base,contract_sha256=jhash(base),trainable_params={'w':torch.arange(8.)},
                          optimizer={'step':torch.tensor(12.)},base_updates=12),
        core={'w':torch.arange(4.),'_extra_state':{'config':{'parallel_devices':4,'n_base':8,'n_edit':8},
             'optimizer':{'step':12},'update_calls':12}},
        rng={'candidate':torch.tensor([9]),'cuda':[torch.tensor([i]) for i in range(4)],'python':(1,2)},
        replay={'episodes':5},cadence={'physical_actions':100},progress={'evals':[10]})

class DeviceMigration(unittest.TestCase):
    def test_only_authorized_metadata_changes(self):
        for count in (1,2,4):
            original=saved_state(); result=change_devices(copy.deepcopy(original),count)
            self.assertEqual(result['base']['contract']['parallel']['devices'],count)
            self.assertEqual(result['base']['contract_sha256'],jhash(result['base']['contract']))
            self.assertEqual(result['core']['_extra_state']['config']['parallel_devices'],count)
            self.assertEqual(len(result['rng']['cuda']),count)
            result['base']['contract']['parallel']['devices']=4
            result['base']['contract_sha256']=original['base']['contract_sha256']
            result['core']['_extra_state']['config']['parallel_devices']=4
            result['rng']['cuda']=original['rng']['cuda']
            self.assertEqual(digest(original),digest(result))
    def test_reject_unknown_counts(self):
        for count in (0,3,8,True,None):
            with self.assertRaises(ValueError): change_devices(saved_state(),count)
    def test_reject_non_four_card_origin(self):
        saved=saved_state(); saved['core']['_extra_state']['config']['parallel_devices']=2
        with self.assertRaises(AssertionError): change_devices(saved,1)
    def test_reject_incomplete_rng(self):
        saved=saved_state(); saved['rng']['cuda'].pop()
        with self.assertRaises(AssertionError): change_devices(saved,1)
    def test_formal_only_device_count_varies(self):
        path=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002/inputs.json')
        original=json.loads(path.read_text())
        original['port_source_manifest']={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()
                                         for p in original['port_source_manifest']}
        for count in (1,2,4):
            data=copy.deepcopy(original)
            for k in ('formal','core'): data[k]['parallel_devices']=count
            config,_=validate_inputs(data,60000); self.assertEqual(config['batch_size'],64)
        for key,value in (('batch_size',16),('num_envs',4),('candidate_microbatch',4),('fm_microbatch',16)):
            data=copy.deepcopy(original); data['formal'][key]=value
            with self.assertRaises(ValueError): validate_inputs(data,60000)
    def test_formal_reject_mismatched_counts(self):
        data=json.loads(Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002/inputs.json').read_text())
        data['formal']['parallel_devices']=1
        with self.assertRaises(ValueError): validate_inputs(data,60000)

if __name__=='__main__': unittest.main()
