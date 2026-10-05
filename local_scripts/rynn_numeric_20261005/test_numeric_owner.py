"""Run on SZ3; reuse exact GPU/parent boundary tests for the new child contract."""
import unittest
import rynn_numeric_owner as O
import test_rynn_diagnostic_owner as original
import test_rynn_diagnostic_owner_v2 as parent

def fixture():
    argv=[O.RYNN_PYTHON,'-u','-B',str(O.D/'code/rynn_numeric_probe.py'),
        '--service-module',str(O.S/'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path',str(O.MODEL),'--manifest-path',str(O.MODEL/'manifest.json'),
        '--physical-gpu','4','--cases-json',str(O.CASES),'--samples-npz',str(O.SAMPLES),
        '--official-inference',str(O.OFFICIAL),'--output',str(O.O/'result.json')]
    return dict(mode='rynn-single-gpu-diagnostic',physical_gpus=[4],untouched_gpus=[5,6,7],
        owner_dir=str(O.O),parent_owner=str(O.PARENT),timeout_seconds=900,argv=argv,
        environment={'CUDA_VISIBLE_DEVICES':'4','CUDA_DEVICE_ORDER':'PCI_BUS_ID'},
        untouched_drivers={key:{} for key in ('gpu5','gpu6','gpu7')})

original.O,original.fixture=O,fixture
parent.O=O
TestDiagnosticOwner=original.TestDiagnosticOwner
TestSingleParent=parent.TestSingleParent
if __name__=='__main__':unittest.main()
