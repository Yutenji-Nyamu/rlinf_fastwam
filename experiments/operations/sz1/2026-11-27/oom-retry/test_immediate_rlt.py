"""CPU regression tests for simulator readiness and immediate clean-RLT retry."""
import json, os, socket, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import rlt_formal_oom_1127 as helper

CAMERA = 'File "/example/envs/camera/camera.py", line 280, in update_picture\n    self.left_camera.take_picture()\nRuntimeError: cannot create buffer\n'
REFUSED = 'File "/example/rlinf/envs/robodojo/robodojo_env.py", line 60, in _rpc\nconnection = Client(self.socket, family="AF_UNIX")\nConnectionRefusedError: [Errno 111] Connection refused\n'

class ImmediateRLT(unittest.TestCase):
    def test_listener_and_stale_file(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"sim.sock"
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as owner:
                owner.bind(str(path))
                self.assertFalse(helper.socket_listening(path))
                owner.listen(1)
                self.assertTrue(helper.socket_listening(path))
                with self.assertRaises(RuntimeError):helper.clear_inactive_socket(path)
            self.assertFalse(helper.socket_listening(path))
            helper.clear_inactive_socket(path)
            self.assertFalse(path.exists())

    def test_other_connection_is_unrelated(self):
        self.assertIsNone(helper.dojo_refused(['ConnectionRefusedError: [Errno 111] Connection refused']))

    def test_repeated_camera_and_transport(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);(d/'config').mkdir()
            (d/'plan.json').write_text(json.dumps({'retry_sim_after_cleanup':True}))
            OmegaConf.save(OmegaConf.create({'runner':{'resume_dir':None}}),d/'config/formal.yaml')
            log=d/'formal.log'; native=d/'native.log';native.touch()
            budget={}
            with patch.object(helper,'select_retry',return_value=(True,None)):
                for i,line in enumerate([CAMERA,REFUSED,'RuntimeError: CUDA out of memory.',CAMERA,REFUSED]*2):
                    log.write_text(line)
                    self.assertTrue(helper.prepare_oom_retry(d,log,(native,0),budget=budget,gpu=6))
                    self.assertEqual(budget['delay'],0)
                log.write_text('ValueError: bad tensor shape')
                self.assertIsNone(helper.prepare_oom_retry(d,log,(native,0),budget=budget,gpu=6))

if __name__=='__main__':unittest.main()

