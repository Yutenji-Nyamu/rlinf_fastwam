from unittest import TestCase, mock
from oom_classify_1127 import driver_failure, find_gpu_oom
from oom_contention import retry_delay
from pair_gate import retry_transport

TRACE = ['ray::EnvWorker.interact() (pid=12345, repr=EnvGroup(rank=0))',
         '  File "/repo/envs/camera/camera.py", line 276, in update_picture',
         '    self.left_camera.take_picture()', 'RuntimeError: cannot create buffer']
class CameraRetryTests(TestCase):
    def test_primary_camera_with_followup_communications_exit(self):
        m = driver_failure(TRACE + ['Connection reset by peer'])[1]
        self.assertEqual((m['kind'],m['env_rank'],m['worker_pid']),('robotwin_camera_buffer',0,12345))
        self.assertFalse(retry_transport(TRACE + ['SimulatorTransportError: peer exited']))
    def test_other_errors_are_not_silently_restarted(self):
        for lines in ([TRACE[-1]], [TRACE[1].replace('envs/camera/camera','model/backend'), *TRACE[2:]],
                      [TRACE[1], 'self.other_function()', TRACE[-1]], TRACE + ['vk::DeviceLostError'],
                      TRACE + ['VK_ERROR_OUT_OF_HOST_MEMORY'], TRACE + ['ValueError: bad shape'],
                      TRACE + ['│ Global Step: 10/300 │ Progress:', 'Connection reset by peer']):
            with self.subTest(lines=lines): self.assertIsNone(driver_failure(lines)[1])
    def test_explicit_error_has_precedence(self):
        self.assertEqual(find_gpu_oom(TRACE+['RuntimeError: CUDA out of memory.'])['kind'],'cuda')
    def test_known_native_restart_scoped_to_the_last_round(self):
        from pair_gate import recoverable_native_error
        sig = 'Fatal Python error: PyGILState_Release: thread state must be current'
        self.assertEqual(recoverable_native_error([sig]),sig)
        self.assertIsNone(recoverable_native_error([sig, 'VK_ERROR_DEVICE_LOST']))
        self.assertIsNone(recoverable_native_error([sig, 'ValueError: bad settings', 'EOFError']))
        self.assertIsNone(recoverable_native_error(['EOFError']))
        ev = dict(kind='robotwin_native_restart')
        self.assertEqual(retry_delay(ev,[4,7],0,0)[0],30)
        self.assertIsNone(retry_delay(ev,[4,7],1,0)[0])
    def test_pair_handles_repeat_after_recovered_native_training(self):
        import json, tempfile
        from pathlib import Path
        from pair_gate import oom_evidence
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            log = p/'formal-retry.log'
            failed = dict(phase='FAILED_RETURN_RLT',error='driver exit 255')
            error = 'RuntimeError: RoboTwin simulator RPC exceeded 900 seconds'
            for reached in (False, True):
                log.write_text(('│ Global Step: 121/300 │ Progress:\n' if reached else '')+error+'\n')
                ev = oom_evidence(p,{},failed,0,'fastwam')
                self.assertEqual((ev['kind'],ev['internal_oom_retry']),('robotwin_native_restart',not reached))
            log.write_text('SimulatorTransportError: generic\nConnection reset by peer')
            self.assertIsNone(oom_evidence(p,{},failed,0,'fastwam'))
    @mock.patch('oom_contention.os.getuid',return_value=1001)
    def test_rank_mapping_and_account_receipt(self,_):
        m = dict(find_gpu_oom(TRACE), physical_gpu=4)
        rows={4:[(100,1001,41500),(110,65534,33000)],7:[(99,65534,35000)]}
        delay,a=retry_delay(m,[4,7],3,3,snapshot=rows)
        self.assertEqual((delay,a['external'],len(a['external_processes'])),(120,True,1))
        self.assertEqual(a['external_processes'][0]['pid'],110)
        self.assertTrue(a['external_processes'][0]['account'])
        rows[4]=[(100,1001,41500)]
        self.assertIsNone(retry_delay(m,[4,7],1,0,snapshot=rows)[0])
        self.assertEqual(retry_delay(m,[4,7],0,0,snapshot=rows)[0],30)

