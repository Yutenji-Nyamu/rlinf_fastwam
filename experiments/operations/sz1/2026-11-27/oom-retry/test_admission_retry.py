import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch,Mock
import os
import subprocess
import json
import queue_admission as admission
from patch_dojo_admission import ORIGINAL, update


class PairAdmissionTest(unittest.TestCase):
    def test_normal_launcher(self):
        with TemporaryDirectory() as root, patch.object(admission, "same", side_effect=AssertionError):
            admission.wait_for_oom_retry(Path(root))

    def test_each_successor_waits_for_same_gate(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            gate = root / "gate-release.json"
            primary = root / "identity.json"
            policy = {"oom_retry_gate":{"identity":{"pid":33}, "release":str(gate), "primary_identity":str(primary)}}
            for card in ("4", "7"):
                gate.unlink(missing_ok=True)
                d=root/card;d.mkdir();(d/"wait.json").touch()
                with patch.object(admission,"read",return_value=policy), patch.object(admission,"same",return_value=True), patch.object(admission.time,"sleep",side_effect=lambda _:gate.touch()) as pause:
                    admission.wait_for_oom_retry(d)
                    self.assertEqual(pause.call_count,1)

    def test_dead_gate_with_live_relaunched_job(self):
        with TemporaryDirectory() as root:
            d=Path(root);(d/"wait.json").touch()
            p=d/"primary.json";p.touch();g=d/"released.json"
            guards={"oom_retry_gate":{"identity":{"pid":1},"release":str(g),"primary_identity":str(p)}}
            def get(path):return guards if path.name=="wait.json" else {"pid":2}
            live=iter((False,True,False,False))
            with patch.object(admission,"read",side_effect=get),patch.object(admission,"same",side_effect=lambda obj:next(live)),patch.object(admission.time,"sleep") as delay:
                admission.wait_for_oom_retry(d);self.assertEqual(delay.call_count,1)

    def test_recovery_shares_with_external_and_other_own_jobs(self):
        with TemporaryDirectory() as root:
            job = Path(root)
            with (patch.dict(os.environ, {"PAIR_OOM_RETRY_RULES": str(job)}),
                  patch.object(admission, "gpus", side_effect=AssertionError),
                  patch.object(admission, "retry_gpu_processes", return_value={4: [11, 22], 7: [33]}),
                  patch.object(admission, "retry_owner", return_value=False),
                  patch.object(admission, "save") as save):
                admission.wait_for_cards(job, {4: {}, 7: {}})
                self.assertEqual(save.call_args.args[1]["phase"], "READY")
                self.assertEqual(save.call_args.args[1]["other_gpu_processes"], {4: 2, 7: 1})

    def test_waits_only_for_previous_worker_and_reports_a_stuck_worker(self):
        with TemporaryDirectory() as root:
            job = Path(root)
            with (patch.object(admission, "retry_gpu_processes", side_effect=[{4:[11,22]}, {4:[22]}]),
                  patch.object(admission, "retry_owner", side_effect=lambda pid, *_, **opts:pid==11),
                  patch.object(admission.time, "sleep") as pause,
                  patch.object(admission, "save") as save):
                admission.wait_retry_cleanup(job, [4])
                self.assertEqual(pause.call_count, 1)
                self.assertEqual(save.call_args_list[0].args[1]["own_gpu_pids"], {4:[11]})
            with (patch.object(admission, "retry_gpu_processes", return_value={4:[11]}),
                  patch.object(admission, "retry_owner", return_value=True),
                  patch.object(admission.time, "monotonic", side_effect=[0,100]),
                  patch.object(admission, "save") as save,
                  self.assertRaises(TimeoutError)):
                try:
                    admission.wait_retry_cleanup(job, [4])
                finally:
                    self.assertEqual(save.call_args.args[1]["phase"], "CLEANUP_TIMEOUT")

    def test_timeout_does_not_start_a_second_probe(self):
        with TemporaryDirectory() as root:
            failure = subprocess.TimeoutExpired(["nvidia-smi"], 8)
            child = Mock()
            child.communicate.side_effect = failure
            with (patch.object(subprocess, "Popen", return_value=child) as start,
                  patch.object(admission, "save") as save,
                  self.assertRaisesRegex(RuntimeError, "GPU driver query")):
                try:
                    admission.wait_retry_cleanup(Path(root), [5,6])
                finally:
                    start.assert_called_once()
                    child.kill.assert_called_once()
                    child.wait.assert_not_called()
                    self.assertEqual(save.call_args.args[1]["phase"], "GPU_QUERY_FAILED")

    def test_process_identity_requires_job_marker_or_live_receipt(self):
        with TemporaryDirectory() as root:
            proc = Path(root) / "11"
            proc.mkdir()
            job = Path("/training/run-abc")
            env = proc / "environ"
            (proc / "cmdline").write_bytes(b"ray::worker\0")
            env.write_bytes(b"DOJO_GRPO_OWNER=/training/run-abcd\0")
            self.assertFalse(admission.retry_owner(11, job, {}, Path(root)))
            env.write_bytes(b"DOJO_GRPO_OWNER=/training/run-abc\0")
            self.assertTrue(admission.retry_owner(11, job, {}, Path(root)))
            with patch.object(os, "getuid", return_value=proc.stat().st_uid+1):
                self.assertFalse(admission.retry_owner(11, job, {}, Path(root)))
            env.write_bytes(b"")
            with patch.object(admission, "same", return_value=False):
                self.assertFalse(admission.retry_owner(11, job, {11:{"pid":11,"start":123}}, Path(root)))

    def test_dojo_patch_keeps_normal_admission_and_is_idempotent(self):
        original = "def launch():\n    if True:\n" + ORIGINAL
        result = update(original)
        self.assertEqual(update(result), result)
        self.assertIn("wait_retry_cleanup(d, cards)", result)
        self.assertIn("deadline = time.monotonic() + 240", result)

    def test_live_ray_binding_survives_dashboard_loss_only_for_exact_archived_driver(self):
        with TemporaryDirectory() as root:
            job=Path(root)
            (job/"fastwam_driver.py").touch()
            argv=str(job/"fastwam_driver.py")
            address="172.17.0.1:6389"
            info={"pid":55,"uid":os.getuid(),"start":10,"state":"R"}
            (job/"training-environment.json").write_text(json.dumps({"RAY_ADDRESS":address}))
            (job/"plan.json").write_text(json.dumps({"ray_dashboard_url":"http://172.17.0.1:8266"}))
            driver=job/"driver-identity.json"
            driver.write_text(json.dumps(info))
            get=lambda pid,entry,jid,node: dict(driver_info={"pid":str(pid),"node_ip_address":node},
                                                 entrypoint="python -B "+entry,job_id=jid)
            jobs=[get(55,argv,"01000000","172.17.0.1"), get(55,argv+"2","02000000","172.17.0.1"),
                  get(66,argv,"03000000","172.17.0.1"), get(55,argv,"04000000","172.17.0.2")]
            with patch.object(admission,"same",return_value=True), patch.object(admission,"ray_jobs",return_value=jobs) as api:
                result=admission.bind_ray_driver(job)
                self.assertEqual(result["job_id"],"01000000")
                api.assert_called_once_with("http://127.0.0.1:8266")
            arch=job/"prior-oom-retry-1";arch.mkdir()
            driver.rename(arch/driver.name)
            (arch/"released.json").write_text(json.dumps({"time":admission.time.time()}))
            with patch.object(admission,"same",return_value=False), patch.object(admission,"ray_jobs",side_effect=AssertionError):
                cached=admission.bind_ray_driver(job)
                self.assertEqual(cached["job_id"],"01000000")
                self.assertTrue(cached["until"])
            (job/"oom-ray-driver.json").unlink()
            completed = dict(jobs[0], start_time=(cached["until"]-5)*1000)
            with patch.object(admission,"same",return_value=False), patch.object(admission,"ray_jobs",return_value=[completed]):
                self.assertEqual(admission.bind_ray_driver(job)["job_id"], "01000000")
            other=dict(info,start=11)
            (arch/driver.name).write_text(json.dumps(other))
            with patch.object(admission,"same",return_value=False), patch.object(admission,"ray_jobs",return_value=jobs):
                self.assertIsNone(admission.bind_ray_driver(job))

    def test_worker_requires_matching_ray_job_cluster_and_lifetime(self):
        with TemporaryDirectory() as root:
            proc=Path(root)/"75";proc.mkdir()
            (proc/"cmdline").write_bytes(b"ray::FSDPActor\0")
            (proc/"stat").write_text("75 (ray::worker) "+" ".join(["S"]+["0"]*18+["150"]))
            env=proc/"environ"
            env.write_bytes(b"RAY_JOB_ID=01000000\0RAY_ADDRESS=node:6389\0")
            record=dict(job_id="01000000",address="node:6389",boot_sec=0,
                        driver=dict(start=100),until=None)
            match=lambda b:admission.retry_owner(75,Path("/new-job"),{},Path(root),ray_job=b)
            self.assertTrue(match(record))
            self.assertFalse(match(dict(record,address="node:6388")))
            self.assertFalse(match(dict(record,job_id="02000000")))
            self.assertFalse(match(dict(record,driver=dict(start=350))))

