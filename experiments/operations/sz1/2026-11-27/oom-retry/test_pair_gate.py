"""Run with the server's Python: python3 -B -m unittest test_pair_gate."""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pair_gate import complete_checkpoint, log_positions, oom_evidence, retry_transport

FAILED = {"result": {"phase": "FAILED_RETURN_RLT", "error": "Driver exited 1"}}
STEP = "│ Global Step:   89/300 │ Progress: ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  29.7% │\n"
OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB.\n"
RPC = "SimulatorTransportError: RoboTwin simulator RPC exceeded 900 seconds\n"


class PairGateTests(unittest.TestCase):
    def test_completed_round_and_nested_ray_failure(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            driver = d / "formal-driver.log"
            driver.write_text(OOM + RPC + STEP + "KeyError: task\n")
            self.assertIsNone(oom_evidence(d, {}, FAILED, 0))
            driver.write_text(OOM + "KeyError: task\n")
            self.assertIsNone(oom_evidence(d, {}, FAILED, 0))
            driver.write_text(STEP + OOM + "ray.exceptions.ActorDiedError: actor exited\n" + RPC)
            self.assertEqual(oom_evidence(d, {}, FAILED, 0)["kind"], "cuda")
            self.assertFalse(retry_transport(driver.read_text().splitlines()))
            self.assertTrue(retry_transport([OOM, STEP, RPC]))
            self.assertFalse(retry_transport([RPC, STEP, "actor exited\n"]))

    def test_already_consumed_transport_retry_for_oom(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/"formal-driver.log").write_text(OOM + RPC)
            (d/"formal-retry.log").write_text(STEP + OOM)
            (d/"simulator-retry.json").write_text(json.dumps({"failed_log":"formal-driver.log"}))
            self.assertIsNone(oom_evidence(d, {}, FAILED, 0, "fastwam"))
            (d/"formal-driver.log").write_text(OOM + STEP + RPC)
            self.assertEqual(oom_evidence(d, {}, FAILED, 0, "fastwam")["log"], "formal-retry.log")

    def test_native_oom_only_if_stamped_after_completed_checkpoint(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            formal, native = d/"formal-driver.log", d/"native-g6.log"
            formal.write_text(STEP + RPC)
            marker = d/"global_step_89/dojo-complete.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({"step":89, "time":1000}))
            for stamp, expected in [("1970-01-01T00:16:39Z ", False), ("",False),
                                    ("1970-01-01T00:16:41Z ", True)]:
                native.write_text(stamp + OOM)
                with self.subTest(stamp=stamp):
                    self.assertEqual(bool(oom_evidence(d, {}, FAILED, 0, "dojo", True, d)), expected)

    def test_only_new_output_from_final_attempt_counts(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            first = d / "formal-driver.log"
            first.write_text("torch.OutOfMemoryError: CUDA out of memory before arming\n")
            positions = log_positions(d)
            self.assertIsNone(oom_evidence(d, positions, FAILED, 0))
            with first.open("a") as stream:
                stream.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB\n")
            self.assertEqual(oom_evidence(d, positions, FAILED, 0)["log"], "formal-driver.log")
            second = d / "formal-retry.log"
            second.write_text("SimulatorTransportError: RPC lost")
            self.assertIsNone(oom_evidence(d, positions, FAILED, 0))
            second.write_text("RuntimeError: CUDA error: out of memory")
            self.assertEqual(oom_evidence(d, positions, FAILED, 0)["log"], second.name)
            self.assertIsNone(oom_evidence(d, positions, {"result": {"phase": "COMPLETE"}}, 0))
            self.assertIsNone(oom_evidence(d, positions, {"result": {"phase": "FAILED_RETURN_RLT",
                                                                         "error": "SystemExit: 143"}}, 0))

    def test_non_gpu_and_native_loading_failures_are_not_training_oom(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "native-g5.log").write_text("CUDA_ERROR_OUT_OF_MEMORY")
            (d / "formal-driver.log").write_text("MemoryError: host memory allocation failed")
            self.assertIsNone(oom_evidence(d, {}, FAILED, 0))

    def test_full_checkpoint_layouts_and_partial_latest(self):
        for kind, required in [
            ("dojo", ["actor/local_shard_checkpoint/checkpoint_rank_0.pt",
                      "actor/local_shard_checkpoint/checkpoint_rank_1.pt"]),
            ("fastwam", ["actor/dcp_checkpoint/.metadata", "actor/dcp_checkpoint/__0_0.distcp",
                         "actor/dcp_checkpoint/__1_0.distcp"])]:
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                d = Path(tmp)
                cp = d / "global_step_111"
                cp.mkdir()
                for name in required:
                    path = cp / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
                (cp / "dojo-complete.json").write_text(json.dumps({"step": 111}))
                incomplete = d / "global_step_112"
                incomplete.mkdir()
                self.assertEqual(complete_checkpoint(d, kind), (111, cp))
                (incomplete / "dojo-complete.json").write_text(json.dumps({"step": 112}))
                self.assertEqual(complete_checkpoint(d, kind), (111, cp))



    def test_oidn_message_can_precede_megabytes_of_ray_shutdown(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "formal-driver.log").write_text(
                "(EnvGroup pid=536256) [svulkan2] [error] OIDN Error: out of memory\n"
                + ("Ray worker closing\n" * 140000))
            self.assertEqual(oom_evidence(d, {}, FAILED, 0)["kind"], "oidn_renderer")

    def test_dojo_native_oom_requires_current_native_evidence_and_formal_rpc_failure(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            formal, native = d/"formal-driver.log", d/"native-g6.log"
            formal.write_text("")
            oom = "RuntimeError: CUDA error: out of memory\n"
            native.write_text(oom)
            offsets = log_positions(d)
            formal.write_text("ConnectionResetError: [Errno 104] Connection reset by peer\n")
            self.assertIsNone(oom_evidence(d, offsets, FAILED, 0, "dojo", True))
            with native.open("a") as f:
                f.write(oom)
            hit = oom_evidence(d, offsets, FAILED, 0, "dojo", True)
            self.assertEqual((hit["log"], hit["driver_log"]), ("native-g6.log", "formal-driver.log"))
            self.assertIsNone(oom_evidence(d, offsets, FAILED, 0, "dojo", False))
            self.assertIsNone(oom_evidence(d, offsets, FAILED, 0, "fastwam", True))
            formal.write_text("KeyError: stage configuration\n")
            self.assertIsNone(oom_evidence(d, offsets, FAILED, 0, "dojo", True))

    def test_formal_retry_real_cpu_child(self):
        import types
        from unittest import mock
        from pair_gate import run

        for result in ("success", "second_oom", "launch_failure"):
            with self.subTest(result=result), TemporaryDirectory() as td:
                base = Path(td)
                pair, gate = base / "pair", base / "gate"
                pair.mkdir()
                gate.mkdir()
                origin = {"pid": 50, "start": 1, "uid": 1003, "state": "R"}
                controller = {"pid": 70, "start": 2, "uid": 1003, "state": "R"}
                active_file = base / "active.json"
                read = lambda p: json.loads(p.read_text())
                live = [True]
                def save(path, value):
                    path.write_text(json.dumps(value))
                    if path == gate / "identity.json":
                        (gate / "registered.json").write_text(json.dumps({"identity": value}))
                    if path == gate / "status.json" and value["phase"] == "MONITORING_ORIGINAL":
                        with (pair / "formal-driver.log").open("a") as log:
                            log.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB.\n")
                        (pair / "released.json").write_text(json.dumps(FAILED))
                        live[0] = False
                save(active_file, {"slots": {"4":{"owner":str(pair),"identity":origin},
                                              "7":{"owner":str(pair),"identity":origin},
                                              "5":{"owner":"another","identity":"untouched"}}})
                save(pair / "plan.json", dict(gpus=[4,7], python=sys.executable, training_source=str(base)))
                save(pair / "identity.json", origin)
                save(pair / "registered.json", {"identity":origin})
                (pair / "formal-driver.log").write_text("")
                config = pair / "config" / "formal.yaml"
                config.parent.mkdir()
                config.write_text("initial")
                root = base / "checkpoints"
                for step in (18,20,21):
                    cp = root / f"global_step_{step}"
                    cp.mkdir(parents=True)
                    save(cp / "dojo-complete.json", {"step":step})
                    for rank_file in (".metadata","__0_0.distcp","__1_0.distcp"):
                        if step == 21 and rank_file == "__1_0.distcp":
                            continue
                        item=cp / "actor/dcp_checkpoint" / rank_file
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_bytes(b"model")
                (pair / "expected-result").write_text(result)
                (pair / "fastwam_pair.py").write_text(FAKE_OWNER)
                oc=types.SimpleNamespace(
                    load=lambda p:types.SimpleNamespace(runner=types.SimpleNamespace(
                        resume_dir=str(root/"global_step_18"),max_steps=30)),
                    save=lambda cfg,path:path.write_text(str(cfg.runner.resume_dir)))
                def same(who):
                    return live[0] if who["pid"] == 50 else Path("/proc",str(who["pid"])).exists()
                deps={"common":types.SimpleNamespace(proc=lambda pid:controller,same=same,read=read,save=save),
                      "omegaconf":types.SimpleNamespace(OmegaConf=oc)}
                with mock.patch.dict(sys.modules,deps), mock.patch("pair_gate.ACTIVE_INDEX",active_file), \
                     mock.patch("pair_gate.os.getuid",return_value=1003), mock.patch("pair_gate.signal.signal"):
                    if result == "launch_failure":
                        with mock.patch("pair_gate.subprocess.Popen",side_effect=OSError("launch failed")):
                            with self.assertRaisesRegex(OSError,"launch failed"):
                                run(gate,pair,"fastwam")
                    else:
                        run(gate,pair,"fastwam")
                if result == "launch_failure":
                    self.assertEqual(read(pair/"released.json"),FAILED)
                    self.assertEqual(read(gate/"released.json")["phase"],"GATE_ERROR")
                else:
                    terminal="COMPLETE_RETURN_RLT" if result=="success" else "FAILED_RETURN_RLT"
                    self.assertEqual(read(pair/"released.json")["result"]["phase"],terminal)
                    receipt=read(gate/"released.json")
                    self.assertEqual((receipt["phase"],receipt["resume"]),("RETRY_FINISHED",20))
                    self.assertEqual(read(active_file)["slots"]["7"]["identity"],read(pair/"identity.json"))
                    self.assertEqual(read(active_file)["slots"]["5"]["identity"],"untouched")
                    self.assertEqual((pair/"retry-count").read_text(),"1")
                self.assertTrue((gate/"attempt-1/reason.json").is_file())

    def test_lost_owner_and_driver_reconstruct_release(self):
        from unittest import mock
        from pair_gate import original_result
        with TemporaryDirectory() as td:
            pair=Path(td)
            (pair/"status.json").write_text(json.dumps({"phase":"FORMAL_TRAINING"}))
            read=lambda p:json.loads(p.read_text())
            save=lambda p,v:p.write_text(json.dumps(v))
            with mock.patch("pair_gate.live_driver",side_effect=[True,False]),mock.patch("pair_gate.time.sleep"):
                release=original_result(pair,"fastwam",{"pid":12},lambda *a,**k:None,read,lambda _:False,save)
            self.assertTrue(release["reconstructed_after_exit"])
            self.assertEqual(read(pair/"released.json")["result"]["phase"],"FAILED_RETURN_RLT")


FAKE_OWNER = """import json,os,time,sys
from pathlib import Path
d=Path(sys.argv[1]);w=dict(pid=os.getpid(),uid=os.getuid(),start=100)
assert (Path(os.environ["PAIR_OOM_RETRY_RULES"])/"pair_gate.py").is_file()
assert Path((d/'config/formal.yaml').read_text()).name=='global_step_20'
with (d/'retry-count').open('x') as out: out.write('1')
(d/'identity.json').write_text(json.dumps(w))
for i in range(100):
 if (d/'registered.json').is_file():break
 time.sleep(.03)
assert json.loads((d/'registered.json').read_text())['identity']==w
succeeded=(d/'expected-result').read_text()=='success'
(d/'formal-driver.log').write_text('finished' if succeeded else 'torch.OutOfMemoryError: CUDA out of memory')
(d/'released.json').write_text(json.dumps({'result':{'phase':'COMPLETE_RETURN_RLT' if succeeded else 'FAILED_RETURN_RLT'}}))
sys.exit(0 if succeeded else 1)
"""

