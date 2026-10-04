"""Targeted CPU-only tests. Run on the Linux server with CUDA_VISIBLE_DEVICES=''."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import graphics_scope_prepare as prep
import graphics_scope_runtime as runtime


def fixture_manifest():
    return {"physical_gpus": [4, 5, 6, 7], "cards": {
        str(gpu): {"physical_gpu": gpu, "gpu_uuid": "GPU-fixture-" + str(gpu),
                   "minor": gpu, "mask": 1 << gpu, "commname": "odwf" + str(gpu) + "-12345678"}
        for gpu in (4, 5, 6, 7)}}


class TargetSelectionTests(unittest.TestCase):
    def test_each_numeric_and_uuid_target_is_exact(self):
        manifest = fixture_manifest()
        for gpu in (4, 5, 6, 7):
            self.assertEqual(runtime.select_target(manifest, str(gpu)), gpu)
            self.assertEqual(runtime.select_target(manifest, "GPU-fixture-" + str(gpu)), gpu)

    def test_discovery_and_exact_cpu_full_mask_only_default_to_four(self):
        for value in (None, "", "0,1,2,3,4,5,6,7"):
            self.assertEqual(runtime.select_target(fixture_manifest(), value), 4)

    def test_wrong_or_multidevice_masks_fail(self):
        for value in ("0", "3", "8", "4,5", "0,4", "4,5,6,7", "GPU-other", " 6"):
            with self.assertRaises(RuntimeError):
                runtime.select_target(fixture_manifest(), value)

    def test_cpu_mask_rewrite_does_not_fake_a_nonfour_target(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": runtime.FULL_NODE_MASK}):
            runtime.check_cuda(fixture_manifest()["cards"]["4"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": runtime.FULL_NODE_MASK}):
            with self.assertRaises(RuntimeError):
                runtime.check_cuda(fixture_manifest()["cards"]["6"])

    def test_discovery_never_allows_rendering(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(runtime.check_cuda(fixture_manifest()["cards"]["4"]), "")
            with self.assertRaises(RuntimeError):
                runtime.check_cuda(fixture_manifest()["cards"]["4"], require=True)

    def test_correct_single_card_is_not_remapped(self):
        for gpu in (4, 5, 6, 7):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
                self.assertEqual(runtime.check_cuda(fixture_manifest()["cards"][str(gpu)], require=True), str(gpu))
                other = "5" if gpu == 4 else "4"
                with self.assertRaises(RuntimeError):
                    runtime.check_cuda(fixture_manifest()["cards"][other], require=True)


class ProfileCompatibilityTests(unittest.TestCase):
    def content(self, pattern=None, mask=16):
        if pattern is None:
            pattern = {"feature": "commname", "matches": "odw4-6c25fa1986"}
        return json.dumps({"rules": [{"pattern": pattern, "profile": ["EGLVisibleDGPUDevices", mask]}]})

    def test_known_legacy_private_rule_is_accepted(self):
        self.assertEqual(prep.validate_existing_device_profile(self.content()), ["odw4-6c25fa1986"])

    def test_account_fallback_is_rejected(self):
        with self.assertRaises(RuntimeError):
            prep.validate_existing_device_profile(self.content(pattern=[], mask=240))

    def test_multiple_device_old_rule_is_rejected(self):
        with self.assertRaises(RuntimeError):
            prep.validate_existing_device_profile(self.content(mask=240))

    def test_unrelated_scope_is_not_silently_adopted(self):
        with self.assertRaises(RuntimeError):
            prep.validate_existing_device_profile(self.content(pattern={"feature": "commname", "matches": "unrelated"}))


class UnreadableCpuProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proc = Path(self.tmp.name) / "99999"
        self.proc.mkdir()
        fields = ["S", "1"] + ["0"] * 17 + ["42"] + ["0"] * 3
        (self.proc / "stat").write_text("99999 ((sd-pam)) " + " ".join(fields))
        (self.proc / "comm").write_text("(sd-pam)\n")
        (self.proc / "cmdline").write_bytes(b"(sd-pam) \0")
        (self.proc / "environ").write_bytes(b"")
        self.approved = dict(pid=99999, uid=os.getuid(), start=42, ppid=1,
                             comm="(sd-pam)", cmdline_sha256=prep.digest(self.proc / "cmdline"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_explicit_exact_audit_can_survive_permission_error(self):
        original = Path.read_bytes
        def deny_environment(path):
            if path == self.proc / "environ":
                raise PermissionError("fixture non-dumpable CPU daemon")
            return original(path)
        with patch.object(Path, "iterdir", side_effect=lambda: iter([self.proc])), \
                patch.object(Path, "read_bytes", deny_environment), \
                patch.object(prep, "gpu_process_pids", return_value=set()):
            with self.assertRaisesRegex(RuntimeError, "Cannot establish scope retirement"):
                prep.scoped_processes()
            self.assertEqual(prep.scoped_processes(unreadable_cpu_exemptions=[self.approved]), [])

    def test_identity_changes_scope_comm_and_gpu_context_are_rejected(self):
        for key, value in (("start", 43), ("ppid", 2), ("cmdline_sha256", "changed")):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "identity changed"):
                prep.verify_unreadable_cpu_process(self.proc, dict(self.approved, **{key: value}), set())
        with self.assertRaisesRegex(RuntimeError, "GPU context"):
            prep.verify_unreadable_cpu_process(self.proc, self.approved, {99999})
        (self.proc / "comm").write_text("odwf6-12345678\n")
        with self.assertRaisesRegex(RuntimeError, "graphics scope"):
            prep.verify_unreadable_cpu_process(self.proc, dict(self.approved, comm="odwf6-12345678"), set())

    def test_readable_scope_is_never_exempted(self):
        (self.proc / "environ").write_bytes((runtime.MANIFEST_ENV + "=/frozen/scope.json\0").encode())
        with patch.object(Path, "iterdir", side_effect=lambda: iter([self.proc])), \
                patch.object(prep, "gpu_process_pids", return_value=set()):
            rows = prep.scoped_processes(unreadable_cpu_exemptions=[self.approved])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pid"], 99999)
        self.assertEqual(rows[0]["scopes"], {runtime.MANIFEST_ENV: "/frozen/scope.json"})


class ActivationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scope.json").write_text("{}")
        (self.root / "environment-fragment.json").write_text(json.dumps({runtime.MANIFEST_ENV: str(self.root / "scope.json")}))
        draft = self.root / "staged-profile.json"
        draft.write_text("{\"rules\":[]}\n")
        self.destination = self.root / "active-profile.json"
        self.proof = self.root / "proof.json"
        self.proof.write_text("{}")
        self.receipt = self.root / "scope-activation.json"
        self.manifest = dict(token="unit-test-scope", profile_path=str(self.destination),
                             profile_sha256=prep.digest(draft), staged_profile_path=str(draft),
                             existing_profiles={}, audited_device_profiles=[], driver_version="test",
                             runtime_path="/fixture/graphics_scope_runtime.py", runtime_sha256="fixture")

    def tearDown(self):
        self.tmp.cleanup()

    def test_profile_is_not_written_if_retirement_is_incomplete(self):
        with patch.object(prep, "staged_manifest", return_value=(self.root, self.manifest)), \
                patch.object(prep, "verify_retirement", side_effect=RuntimeError("old process live")):
            with self.assertRaisesRegex(RuntimeError, "old process live"):
                prep.activate(self.root, self.proof, self.receipt)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())

    def test_active_receipt_precedes_post_install_validation(self):
        with patch.object(prep, "staged_manifest", return_value=(self.root, self.manifest)), \
                patch.object(prep, "verify_retirement", return_value={"no_scoped_processes": True}), \
                patch.object(prep, "inventory", return_value=("test", [])), \
                patch.object(prep, "profile_snapshot", return_value=({}, [], [])), \
                patch.object(prep, "current_boot", return_value="test-boot"), \
                patch.object(prep, "read_manifest", side_effect=RuntimeError("probe preparation failed")):
            with self.assertRaisesRegex(RuntimeError, "probe preparation failed"):
                prep.activate(self.root, self.proof, self.receipt)
        self.assertTrue(self.destination.exists())
        active = json.loads(self.receipt.read_text())
        self.assertEqual(active["status"], "active")
        self.assertFalse(active["native_probe_verified"])
        self.assertEqual(active["profile_sha256"], prep.digest(self.destination))
        self.assertEqual(active["environment_fragment"][runtime.MANIFEST_ENV], str(self.root / "scope.json"))


if __name__ == "__main__":
    unittest.main()
