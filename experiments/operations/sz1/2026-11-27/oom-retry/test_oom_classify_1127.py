import unittest

from oom_classify_1127 import find_gpu_oom


class TestGpuOom(unittest.TestCase):
    def test_fastwam_oom_before_ray_shutdown(self):
        error = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB. "
                 "Process 338660 has 18.33 GiB memory in use.")
        lines = ["FSDP wait_for_unshard\n", error + "\n"]
        lines += ["ray::ActorDiedError: worker shut down\n"] * 75
        result = find_gpu_oom(iter(lines))
        self.assertEqual(result["kind"], "cuda")
        self.assertEqual(result["line_number"], 2)
        self.assertEqual(result["line"], error)
        self.assertEqual(result["tail"], [lines[-1].rstrip()] * 8)

    def test_rlt_renderer(self):
        lines = ["RuntimeError: cannot create buffer",
                 "(EnvGroup pid=536256) [svulkan2] [error] OIDN Error: out of memory"]
        result = find_gpu_oom(lines)
        self.assertEqual(result["kind"], "oidn_renderer")
        self.assertEqual(result["line_number"], 2)

    def test_explicit_device_errors(self):
        cases = {
            "RuntimeError: Vulkan call failed with VK_ERROR_OUT_OF_DEVICE_MEMORY": "vulkan",
            "cudaErrorMemoryAllocation: out of memory": "cuda",
            "cuMemAlloc returned CUDA_ERROR_OUT_OF_MEMORY": "cuda",
            "\x1b[31mRuntimeError: CUDA error: out of memory\x1b[0m": "cuda",
        }
        for message, kind in cases.items():
            with self.subTest(message=message):
                self.assertEqual(find_gpu_oom([message])["kind"], kind)

    def test_unrelated_failures(self):
        for message in [
            "kernel: Out of memory: Killed process 112 (python)",
            "torch.OutOfMemoryError: DefaultCPUAllocator: can't allocate memory",
            "ray.exceptions.OutOfMemoryError: Node was low on available memory",
            "RuntimeError: CUDA error: an illegal memory access was encountered",
            "NVRM: Xid (PCI:0000:c7:00): 31, MMU Fault",
            "RuntimeError: cannot create buffer",
            "VK_ERROR_OUT_OF_HOST_MEMORY",
            "[svulkan2] [error] OIDN Error: invalid handle",
            "OIDN Error: out of memory",  # Backend unspecified.
            "[svulkan2] [error] OIDN Error: out of memory on CPU",
            "Enable checks for CUDA_ERROR_OUT_OF_MEMORY",
        ]:
            with self.subTest(message=message):
                self.assertIsNone(find_gpu_oom([message]))
        self.assertIsNone(find_gpu_oom([]))


if __name__ == "__main__":
    unittest.main()
