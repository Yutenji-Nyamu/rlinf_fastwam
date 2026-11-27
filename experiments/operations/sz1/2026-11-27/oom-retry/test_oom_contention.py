import os
from unittest import TestCase, mock
from oom_contention import retry_delay
from oom_classify_1127 import find_gpu_oom

class OomDecisionTests(TestCase):
    def setUp(self):
        user = mock.patch("oom_contention.os.getuid", return_value=2000)
        owner = mock.patch("oom_contention.owner_uid", return_value=None)
        user.start(); owner.start()
        self.addCleanup(user.stop); self.addCleanup(owner.stop)
        self.ev = {"kind":"cuda","line":"RuntimeError: CUDA out of memory. Tried to allocate 80.00 MiB."}
    def decide(self,e,s,prior=2,streak=2,cards=(4,7)):
        return retry_delay(e,cards,prior,streak,snapshot=s)
    def test_each_renewed_external_oom_keeps_retrying(self):
        e=dict(self.ev,line=self.ev['line']+" Process 123 has 18.33 GiB memory in use.")
        s={4:[(123,7777,12800)],7:[]}
        for n,sec in enumerate((30,60,120,120,120)):
            t,r=self.decide(e,s,n,n)
            self.assertEqual((t,r['external']),(sec,True))
    def test_local_device_zero_and_nonfailing_card(self):
        e=dict(self.ev,line="(rank=1 pid=478) "+self.ev['line']+" GPU 0 has ...")
        s={4:[(333,7777,13000)],7:[(478,2000,32000)]}
        self.assertIsNone(self.decide(e,s)[0])
    def test_uncertain_peer_earns_one_more_try_not_infinite(self):
        e=dict(self.ev,line=self.ev['line']+" Process 123 has 18.33 GiB memory in use.")
        for n,t in enumerate((30,60,None)):
            delay,r=self.decide(e,{4:[],7:[]},n,n)
            self.assertEqual(delay,t);self.assertFalse(r["external"])
    def test_native_uses_actual_card(self):
        e={"kind":"oidn_renderer","log":"native-g5.log","line":"[svulkan2] [error] OIDN Error: out of memory"}
        self.assertEqual(self.decide(e,{5:[(11,7777,12000)],6:[]},cards=(5,6))[0],120)
        self.assertIsNone(self.decide(e,{5:[],6:[(11,7777,12000)]},cards=(5,6))[0])
    def test_own_small_graphics_and_insufficient_external(self):
        for s in ({4:[(1,7777,7),(2,2000,18000)]},{4:[]}):
            self.assertIsNone(self.decide(self.ev,s,1,0,cards=(4,))[0])
            self.assertEqual(self.decide(self.ev,s,0,0,cards=(4,))[0],30)
        e=dict(self.ev,line="RuntimeError: CUDA out of memory. Tried to allocate 30.0 GiB. of which 0.5 GiB is free.")
        self.assertIsNone(self.decide(e,{4:[(1,7777,10000)]},1,0,cards=(4,))[0])
    def test_named_cuda_vulkan_and_cpu(self):
        for line in ("torch.cuda.OutOfMemoryError: allocation failed","RuntimeError: vk::Device::allocateMemory: ErrorOutOfDeviceMemory","vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY","CUDA error 2: out of memory"):
            self.assertTrue(find_gpu_oom([line]),line)
        for line in ("VK_ERROR_OUT_OF_HOST_MEMORY","NVRM Xid 109","failed to create buffer","torch.OutOfMemoryError: DefaultCPUAllocator: cannot allocate"):
            self.assertIsNone(find_gpu_oom([line]),line)

