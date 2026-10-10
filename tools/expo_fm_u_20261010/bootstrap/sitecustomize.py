import ctypes,os
n=os.environ.get("EXPO_GPU_PROCESS_NAME")
if n and (n not in ("dojo-scope-g4","dojo-scope-g6") or ctypes.CDLL(None).prctl(15,ctypes.c_char_p(n.encode()),0,0,0)):os._exit(78)
