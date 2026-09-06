# Pinned Fast-WAM SAPIEN scene-fence fix

This is an opt-in native symbol replacement for SAPIEN 3.0.1's bundled
svulkan2 `d8516a4f1467167122ae85f53a8532dbceb1eec2`. It fixes only the timeline
RTRenderer::render overload: reset and submit the existing scene-access fence,
matching the binary overload. `scene_fence.patch` is the upstream-source diff.

`scene_fence.cpp` reproduces that overload and the unchanged abstract Denoiser
interface. It uses the installed wheel's C++ headers and links to the unchanged
original library. No denoiser, shader, simulator, training algorithm or dependency
version is replaced. No global files are installed or overwritten.

The native loading mechanism is environment-local `ctypes.CDLL(RTLD_LOCAL)`
before SAPIEN and after PyTorch, through the opt-in RoboTwin loader. Set only
`RLINF_SCENE_FENCE_LIBRARY`; never globally LD_PRELOAD renderer dependencies:
the bundled library exports ZIP symbols that can interfere with PyTorch's
checkpoint reader. Actor, Rollout and driver processes do not load the patch.
This is not an in-memory or on-disk binary edit. The original wheel's exact
library SHA256 is checked at build time; changing wheel/ABI requires re-audit.
Do not generalize to other SAPIEN builds. Use CUDA-interop and C++11-ABI layout.

Build on the server:

```bash
bash build.sh /absolute/venv /absolute/new-output-directory
source /absolute/new-output-directory/enable.sh
```

Before training, test real multicamera rendering in a separate process and verify
dynamic binding to the replacement symbol. Inspect new Ray EnvWorker maps/env;
also verify PyTorch reads the original checkpoint before and after rendering.
Preserve the original wheel and other jobs. Unsetting this job's
`RLINF_SCENE_FENCE_LIBRARY` for a future process disables the fix.

This repairs a concrete synchronization omission. A short regression does not
prove that it was the sole cause of the previously stalled training frame, nor
does it establish long-run stability. No GIL/timeout/cache changes are included.
