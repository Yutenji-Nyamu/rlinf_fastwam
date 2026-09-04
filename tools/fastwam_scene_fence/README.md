# Pinned Fast-WAM SAPIEN scene-fence fix

This is an opt-in native symbol replacement for SAPIEN 3.0.1's bundled
svulkan2 `d8516a4f1467167122ae85f53a8532dbceb1eec2`. It fixes only the timeline
RTRenderer::render overload: reset and submit the existing scene-access fence,
matching the binary overload. `scene_fence.patch` is the upstream-source diff.

`scene_fence.cpp` reproduces that overload and the unchanged abstract Denoiser
interface. It uses the installed wheel's C++ headers and links to the unchanged
original library. No denoiser, shader, simulator, training algorithm or dependency
version is replaced. No global files are installed or overwritten.

The native loading mechanism is ELF symbol interposition via job-local
`LD_PRELOAD`, not an in-memory or on-disk binary edit. Only enable it before the
new process starts. The original wheel's exact library SHA256 is checked at build
time; changing wheel/ABI requires re-audit. Do not generalize this patch to other
SAPIEN builds. Compilation must use the CUDA-interop and C++11-ABI layout.

Build on the server:

```bash
bash build.sh /absolute/venv /absolute/new-output-directory
source /absolute/new-output-directory/enable.sh
```

Before training, test real multicamera rendering in a separate process and verify
dynamic binding to the replacement symbol. Inspect new Ray EnvWorker maps/env;
checking that a file exists is not sufficient. Preserve the original wheel and
other jobs. Unsetting this job's `LD_PRELOAD` for a future process disables the fix.

This repairs a concrete synchronization omission. A short regression does not
prove that it was the sole cause of the previously stalled training frame, nor
does it establish long-run stability. No GIL/timeout/cache changes are included.
