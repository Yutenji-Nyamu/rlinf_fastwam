# Formal graphics scope handoff

2026-10-04. New files only; no profile has been installed by this work. These files derive from the GPU4 v5 runtime and are for the already authorized physical GPUs 4–7. Keep the current frozen owner, source, manifests and profile unchanged.

## Files and selected device

- `graphics_scope_runtime.py`: numeric single-card CVD 4/5/6/7 or its exact UUID selects that card; any other nonempty mask fails, except the previously observed exact CPU mask `0,1,2,3,4,5,6,7`, which narrows to 4. An empty/unset CVD remains available to CPU discovery but cannot render.
- `graphics_scope_bootstrap.py`: copied to a private `bootstrap/sitecustomize.py`, opt-in through `RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST`. It refuses an interpreter that also carries `RLINF_OPENDW_GPU_SCOPE_MANIFEST`.
- `graphics_scope_prepare.py`: separate private staging and account-profile activation.
- `test_graphics_scope.py`: 12 CPU tests, no Ray connection or CUDA/SAPIEN import. Run on the server with `CUDA_VISIBLE_DEVICES='' python -B -m unittest -v test_graphics_scope.py`. Local AST parsing passed; server execution is still required.

Ray must receive the environment fragment before exec of each new Python worker. The existing per-worker physical placement supplies its single-card CVD before this bootstrap. A process pins its chosen target: changing its CVD to a different card later fails, rather than silently changing a live graphics binding.

The default CPU ChannelWorker fallback is 4, including a future RLT5/6/7 job that gets the full-node list. Its native environment still renders on its own single-card CVD. The fallback remains within the authorized 4–7 range; strict per-job communication-card fallback would be a separate explicit contract change.

## Stage before borrowing; activate after borrowing

`prepare_stage(output, token, audited_device_profiles, cc='cc', inherited=None)` creates a fresh canonical output directory containing:

```text
scope.json
staged.json
graphics-profile.staged.json
environment-fragment.json
marker.c / libopendw_formal_scope_<token>.so
bootstrap/sitecustomize.py / graphics_scope_runtime.py
receipts/
```

It only reads `nvidia-smi` and existing graphics rules, compiles a marker, and writes in the new directory. **It does not install any account profile or enable the fragment.** `audited_device_profiles` must explicitly name existing EGL-device profile files; only existing private per-card commname rules are accepted. Account-wide fallbacks and unknown rule forms are refused. Names/masks are checked against pinned card UUID/PCI/minor records; the new profile contains four private comm rules and no default rule.

The default `environment-fragment.json` contains only the new bootstrap/marker, application-profile switches and owner HOME/USER/LOGNAME; no CVD. Passing `inherited` merges its PYTHONPATH/LD_PRELOAD tails during staging. Alternatively, `staged_environment_fragment(scope_dir, inherited=...)` computes a future fragment without activation. Freeze its file/hash in the formal plan. WM service processes stay outside this scope; enable it only for the training driver and its Ray workers.

The post-borrow hook calls:

```python
activation = graphics_scope_prepare.activate(
    scope_dir,
    retirement_proof_path,
    new_cycle / "scope-activation.json",
)
# Activation receipt exists before any native probe.
fragment = graphics_scope_prepare.environment_fragment(scope_dir / "scope.json", inherited={})
# Run the separately owned native reset/close probe on physical 6 and 7,
# record its own acceptance/failure, then return the unchanged fragment.
```

Required retirement-proof JSON shape:

```json
{
  "schema": 1,
  "uid": 20001,
  "hostname": "h100-gpu01",
  "boot_id": "actual current boot id",
  "previous_owner": {"pid": 123, "uid": 20001, "start": 456},
  "previous_owner_final": {"path": "/exact/previous/final.json", "sha256": "actual hash"},
  "rlt_stop_receipt": {"path": "/exact/new-cycle/rlt-stopped.json", "sha256": "actual hash"}
}
```

The placeholder identity must be replaced from exact receipts. Activation checks the prior owner is no longer that live PID/start/UID, its terminal record is conclusive, the new-cycle receipt reports all four cards released and original drivers/namespaces stopped, and no owned legacy/formal scope process remains in `/proc`. It then verifies the entire profile snapshot is unchanged, writes a unique activation intent, exclusively installs the staged profile, immediately writes activation, and validates the active runtime. A repeated call only recovers the same exact intent; it never overwrites a profile.

Activation receipt fields used by the return layer:

```text
scope_id (= token), status=active, uid, hostname, boot_id
manifest, manifest_sha256
runtime_path, runtime_sha256
environment_fragment (dictionary), environment_fragment_file, environment_fragment_sha256
profile, profile_sha256, native_probe_verified=false
retirement, activation_receipt
```

`native_probe_verified=false` deliberately remains in this immutable installation receipt. Write native acceptance separately. Profile installation is not a successful native-probe claim.

## Return and failure semantics

The generic return layer preserves each frozen original environment file. Only its exact new-cycle prepared/runtime environment reads receive the active fragment. Remove the **exact old** legacy manifest variable, old marker from LD_PRELOAD and old bootstrap entry from PYTHONPATH using the frozen old manifest; then prepend the new marker/bootstrap and preserve that RLT run's own path tails. Do not transplant the WM driver's unrelated PYTHONPATH into RLT.

If the native probe fails after activation, the active receipt remains and RLT must use the new scope on return. An explicit `rollback_exact(scope_dir, activation_receipt, rollback_receipt)` is available only after all users of the new scope have exited; it removes exactly the owned, hash-matching profile and writes a separate rollback receipt. The return layer must recognize rollback before using an old active receipt. The automatic hook should not invoke rollback unless that path is explicitly integrated and verified.

If activation is interrupted between profile creation and receipt writing, its unique intent plus exact profile hash identify the partial transaction; re-entering `activate` with the same proof/receipt can finalize it after verifying retirement again. A conflicting or incomplete profile hash remains an explicit error, never an assumption that old scope is safe.

No rollback, failed probe, or copied source is grounds to claim GPU0 is clear. Acceptance still requires actual C/G process evidence during native rendering and after release. Old profiles stay on disk; old scope processes must stay retired because their v5 manifest correctly rejects this newly added rule file.
