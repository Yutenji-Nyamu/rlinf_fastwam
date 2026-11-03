RoboDojo native GRPO adapter
===========================

This experimental adapter connects the existing RoboDojo native multi-environment
runtime to RLinf Clean GRPO. Acceptance and limitations are recorded in
``docs/dojo_rl/08_IMPLEMENTATION_20261103.md``. Interface availability does not
establish an RL success-rate improvement.

Environment and policy
----------------------

The first task is ``put_bottles_into_dustbin`` with the official seed-1 Dojo
pi0.5 checkpoint. Inputs are three RGB cameras, 14 robot state coordinates and a
task instruction. Preserve quantile normalization, arm-joint deltas and absolute
grippers. Output transforms recover physical joint targets. Prediction and
execution lengths are 50, with 10 flow steps and a 700-action cap.

Native Isaac processes advance environments using ``take_action_batch`` and
communicate with RLinf through private authenticated Unix sockets. Training and
evaluation share persistent native processes. Each eight-trajectory GRPO group
loads the same saved layout. Training and evaluation layout sets are separate.
Success earns one reward at its actual executed action position.

Integration and acceptance
--------------------------

Reuse the working native RoboDojo installation and exact assets. This adapter
does not replace Isaac or its driver. ``tools/dojo_grpo/convert_jax.py`` and
``convert_pytorch_layout.py`` reuse RLinf's converters for the original weights.
Compare native JAX and converted PyTorch actions with the same observation/noise.

Configure both environment types as ``robodojo``, with ``physical_gpus``,
``rpc_dir``, ``group_size`` and layout pools. Select ``pi05_robodojo`` as the
OpenPI configuration. Deployment supplies machine-specific paths and GPU placement.
CPU contract checks run on the server::

   python tools/dojo_grpo/test_protocol.py

GPU acceptance covers model parity, batched inference, initial-state equivalence,
native rollouts, one Clean update and complete checkpoint restoration. Reuse the
existing RLT handoff and isolated Ray-namespace cleanup.

Evaluation
----------

The plan evaluates 32 episodes at the start, round 2, then every 5 rounds.
RL evaluation collects neither videos nor auxiliary uncertainty signals.
Full training state is saved; curves and small diagnostics remain independently
of checkpoint retention. See the implementation record for actual acceptance.
