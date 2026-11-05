# Third-party references and adaptations

`dojo_runtime.py` adapts the opt-in control-state, unpaced-render and physics-thread
patches in [MINT-SJTU/RoboScope-Kit](https://github.com/MINT-SJTU/RoboScope-Kit),
revision `30158a77ae222701f5c31c327ebdc5085281b6fa`, under Apache License 2.0.
RoboScope credits these experimental Dojo candidates to @QLL-CCDD's RoboDojo work.
The adapter retains the source attribution, adds process-local selection and runtime
guards, and leaves the installed RoboDojo source unchanged.

The control loop is adapted from the already deployed RoboDojo `RobotManager`,
SHA256 `1306cb6d17b58a04ad8fc298edb6eb9122469048dbc9717dcf246f4f985d9974`.
The adapter refuses a different native source. Selected joint arrays remain copies;
the per-call action cache is limited to the exact `MetaControl` class.

The SAC critic-gradient optimization follows the idea documented in
[OpenAI Spinning Up SAC](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py):
actor optimization needs the critic's action derivative, but not critic parameter
gradients. No Spinning Up implementation was copied. The rejected experiment used detached
parameter views inside the existing FSDP forward path. It failed the complete FSDP
lifecycle check and was removed from the runtime; its two model patches are archived
under `rejected/` and are never installed.

The surrounding RLinf source remains under its existing Apache License 2.0.
The full Apache license is included in the repository's root `LICENSE` file.

See `sources.json` for the source-to-switch mapping, changed behavior and required checks.
