# Complete-round K timing and Dojo execution video

## Timing

Online success FM-BC on RoboTwin `turn_switch`, Sidney pi0.5, H100 80GB, 32 parallel environments, 200 action steps, H50/C50, 10 denoising steps. Each K=1/2/4/8 starts from the same model weights and fresh optimizer. One warmup round and three measured rounds per K. Four AdamW updates per round, global batch 128, micro batch 8, LR 2.5e-5. The updated actor is copied to the rollout model after each round. Training outputs are temporary and never replace formal experiment weights.

K1 uses the ordinary main action and unit weights. K>1 preserves that main action and adds K-1 independent samples, restoring the main RNG state afterwards. The entropy uses normalized D14, sample variance (ddof=1), epsilon 1e-6. Entropy is standardized over the current query batch and mapped by the existing bounded-linear 0/1/5 rule; the original masked FM loss applies these per-action weights. This timing configuration is recorded explicitly, including the calibration scope. Successful online trajectories supply the fixed update budget; a zero-success round is reported as incomplete rather than silently skipping updates.

Record reset/environment, main inference, candidate inference, entropy/weights, parameter update, complete wall time and module sum. GPU memory is sampled from the whole physical card as well as PyTorch peak allocations. Collection and updates are synchronous. The table reports mean of three rounds, K4-K1 wall-time difference, relative increase and candidate-inference share of K4. Model loading and evaluation/checkpointing are outside these training rounds.

Sources reused: RLinf `OpenPi0ForRLActionPrediction`, `prepare_dagger_sft_batch`, `sft_forward`, `rlinf.data.online_bc.masked_fm_loss`; the bounded-linear mapping from `rlinf.algorithms.online_bc_dvac`; the existing ProcessVectorEnv lifecycle and appendix K entropy definition. Raw values and configuration accompany the figure.

## Dojo video

Reuse original Put Bottles Into Dustbin layout IDs, native seed 1, policy seed 20261111 and original H50/C50 settings. Preserve the original 12-environment batch. Record slots 0, 11 and 9 as D-B00/D-B11/D-B09. Enable the existing per-action readback and encode frames without adding physics steps or extra rendering calls. Recompute K4 on every observation, keeping the main RNG unchanged. The initial-state comparison is saved; the new curve is paired with this execution's frames.

Each selected episode includes MP4, frame/action/query CSV, observations, four action/noise arrays, continuous action-level entropy, A/B/C images around its main peak, and plotting code. Video plays at 25 fps with one frame per action; the action index, rather than playback seconds, is the plot axis.

## Queue and output

Run each bounded collection within the existing card reservation after the paired teacher evaluation and before formal continuation. Completion or failure returns to the prepared trainer. All raw data are written to the server Home volume. The result renderer and publisher run on CPU after GPU cleanup. Code and finished figure bundles use the existing `codex/teacher-eval-20261117` branch.

Entrypoints: `round_cost.py --job <local-job.json>`; `dojo_collect.py --deployment <local-deployment> --data <output> --original <recorded-episode-root>`. Machine paths and credentials belong to local deployment files, not this package.
