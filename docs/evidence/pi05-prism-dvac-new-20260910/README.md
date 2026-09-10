# Verified two-round smoke

Tested production commit: `443573542ff3aed6958603a121b240184d86771e`.

Physical GPUs 0/3, 256 attempts/round, G8/U2, global/microbatch 1024/32, LR5e-6, H50/M10/noise0.5. Dedicated Ray namespace; no evaluation or model checkpoints.

119 focused CPU tests passed. Both training rounds exited successfully. Four real actor artifacts independently reproduced trajectory quality, complete RLOO advantages, local/outer factors and combined weights. Owned smoke resources were released; all protected own training GPU processes retained their identities. One unrelated user process was observed exited; its cause was not audited. The cleanup receipt contains only owned smoke targets, with no unrelated Ray names removed.

| Metric | Round 1 | Round 2 |
|---|---:|---:|
| env/success_once | 0.35546875 | 0.37109375 |
| train/actor/policy_loss | 0.021833476 | 0.027817924 |
| train/actor/dvac_weight_mean | 1 | 1 |
| train/actor/grad_norm | 11.275128 | 13.662132 |
| train/actor/dvac_chunk_factor_std | 0.24684407 | 0.24295667 |
| train/actor/dvac_positive_adv_mass_ratio | 0.9296872 | 0.93353945 |
| train/actor/dvac_negative_adv_mass_ratio | 1.0431831 | 1.0605037 |
| train/actor/dvac_weight_min | 0.1939642 | 0.19265506 |
| train/actor/dvac_weight_max | 2.7263374 | 2.9112375 |
| train/actor/dvac_weight_ess_fraction | 0.89430159 | 0.8945114 |
| train/actor/dvac_local_center_error | 1.7881393e-07 | 1.7881393e-07 |
| train/prism_dvac/quality_adv_abs_mean | 0.065306127 | 0.065306127 |
| train/prism_dvac/rescued_same_outcome_group_fraction | 0.125 | 0.15625 |
| train/prism_dvac/all_failure_group_fraction | 0.09375 | 0.15625 |
| train/prism_dvac/all_success_group_fraction | 0.03125 | 0 |
| train/prism_dvac_new/adv_composition_max_error | 2.2910535e-07 | 1.937151e-07 |
| train/prism_dvac_new/binary_adv_mass_ratio | 0.98308367 | 0.99559176 |
| train/prism_dvac_new/quality_adv_mass_ratio | 0.99814689 | 1.0012158 |

Success rates here describe smoke collection, not fixed evaluation. Rescued-group fraction is relative to all groups. These two rounds establish the training path, not performance improvement. Mean weight one does not imply unchanged gradient magnitude. Joint method-state contracts were tested on CPU; this no-checkpoint smoke does not validate full GPU model restoration.
