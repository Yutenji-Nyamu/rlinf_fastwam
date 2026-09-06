# Sidney pi0.5 GRPO DVAC action advantage [0.5,1.5]

Control baseline source 81be3193d91fe9950a3fc1bdedd14063a85e72d8; closed at step168, fixed165=24/32, saved checkpoint160. Control light archive published as 1d015a2aa03ec8132d8207ba47a2be3dbe1d9591 on codex/sz-sidney-pi05-current-rlinf.

Only production addition: config/dvac_grpo/adv_half.yaml. Algorithm, model, FSDP, environment source unchanged. Native primary config +dvac_grpo=adv_half. Same original SFT, 200 rounds, GPU4/5, 256 attempts/round, micro32/global1024/U2, M10/noise0.5, LR5e-6, fixed32/eval5/save10. Action-level ratio/clip and advantage weighting are method changes, not equivalent to chunk-level clipping with weights=1. 14 CPU tests passed; actual startup is recorded separately.
