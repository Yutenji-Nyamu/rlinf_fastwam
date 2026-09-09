# Online BC with frozen target-TCP-motion weights

Independent branch from clean BC `01d770db3988da7862454e97434d4ff08f726fa2`.
Original labels remain raw 50x14 joint targets passed through the unchanged
Sidney model transforms. Accepted success queries get one CPU FK annotation.

For each arm, compute consecutive target TCP translation distance d and
SO(3) rotation angle theta. The first command uses its own pre-query state.
Combine arms as `u = sqrt(dL^2 + (0.1*thetaL)^2) + sqrt(dR^2 + (0.1*thetaR)^2)`.
One frozen scale c gives `w = clip(1/max(u/c,1e-3)^2,0.5,2)`, broadcast over the
original valid action dimensions in FM loss, with the original denominator.
No DVAC signal/statistics, mean-weight normalization, extra model forward,
training-time SAPIEN simulation, or change to the optimizer/sampling loop.

Geometry uses the actual aloha-agilex URDF with named arm joints and validated
link-to-joint child frames plus TCP offset. Raw geometry features, version/hash,
and full method/calibration identity support checked checkpoint restoration.

The fixed c=0.005978549941035213m is the median positive summed-arm magnitude
from historical clean 4/U5 R100 training success queries after length<=3:
179 episodes, 537 queries, 26850 positions. This historical preprocessing
information is an explicit method input. No reference record enters the new
replay, and no new collection or optimizer steps are used to calibrate.
It replaces the earlier combined-arm-norm candidate scale.

Planned GPU6 formal setting: fresh original model/empty replay, 4/U5,
batch1024/micro32, LR2.5e-5 constant, original Adam, 100 rounds, fixed32 every5,
save every10. In addition to weighting, length<=3 differs from historical
unfiltered clean BC; comparisons must disclose this admission difference.

AttenA+ reference: DaojiePENG/openpi at fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8,
models/pi0.py inverse_squared/epsilon/clip rule. FK, dual-arm sum and frozen
motion scale are explicit RoboTwin adaptations, not an exact LIBERO replication.
