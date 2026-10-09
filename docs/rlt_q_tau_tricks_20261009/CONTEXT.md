# RLT-Q U / Norm: tau and two controls

Base: 348005d5f (algorithm b1d2d8d55). U, Norm, and configurable temperature already exist. This change adds two independently enabled controls to the scalar actor Q weight, reusing the RLT DVAC control functions. BC, critic TD/targets, replay sampling, and teacher signals keep their existing behavior. The method defaults remain the original Clean4 Q recipe.

- Default tau remains 2.5. Both controls default off; disabled controls preserve the original resume contract.
- Chunk dropout p=0.2 restores selected Q weights to 1, including failed episodes because Q weighting covers all sampled rows. No chunks are discarded; no renormalization or inverse-probability scaling.
- Alpha decreases linearly from 1 at R1 to 0 at R500 in the RLT example. Only chunk alpha exists for scalar Q. The schedule uses runner rounds; dropout uses the persisted update counter and a private CPU RNG.
- Compute the full global batch once before microbatch splitting. Checkpoints reject changed enabled controls or tau.

Execution authorization: SZ3 GPU6/7, minimal smoke only, then restore the original per-card RLT queue. No formal Q training. The live check at 18:42 Oct9 found the BC jobs already concluded and GPU6/7 removed from the old queue for other windows; no C/G processes remained on those cards. The 4/5 WM queue is preserved.

CPU checks cover actual actor loss/gradient, full-vs-microbatch equality, controls/RNG, resume, tau, and unit-weight endpoints. One two-round smoke per signal uses the installed SZ3 lift_pot Control/Stage1 and reuses the previous Q smoke budget: N4/U1, B32/MB16, initial pool4/critic2, Q coefficient0.05. Smoke alone uses R1->R2 annealing to exercise both endpoints. Paths, command, and measured outcomes are recorded in EXECUTION.md.
