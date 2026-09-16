# RLT DVAC temperature pair: natural completion at R600

Both fresh Stage2 runs reached the planned 600 rounds and exited with code 0 before replacement. Both use 4 env, B512/micro256, UTD5, success scale 1, and the same current-AR Stage1 checkpoint. Only the local/chunk temperature and run identity differ.

| Run | Temperature | Fixed R600 | Final checkpoint | Curves and metadata |
| --- | --- | --- | --- | --- |
| GPU6 | 0.5 | 17/20 | CP600, complete | [Archive](rlt_t05/README.md) |
| GPU7 | 1 | 19/20 | CP600, complete | [Archive](rlt_t1/README.md) |

Each archive has separate raw, MA10, MA20, MA50 and fixed-evaluation figures, complete scalar history, original light logs/events, resolved configuration, source identities, and checkpoint metadata. Collection transitions from teacher to student; fixed evaluation always tests student. The final fixed counts alone do not establish a training-method ranking.

No model, optimizer or replay payload is included, loaded or deleted. These ZIPs preserve evidence rather than independently restorable checkpoints; final CP600 and its dependencies remain on the server. Completion markers and seven required component metadata entries passed inspection; no restore test was performed.
