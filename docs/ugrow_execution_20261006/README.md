# U signal integration: exact source and live acceptance evidence

The two training sources remain frozen at BC `1c4b3810a1cddcd1ae38312126448c80348f5686`
and RLT `bfbc9c889bfe687dc24de7c3dc1d66138d3010df`. This branch adds the corrected
CPU owner and dated lightweight evidence. It is not the running model checkout.
See [the consolidated context](../ugrow_CONTEXT_20261006.md) for the method,
original recipes, physical GPU ownership and current phase.

BC's first two-round smoke exited zero, collected natural successes and applied
nonunit U weights in its second round. Its original owner gate looked for a
nonexistent `update_step` scalar. The corrected gate uses completed same-round
optimizer/replay metrics and reuses that already-complete smoke before a fresh
100-round formal run.

RLT's first two-round smoke exited zero and saved complete CP2 with update count
4. It collected no success: after just two initialization updates its second
round had already switched to the untrained student. It was not accepted as
successful-weighting coverage. One additional fixed four-round smoke preserves
the teacher for collection, with exactly three smoke-only recipe changes in
`tools/ugrow_ops_v2/rlt_smoke_v2_recipe.py`. The formal N4/U5, 800-round recipe is
unchanged and starts from the original Stage1 with fresh Stage2 replay.

Both v1 lanes released their exact Ray jobs and GPU contexts and returned the
original RLT. The v2 adapter binds those exact returned drivers, retains their
latest complete checkpoint, and checks their own cleanup before borrowing the
same cards. No GPU reset, shared Ray restart or operation on GPUs 6/7 is used.
The lane owner returns RLT once only after its whole U transaction terminates
and clears its namespace, process identities and compute/graphics contexts.

`current.json` carries a capture timestamp. A missing `formal_first_round` means
that acceptance evidence is still pending at that capture. RLT's first formal
receipt may prove only teacher precollection while the original 10k replay
threshold and 15k initialization-update schedule delay online learning.
These engineering checks do not establish a training improvement.

Only source, resolved recipes, scalar histories and small receipts are included.
Model weights, replay tensors, credentials and full logs remain on the server.
