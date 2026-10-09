# SZ2 storage recovery, 2026-10-09

All times China. Source e04e9d4fbc041519019f7e642765f157ffb78553 is pushed; 196 CPU checks passed. U smoke passed a complete Clean256 round; Norm retry also passed a complete Clean256 round. No formal GRPO run is authorized.

## Cause and observed effect

Ordinary SSH authenticated but command-channel creation timed out. /data and chenyiteng HOME resolve through /home/nvme. The installed mergerfs 2.33.3 daemon (PID3521640, start321205801, UID0, FUSE0:55) held its node allocator lock throughout prolonged garbage collection. Source-validated read-only metadata identified the maintenance thread as lock owner and workers waiting behind it. At19:45, 1784 to1781 one-MiB slabs were reclaimed in5 seconds; this was progressing GC, not an established hardware failure. Raw NVMe reads remained fast. CPU, memory and process limits were not exhausted.

U PID696929/start434856586 began19:15:24 and was delayed during initialization. It reached its first rollout epoch after20:05; the initial13m39s epoch includes the storage stall. Do not use that duration as method overhead.

Upstream fixed blocking node-slab GC in2.33.4; the selected minimal subsequent patch is2.33.5. Sources: [fix](https://github.com/trapexit/mergerfs/pull/1020), [release](https://github.com/trapexit/mergerfs/releases/tag/2.33.4), [online upgrade documentation](https://github.com/trapexit/mergerfs/blob/2.33.5/README.md#upgrade).

## Authorized online repair

The user explicitly approved root diagnosis and maintenance. Automatic approval initially rejected live switching because the earlier approval required new checkpoints and stopped tasks. The user then explicitly approved an online switch without pausing jobs or forcing new checkpoints, retaining old handles and locks. The approved switch succeeded at20:17:12.

The Ubuntu22.04/amd64 package SHA256 is8cef37b8afa535359265e1ae355edd08f85d7d4aeaca5b12bc3fe1871eacccf8. The installed2.33.5 binary SHA256 isc69a200c2b62b02db85cb26f77098e6e74cdcb31b25254e969771e29c4b90d56. A rollback2.33.3-1 package was rebuilt from the original installed files before installation. /etc/fstab and original branch/cache/create options are unchanged.

Private tests verified ordinary reads/writes, retained open handles and lock exclusion. They also established two relevant boundaries: locks are mount-local, and the old alias must be private to prevent shared mount propagation. The final procedure prepared the complete new mount tree with retained lock submounts, then published it using one recursive bind. All extant old-mount locks were checked against /proc immediately before the switch.

The active /home/nvme root is now2.33.5, device0:57, daemon PID2374949. Existing jobs, owner PID1586766, shared Ray and other users' sessions were not stopped. Ordinary chenyiteng SSH and UID20001 commands were verified afterward. Plan/HOME reads measured0.81ms/0.48ms. At20:58 both FUSE waiting queues were0 and load had fallen to about10.5, from the stalled snapshot near895. Old references remain deliberately alive; this is not a claim that the old daemon has exited.

## Exact recovery state and later retirement

Root transaction directory: /tmp/grpo-un-mergerfs-repair-20261009. Keep it until the retained mounts are retired. Records: before.json, shared-prepared.json, switch-intent.json, switched.json, recovery-references.json, unix-socket-inventory.json, socket-retention.json, fstab.backup, rollback-2.33.3-1.deb. fixed-candidate is the new mount source; retired-root is a private alias to the old0:55 mount.

Three old inode domains are retained beneath the new root and its prepared source:

- /home/nvme/team-home/chenyiteng/.cache/nvidia/GLCache
- /home/nvme/team-data/chenyiteng/deployment-20261008/bc-signal-tau-v1/owner.lock
- /home/nvme/team-data/chenyiteng/tmp/r26/session_2026-09-26_15-41-32_661562_140751/sockets

Do not unmount these while their existing lock holders or Ray socket users are active, kill the old daemon, or recursively clean the transaction directory. Later retirement requires fresh mount IDs, open-handle/lock ownership and an exact release window; it is not part of GRPO method behavior. The old underlying /home/nvme mount, its private alias and existing process references are intentionally retained for compatibility. The installed executable and future normal fstab mounts use2.33.5.

## Ray socket correction

The initial compatibility inventory retained locks but missed AF_UNIX sockets. Norm attempt1 (PID907836, start435359064) exited at20:39:36 before creating any training worker: raylet_ipc_client could not connect through the new FUSE inode. It did not execute Norm or consume a training round. Existing Ray connections were still live. Its logs and failed-before-workers.json are preserved; no successful completion was fabricated.

The live Unix-socket inventory identified11 unique paths, all in the existing Ray session sockets directory. At20:47:06 that exact old subtree was bound privately into both the prepared source and live root. Raylet PID141644/start321253905 remained unchanged; Ray was not restarted. An ordinary-user CPU ray.init/shutdown probe passed at20:47:41 (job5f000000). Norm attempt2 then created job60000000 and its GPU6/7 workers successfully, with unchanged code and budget.

## Resume GRPO

Own control: /data/chenyiteng/deployment-20261009/grpo-un-smoke-g67-v1. U passed all256 trajectories and both actor-update epochs, with finite nonconstant signals, nonuniform weights and successful exact cleanup. Norm retry also passed at21:22, including real updates, finite signal/weight artifacts and exact namespace/C+G release. Original GPU6 Clean/GPU7 Combo RLT was restored from step325 with unchanged requests; both advanced to326 and started their next rollout, verified21:32. See EXECUTION.md for the exact return owner and receipt. Shared4/5 ownership stays with its existing owner; GPU4 had independently fallen back before the storage switch, while GPU5 Norm-tau3 remained active.
