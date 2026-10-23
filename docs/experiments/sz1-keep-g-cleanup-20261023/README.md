# Shenzhen1 checkpoint cleanup: retain the G takeover pair

The user explicitly selected the G experiment (Clean CP140 weak, U tau3 CP70 strong, C20/H50, takeover allowed only in the first three chunks). Both retained Shenzhen1 model files were checked after cleanup; the complete Clean CP140 recovery state remains. Shenzhen1 has only an inference-weight copy of U tau3 CP70; its original full recovery location on another currently unreachable server was not modified or revalidated.

Removed 1,865 explicitly authorized binary checkpoint/replay files of at least 1 MiB: other local GRPO checkpoints, all five reviewed EXPO bundles, reviewed old BC state on data/home, and large files in the exact 13 older RLT points. Inode-deduplicated allocation: 488.018 GiB. Directories, small records, configs, curves and logs remain. The 13 RLT points contributed only 0.528 GiB in large files; approximately 10 GiB of small replay/index files remains.

All four active fallback configurations (225/225/25/300) were excluded. Current distillation and other users' processes were not signalled or restarted. Exact local/server manifests retain path-level deletion evidence; this public receipt contains the lightweight accounting and retained paths. Retention registry updates revoke only the explicitly removed Shenzhen1 copies; other servers remain unchanged.
