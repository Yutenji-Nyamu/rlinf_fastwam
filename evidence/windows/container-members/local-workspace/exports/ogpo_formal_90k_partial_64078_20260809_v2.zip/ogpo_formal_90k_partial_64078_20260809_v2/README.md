# OGPO pi0 RoboTwin formal v2: interrupted 64k evidence package

This is a lightweight, read-only evidence package. The fresh 90k run stopped after 64,078 committed primitive rows and 2,703 paired updates. Ray killed a worker when container memory reached 228.21/240 GiB; the later NCCL timeout was a consequence.

The package contains logs, TensorBoard scalars, resolved configuration, provenance, one-second resources, two small checkpoint completion manifests, derived tables, and plots. It contains no checkpoint tensors or replay payloads. The 30,959-row and 60,968-row full checkpoints remain on the server and are complete.
