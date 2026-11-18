# GeoWAM offline action entropy

36 dense recorded trajectories. The requested episodes are G08 (Plug Charger) and G02 (Sort Block). Each row links to its action-level CSV, overview, H.264 video and complete drawing materials.

**0 / 36 episodes published.**

| ID | Task | Recording | Valid / total | Global peak (step) | Materials |
|---|---|---|---:|---:|---|

## Method

At every synchronized head/wrist RGB, head depth and robot state, use the G01 encoder, checkpoint and native sampler to draw four independent 32-step predictions. Save all four normalized and physical action chunks. The curve is the diagonal Gaussian differential entropy of the four first-step predictions in the seven normalized action dimensions: 0.5 × sum(log(2πe × (sample variance + 1e−6))). The terminal observation also receives a prediction.

The CSV retains one row per original observation. Action steps start at 1. The original frame ID comes from sync.csv, the MP4 index starts at 0, and time is the controller timestamp relative to the first observation. Invalid depth leaves a blank entropy value. Videos are frame-for-frame head views at the median recorded frame rate; use the CSV for capture timing.

The triptych uses the source observation for the plotted action. Before and after are the nearest valid observations to one second either side. The global maximum is recorded separately if it lies at a boundary; in that case the triptych uses the largest interior local peak. Each archive includes original and labeled frames, CSV, PNG/PDF, state and sampled action arrays. The original RGB-D source IDs and split are in [the manifest](manifest.json).

[G00/G01 dense reference](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/1f97ca9b420e314a71ed7432e3c016ff814262af/research/appendix_followup_20261117/geowam-dense-offline) · [Earlier sparse samples](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/appendix-signals-20261112)
