# Motus Data Format

This document describes the data formats supported by Motus.

## Overview

Motus supports three types of datasets. Each dataset type has its own directory structure:

### 1. RoboTwin 2.0 (Simulation)

```
/path/to/robotwin2/
├── clean/
│   ├── task_name/
│   │   ├── qpos/           # Robot joint positions (.pt)
│   │   │   ├── 0.pt
│   │   │   └── ...
│   │   ├── videos/         # MP4 video files
│   │   │   ├── 0.mp4
│   │   │   └── ...
│   │   └── umt5_wan/       # Pre-encoded T5 language embeddings (.pt)
│   │       ├── 0.pt
│   │       └── ...
│   └── ...
└── randomized/
    └── ... (same structure)
```

**📖 Data Conversion Guide:** [RoboTwin Data Conversion](data/robotwin2/robotwin_data_convert/README.md)

### 2. Real-World Robot Data (AC-One, Aloha-Agilex-2)

```
/path/to/ac_one/
├── task_category/
│   ├── task_variant/
│   │   ├── videos/
│   │   │   ├── 0.mp4
│   │   │   └── ...
│   │   ├── qpos/           # Robot joint positions (.pt)
│   │   │   ├── 0.pt
│   │   │   └── ...
│   │   └── instructions/   # Language instructions
│   │       ├── task.txt    # Text instruction
│   │       └── task.pt     # Pre-encoded T5 embedding
│   └── ...
└── ...
```

### 3. Latent Action Pretraining (Stage 2)

```
/path/to/latent_action_data/
├── videos/                 # MP4 video files
│   ├── episode_0.mp4
│   └── ...
├── umt5_wan/              # Pre-encoded T5 language embeddings
│   ├── episode_0.pt
│   └── ...
└── latent_action_dim14/   # Latent action labels (from optical flow)
    ├── episode_0.pt
    └── ...
```

## Configuration

**Configure dataset in YAML:**
```yaml
dataset:
  type: robotwin           # Options: robotwin, ac_one, aloha_agilex_2, latent_action
  dataset_dir: /path/to/dataset
  data_mode: both          # For robotwin: clean, randomized, or both
  task_mode: multi         # single or multi
```

---

**📖 Related Guides:**
- [Training Guide](TRAINING.md)
- [Inference Guide](INFERENCE.md)
- [RoboTwin Data Conversion](data/robotwin2/robotwin_data_convert/README.md)
- [Multi-Camera Concatenation](data/utils/multi_camera_concat.py)
