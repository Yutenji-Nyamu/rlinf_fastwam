set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
XPL="$ROBOTWIN/XPolicyLab"
ACT_DIR="$XPL/policy/ACT"
OUT=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/06_act_train_smoke_1epoch_not_for_eval

test ! -e "$OUT"
test -f "$ACT_DIR/processed_data/demo_clean/adjust_bottle/aloha_agilex-joint/episode_49.hdf5"
mkdir -p "$OUT"

export ACT_ACTION_DIM="$(bash "$XPL/utils/get_action_dim.sh" "$ROBOTWIN" aloha_agilex)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TORCH_HOME=/home/chenyiteng/.cache/torch

cd "$ACT_DIR"
nvidia-smi --id=0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

timeout --signal=INT --kill-after=60s 1800s \
  python3 imitate_episodes.py \
    --bench_name demo_clean \
    --task_name adjust_bottle \
    --ckpt_setting demo_clean-adjust_bottle-aloha_agilex-joint \
    --ckpt_dir "$OUT" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --num_epochs 1 \
    --lr 1e-5 \
    --save_freq 1 \
    --seed 0 2>&1 | tee "$OUT/train.log"

test -s "$OUT/dataset_stats.pkl"
test -s "$OUT/policy_last.ckpt"
find "$OUT" -maxdepth 1 -type f -printf '%s\t%f\n' | sort
