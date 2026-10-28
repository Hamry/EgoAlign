#!/usr/bin/env bash
# One human pick video -> sim robot dataset -> SmolVLA fine-tune -> sim eval.
# Needs an NVIDIA GPU (24 GB+) and Python >= 3.12. Examples:
#   VIDEO=pick.mp4 MARKER_X=0.35 MARKER_Y=0.25 bash cloud/run_all.sh
#   SYNTHETIC=1 STEPS=2000 bash cloud/run_all.sh   # smoke test, no video
set -euo pipefail

TASK="${TASK:-pick up the mug}"
EPISODES="${EPISODES:-50}"
STEPS="${STEPS:-20000}"
WORK="${WORK:-work}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

pip install -q -e . "lerobot[smolvla,dataset]==0.6.1"
# egoalign, mediapipe and lerobot each pull a different OpenCV wheel.
python -c "import cv2" || {
  echo "cv2 import failed; try: apt-get install -y libgl1 libglib2.0-0" >&2
  exit 1
}
# Stale outputs would leak into the dataset or block lerobot-train.
rm -rf "$WORK/demo" "$WORK/rollouts" "$WORK/dataset" "$WORK/train"

if [[ -n "${SYNTHETIC:-}" ]]; then
  DEMO="$WORK/demo/synthetic.npz"
  python cloud/make_synthetic_demo.py "$DEMO" --instruction "$TASK"
else
  : "${VIDEO:?set VIDEO to the recorded demo (or SYNTHETIC=1)}"
  : "${MARKER_X:?set MARKER_X, marker centre x in the robot frame [m]}"
  : "${MARKER_Y:?set MARKER_Y, marker centre y in the robot frame [m]}"
  egoalign process "$VIDEO" --instruction "$TASK" --no-embed \
    --marker-xy "$MARKER_X" "$MARKER_Y" \
    --marker-size "${MARKER_SIZE:-0.1}" --fov "${FOV:-65}" --out "$WORK/demo"
  stem="$(basename "$VIDEO")"
  DEMO="$WORK/demo/${stem%.*}.npz"
  if [[ ! -f "$DEMO" ]]; then
    echo "demo was rejected by the kinematic filter" >&2
    exit 1
  fi
fi

egoalign augment "$DEMO" --episodes "$EPISODES" --out "$WORK/rollouts"
python cloud/to_lerobot.py "$WORK/rollouts" --root "$WORK/dataset"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/egoalign_pick \
  --dataset.root="$WORK/dataset" \
  --batch_size=32 \
  --steps="$STEPS" \
  --output_dir="$WORK/train" \
  --job_name=egoalign_smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

python cloud/eval_smolvla.py \
  --policy "$WORK/train/checkpoints/last/pretrained_model" \
  --demo "$DEMO" --video "$WORK/eval_episode0.mp4"
