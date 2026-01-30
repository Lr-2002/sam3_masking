#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/lr-2002/anaconda3/envs/sam3/bin/python"
MODEL_PATH="${MODEL_PATH:-sam3.pt}"
INPUT_DIR="${INPUT_DIR:-/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_mv-bottle_dataset/videos/observation.images.cam_top/chunk-000/}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
PROMPT="${PROMPT:-human}"
SAVE_MASK="${SAVE_MASK:-1}"
PROFILE="${PROFILE:-0}"
PROGRESS="${PROGRESS:-1}"

SAVE_MASK_FLAG=""
PROFILE_FLAG=""
PROGRESS_FLAG=""
if [ "$SAVE_MASK" = "0" ] || [ "$SAVE_MASK" = "false" ]; then
  SAVE_MASK_FLAG="--no-save-mask"
fi
if [ "$PROFILE" = "1" ] || [ "$PROFILE" = "true" ]; then
  PROFILE_FLAG="--profile"
fi
if [ "$PROGRESS" = "1" ] || [ "$PROGRESS" = "true" ]; then
  PROGRESS_FLAG="--progress"
fi

"$PYTHON" mask_humans.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT" \
  $SAVE_MASK_FLAG \
  $PROFILE_FLAG \
  $PROGRESS_FLAG
