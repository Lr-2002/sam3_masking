# SAM 3 Human Mask Processor

This repo contains a small processor script that uses Ultralytics SAM 3 to mask humans out of MP4 videos.

## Quick start

1) Download `sam3.pt` and place it in this repo (or point to it with `--model`).
2) Put MP4s in this repo root (or use `--input-dir`).
3) Run:

```bash
/home/lr-2002/anaconda3/envs/sam3/bin/python mask_humans.py \
  --input-dir /home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_mv-bottle_dataset/videos \
  --output-dir outputs \
  --prompt "person"
```

Outputs:
- `outputs/<video>_masked.mp4`: human pixels set to black
- `outputs/<video>_mask.mp4`: binary mask video (white = human)

## HF dataset download + mask

Use `hf_mask_dataset.py` to download a dataset from Hugging Face via `datasets`,
copy the cached files to a local dataset directory, then mask all MP4s.

```bash
/home/lr-2002/anaconda3/envs/sam3/bin/python hf_mask_dataset.py \
  --dataset-id lr-2002/exp-insert_lego \
  --progress
```

## Notes
- Default prompt is `"person"`. Use comma-separated prompts for multiple classes.
- The script looks for MP4s in the dataset path first, then `./videos`, then current directory.
- Add `--progress` to show a tqdm progress bar (install via `pip install tqdm` if needed).
- Add `--profile` to print per-video timing breakdown.
- Use `--no-save-mask` if you only want the masked video (skips writing the mask MP4).
