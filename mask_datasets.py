#!/usr/bin/env python3
"""Batch mask videos for multiple datasets, preserving original paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mask_humans


DEFAULT_DATA_ROOT = Path("/home/lr-2002/project/DataArm/dataarm/data")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/lr-2002/project/DataArm/dataarm/third-plays/mask_processing/outputs"
)
DEFAULT_DATASETS = [
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-insert_lego_dataset",
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-isert_lego_dataset",
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-move_bottle_dataset",
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-pick_chicken_dataset",
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-press_button_dataset",
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/lr-2002_exp-put_ball_into_cup_dataset",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask humans for multiple dataset folders, preserving paths."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset root directories to scan for videos.",
    )
    parser.add_argument(
        "--videos-subdir",
        default="videos",
        help='Subdirectory under each dataset that contains videos (default: "videos").',
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory to mirror the input paths under.",
    )
    parser.add_argument(
        "--relative-to",
        default=str(DEFAULT_DATA_ROOT),
        help="Base directory to preserve relative paths from.",
    )
    parser.add_argument(
        "--model",
        default="sam3.pt",
        help="Path to SAM 3 weights (sam3.pt).",
    )
    parser.add_argument(
        "--prompt",
        default="person",
        help='Comma-separated text prompts (default: "person").',
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        default="",
        help='Device string for Ultralytics (e.g. "cuda:0", "cpu").',
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 (recommended only on GPU).",
    )
    parser.add_argument(
        "--force-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force model to float32 to avoid mixed dtype errors "
            "(default: True). Use --no-force-fp32 to disable."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable Ultralytics verbose logging.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar per video.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-video timing breakdown (inference vs postprocess/write).",
    )
    parser.add_argument(
        "--save-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save binary mask video (default: True). Use --no-save-mask to disable.",
    )
    return parser.parse_args()


def _iter_videos(dataset_dir: Path, videos_subdir: str) -> list[Path]:
    if videos_subdir:
        candidate = dataset_dir / videos_subdir
        search_root = candidate if candidate.exists() else dataset_dir
    else:
        search_root = dataset_dir
    return sorted(search_root.rglob("*.mp4"))


def _relative_parent(
    src: Path, base: Path | None, fallback_base: Path | None
) -> Path:
    if base is not None:
        try:
            return src.relative_to(base).parent
        except ValueError:
            pass
    if fallback_base is not None:
        try:
            return src.relative_to(fallback_base).parent
        except ValueError:
            pass
    return Path()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"ERROR: model weights not found: {model_path}\n"
            "Download sam3.pt and pass --model /path/to/sam3.pt"
        )

    prompts = mask_humans._split_prompts(args.prompt)
    if not prompts:
        raise SystemExit("ERROR: at least one text prompt is required.")

    overrides = mask_humans._build_overrides(args)
    predictor = mask_humans.SAM3VideoSemanticPredictor(overrides=overrides)
    mask_humans._maybe_force_fp32(predictor, args.force_fp32, args.half)

    output_root = Path(args.output_root)
    relative_to = Path(args.relative_to) if args.relative_to else None

    dataset_dirs = [Path(d) for d in args.datasets]
    missing = [str(d) for d in dataset_dirs if not d.exists()]
    if missing:
        print("WARNING: missing dataset dirs:", file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        videos = _iter_videos(dataset_dir, args.videos_subdir)
        if not videos:
            print(f"WARNING: no .mp4 files found under {dataset_dir}", file=sys.stderr)
            continue
        for src in videos:
            rel_parent = _relative_parent(src, relative_to, dataset_dir)
            out_dir = output_root / rel_parent
            print(f"Processing {src} -> {out_dir}")
            mask_humans._process_video(
                src,
                out_dir,
                predictor,
                prompts,
                args.progress,
                args.save_mask,
                args.profile,
            )


if __name__ == "__main__":
    main()
