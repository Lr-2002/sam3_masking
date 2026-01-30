#!/usr/bin/env python3
"""Mask humans out of MP4 videos using Ultralytics SAM 3."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "ERROR: OpenCV is required. Install opencv-python in the sam3 env."
    ) from exc

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "ERROR: Ultralytics SAM3VideoSemanticPredictor not available. "
        "Check that ultralytics is installed in the sam3 env."
    ) from exc

try:  # optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


DEFAULT_DATA_DIR = Path(
    "/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets/"
    "lr-2002_mv-bottle_dataset/videos"
)


def _split_prompts(prompt_text: str) -> list[str]:
    return [p.strip() for p in prompt_text.split(",") if p.strip()]


def _resolve_input_dir(user_dir: str | None) -> Path:
    if user_dir:
        return Path(user_dir)
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    local_videos = Path.cwd() / "videos"
    if local_videos.exists():
        return local_videos
    return Path.cwd()


def _collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [Path(p) for p in args.inputs]

    input_dir = _resolve_input_dir(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"ERROR: input dir not found: {input_dir}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"ERROR: no .mp4 files found in {input_dir}")
    return videos


def _read_fps(video_path: Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return None
    return fps


def _read_frame_count(video_path: Path) -> int | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if count is None or count <= 0:
        return None
    return int(count)


def _sync_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ensure_mask_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        imgsz=args.imgsz,
        model=str(Path(args.model)),
        half=args.half,
        save=False,
        verbose=args.verbose,
    )
    if args.device:
        overrides["device"] = args.device
    return overrides


def _process_video(
    src: Path,
    out_dir: Path,
    predictor: SAM3VideoSemanticPredictor,
    prompts: list[str],
    show_progress: bool,
    save_mask: bool,
    profile: bool,
    fps_fallback: float = 30.0,
) -> None:
    fps = _read_fps(src) or fps_fallback
    total_frames = _read_frame_count(src) if show_progress else None
    results = predictor(source=str(src), text=prompts, stream=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    masked_path = out_dir / f"{src.stem}_masked.mp4"
    mask_path = out_dir / f"{src.stem}_mask.mp4"

    writer = None
    mask_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if show_progress:
        if tqdm is None:
            raise SystemExit("ERROR: tqdm not installed. Run: pip install tqdm")
        pbar = tqdm(
            total=total_frames,
            unit="frame",
            desc=src.name,
            dynamic_ncols=True,
        )
    else:
        pbar = None

    frames = 0
    fetch_total = 0.0
    post_total = 0.0
    _sync_cuda()
    wall_start = time.perf_counter()

    iterator = iter(results)
    while True:
        fetch_start = time.perf_counter()
        try:
            r = next(iterator)
        except StopIteration:
            break
        _sync_cuda()
        fetch_total += time.perf_counter() - fetch_start

        post_start = time.perf_counter()
        frame = r.orig_img
        if frame is None:
            continue

        height, width = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(masked_path), fourcc, fps, (width, height))
            if save_mask:
                mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (width, height))

        mask_bool = None
        if r.masks is not None and getattr(r.masks, "data", None) is not None:
            data = r.masks.data
            if hasattr(data, "detach"):
                data = data.detach()
            if hasattr(data, "cpu"):
                data = data.cpu()
            data = data.numpy()
            if data.size:
                mask_bool = np.any(data > 0.5, axis=0)

        if mask_bool is None:
            mask_bool = np.zeros((height, width), dtype=bool)
        else:
            mask_bool = _ensure_mask_shape(mask_bool, (height, width))

        masked = frame.copy()
        masked[mask_bool] = 0

        writer.write(masked)
        if save_mask and mask_writer is not None:
            mask_frame = (mask_bool.astype(np.uint8) * 255)
            mask_frame = np.stack([mask_frame, mask_frame, mask_frame], axis=-1)
            mask_writer.write(mask_frame)

        frames += 1
        if pbar is not None:
            pbar.update(1)
        post_total += time.perf_counter() - post_start

    _sync_cuda()
    wall_total = time.perf_counter() - wall_start
    if pbar is not None:
        pbar.close()

    if writer is not None:
        writer.release()
    if mask_writer is not None:
        mask_writer.release()

    if profile and frames > 0:
        infer_avg = fetch_total / frames
        post_avg = post_total / frames
        fps_avg = frames / wall_total if wall_total > 0 else 0.0
        overhead = max(0.0, wall_total - (fetch_total + post_total))
        print(
            f"Profile {src.name}: frames={frames} "
            f"fps={fps_avg:.2f} "
            f"infer_avg={infer_avg:.4f}s "
            f"post_avg={post_avg:.4f}s "
            f"overhead={overhead:.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask humans out of MP4 videos with SAM 3 text prompts."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing MP4s. Defaults to dataset path if present, "
        "then ./videos, then current directory.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit list of MP4 files to process (overrides --input-dir).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for masked videos and mask videos.",
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


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"ERROR: model weights not found: {model_path}\n"
            "Download sam3.pt and pass --model /path/to/sam3.pt"
        )

    inputs = _collect_inputs(args)
    prompts = _split_prompts(args.prompt)
    if not prompts:
        raise SystemExit("ERROR: at least one text prompt is required.")

    overrides = _build_overrides(args)
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    out_dir = Path(args.output_dir)
    for src in inputs:
        if not src.exists():
            print(f"Skipping missing file: {src}", file=sys.stderr)
            continue
        print(f"Processing {src} -> {out_dir}")
        _process_video(
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
