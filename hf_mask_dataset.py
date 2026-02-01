#!/usr/bin/env python3
"""Download an HF dataset via datasets, copy cache to local, then mask videos."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import datasets
from datasets import config as ds_config

try:  # optional, used for hub cache discovery
    from huggingface_hub import constants as hub_constants
except Exception:  # pragma: no cover - optional dependency
    hub_constants = None

import mask_humans


DEFAULT_LOCAL_ROOT = Path("/home/lr-2002/project/DataArm/dataarm/data/lerobot_datasets")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/lr-2002/project/DataArm/dataarm/third-plays/mask_processing/outputs"
)
DEFAULT_RELATIVE_TO = Path("/home/lr-2002/project/DataArm/dataarm/data")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HF dataset, copy to local, then mask all MP4 videos."
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help='HF dataset id, e.g. "lr-2002/exp-insert_lego".',
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to trigger download (default: all).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config name if the dataset defines configs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache directory for datasets downloads.",
    )
    parser.add_argument(
        "--local-root",
        default=str(DEFAULT_LOCAL_ROOT),
        help="Root directory to copy the dataset into.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Explicit local dataset directory (overrides --local-root).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for masked videos and masks.",
    )
    parser.add_argument(
        "--relative-to",
        default=str(DEFAULT_RELATIVE_TO),
        help="Base path used to preserve relative output structure.",
    )
    parser.add_argument(
        "--videos-subdir",
        default="videos",
        help='Subdirectory under the dataset that contains videos (default: "videos").',
    )
    parser.add_argument(
        "--overwrite-copy",
        action="store_true",
        help="Overwrite local files during cache copy.",
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


def _dataset_to_local_dir(dataset_id: str, local_root: Path) -> Path:
    if "/" in dataset_id:
        owner, name = dataset_id.split("/", 1)
        return local_root / f"{owner}_{name}_dataset"
    return local_root / f"{dataset_id}_dataset"


def _normalize_token(token: str) -> set[str]:
    token = token.strip()
    if not token:
        return set()
    return {token, token.replace("-", "_")}


def _cache_roots(cache_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir:
        roots.append(Path(cache_dir))
        roots.append(Path(cache_dir) / "downloads")
    roots.append(Path(ds_config.HF_DATASETS_CACHE))
    roots.append(Path(ds_config.DOWNLOADED_DATASETS_PATH))
    if hub_constants is not None:
        roots.append(Path(hub_constants.HF_HUB_CACHE))
    return [root for root in roots if root.exists()]


def _matches_tokens(path: str, tokens: set[str]) -> bool:
    if not tokens:
        return True
    path_lower = path.lower()
    return any(token.lower() in path_lower for token in tokens)


def _find_candidate_roots(dataset_id: str, roots: list[Path]) -> list[Path]:
    owner = ""
    name = dataset_id
    if "/" in dataset_id:
        owner, name = dataset_id.split("/", 1)
    owner_tokens = _normalize_token(owner)
    name_tokens = _normalize_token(name)

    candidates: list[Path] = []
    for root in roots:
        for videos_dir in root.rglob("videos"):
            if not videos_dir.is_dir():
                continue
            path_str = str(videos_dir)
            if owner_tokens and not _matches_tokens(path_str, owner_tokens):
                continue
            if name_tokens and not _matches_tokens(path_str, name_tokens):
                continue
            if next(videos_dir.rglob("*.mp4"), None) is None:
                continue
            candidates.append(videos_dir.parent)
    return candidates


def _select_best_root(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None

    def count_mp4(root: Path) -> int:
        videos_dir = root / "videos"
        if not videos_dir.exists():
            return 0
        return sum(1 for _ in videos_dir.rglob("*.mp4"))

    best_root = None
    best_count = -1
    for root in candidates:
        count = count_mp4(root)
        if count > best_count:
            best_root = root
            best_count = count
    return best_root


def _copy_tree(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        dest = dst_root / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if dest.exists() and not overwrite:
            try:
                if dest.stat().st_size == path.stat().st_size:
                    continue
            except OSError:
                pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def _iter_videos(dataset_dir: Path, videos_subdir: str) -> list[Path]:
    if videos_subdir:
        candidate = dataset_dir / videos_subdir
        search_root = candidate if candidate.exists() else dataset_dir
    else:
        search_root = dataset_dir
    return sorted(search_root.rglob("*.mp4"))


def _relative_parent(src: Path, base: Path | None, fallback_base: Path | None) -> Path:
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

    # Trigger HF download via datasets.
    load_kwargs = dict(
        path=args.dataset_id,
        name=args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=True,
    )
    try:
        datasets.load_dataset(**load_kwargs)
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to load dataset {args.dataset_id}: {exc}") from exc

    cache_roots = _cache_roots(args.cache_dir)
    candidates = _find_candidate_roots(args.dataset_id, cache_roots)
    dataset_root = _select_best_root(candidates)
    if dataset_root is None:
        raise SystemExit(
            "ERROR: could not locate downloaded dataset in HF cache. "
            "Try setting --cache-dir or inspect ~/.cache/huggingface."
        )

    local_root = Path(args.local_root)
    local_dir = Path(args.local_dir) if args.local_dir else _dataset_to_local_dir(
        args.dataset_id, local_root
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying from cache {dataset_root} -> {local_dir}")
    _copy_tree(dataset_root, local_dir, args.overwrite_copy)

    prompts = mask_humans._split_prompts(args.prompt)
    if not prompts:
        raise SystemExit("ERROR: at least one text prompt is required.")

    overrides = mask_humans._build_overrides(args)
    predictor = mask_humans.SAM3VideoSemanticPredictor(overrides=overrides)

    videos = _iter_videos(local_dir, args.videos_subdir)
    if not videos:
        raise SystemExit(f"ERROR: no .mp4 files found under {local_dir}")

    output_root = Path(args.output_root)
    relative_to = Path(args.relative_to) if args.relative_to else None
    for src in videos:
        rel_parent = _relative_parent(src, relative_to, local_dir)
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
