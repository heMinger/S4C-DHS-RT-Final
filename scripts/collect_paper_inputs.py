#!/usr/bin/env python3
"""Collect the three KITTI-360 left-camera images used in the paper figure."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_FRAMES = (300, 5785, 12200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssc-root",
        required=True,
        type=Path,
        help="SSCBench-KITTI-360 root containing data_2d_raw/.",
    )
    parser.add_argument("--sequence", type=int, default=9)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    args = parser.parse_args()

    sequence = f"2013_05_28_drive_00{args.sequence:02d}_sync"
    image_dir = (
        args.ssc_root.expanduser()
        / "data_2d_raw"
        / sequence
        / "image_00"
        / "data_rect"
    )
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for frame in args.frames:
        candidates = (image_dir / f"{frame:06d}.png", image_dir / f"{frame:010d}.png")
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            missing.append((frame, candidates))
            continue
        destination = out_dir / f"{frame:06d}.png"
        shutil.copy2(source, destination)
        copied.append(destination)

    for destination in copied:
        print(f"Copied: {destination}")
    if missing:
        for frame, candidates in missing:
            print(f"Missing frame {frame:06d}; tried:")
            for candidate in candidates:
                print(f"  {candidate}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
