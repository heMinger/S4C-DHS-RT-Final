#!/usr/bin/env python3
"""Build an evenly distributed SSCBench frame pool before expensive inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


GT_RE = re.compile(r"^(\d{6})_1_1\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an evenly spaced pool of SSCBench GT frame IDs."
    )
    parser.add_argument("--voxel-gt-root", required=True, type=Path)
    parser.add_argument(
        "--sequence",
        default="2013_05_28_drive_0009_sync",
        help="Sequence directory name used by the evaluator.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=400,
        help="Number of frames distributed across the complete sequence.",
    )
    parser.add_argument(
        "--include",
        type=Path,
        default=None,
        help="Optional existing one-ID-per-line file whose frames must be retained.",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_ids(path: Path) -> List[int]:
    result: List[int] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            result.append(int(value))
        except ValueError as error:
            raise ValueError(f"Invalid frame ID at {path}:{line_number}: {value!r}") from error
    return result


def main() -> None:
    args = parse_args()
    sequence_dir = args.voxel_gt_root.expanduser() / args.sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    all_ids = []
    for path in sorted(sequence_dir.glob("*_1_1.npy")):
        match = GT_RE.match(path.name)
        if match:
            all_ids.append(int(match.group(1)))
    all_ids = sorted(set(all_ids))
    if not all_ids:
        raise RuntimeError(f"No *_1_1.npy ground-truth files found in {sequence_dir}")
    if args.num <= 0:
        raise ValueError("--num must be positive")

    count = min(args.num, len(all_ids))
    selected = []
    for index in range(count):
        position = round((index + 0.5) * len(all_ids) / count - 0.5)
        frame_id = all_ids[min(max(position, 0), len(all_ids) - 1)]
        if frame_id not in selected:
            selected.append(frame_id)
    if args.include:
        available = set(all_ids)
        for frame_id in read_ids(args.include.expanduser()):
            if frame_id in available and frame_id not in selected:
                selected.append(frame_id)
    selected.sort()

    output = args.out.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(f"{frame_id:06d}" for frame_id in selected) + "\n")
    print(f"Available GT frames: {len(all_ids)}")
    print(f"Selected frame pool: {len(selected)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
