#!/usr/bin/env python3
"""
查找 KITTI-360 数据集中损坏的 PNG 图片。

【Shuffle 与范围缩小】
  - train_loader 使用 shuffle=True（见 models/bts/trainer.py:656）
  - 因此无法根据「已跑 3500 步」缩小问题范围，损坏文件可能出现在数据集任意位置
  - 必须全量或按 split 扫描才能定位

用法:
  python scripts/find_corrupted_png.py
  python scripts/find_corrupted_png.py --data-path /path/to/KITTI360
  python scripts/find_corrupted_png.py --scope split  # 仅检查 split 中 sequence 的文件（更快）
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def check_png(path: Path) -> tuple[bool, str | None]:
    """检查 PNG 是否可读。返回 (ok, error_msg)。"""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False, "cv2.imread returned None"
        # 尝试 cvtColor 模拟 dataset 的实际操作
        _ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return True, None
    except cv2.error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


CAM_FOLDERS = {
    "image_00": "data_rect",
    "image_01": "data_rect",
    "image_02": "data_rgb",
    "image_03": "data_rgb",
}


def scan_directory(base_path: Path, cams: list[str] | None = None, seqs: list[str] | None = None, verbose: bool = False) -> list[tuple[Path, str]]:
    """扫描 data_2d_raw 下 PNG。cams/seqs 为 None 表示全部。"""
    corrupted = []
    raw_dir = base_path / "data_2d_raw"
    if not raw_dir.exists():
        print(f"[ERROR] 目录不存在: {raw_dir}")
        return corrupted

    cam_list = cams if cams else list(CAM_FOLDERS.keys())
    seq_dirs = sorted(raw_dir.iterdir()) if not seqs else [raw_dir / s for s in seqs if (raw_dir / s).exists()]

    total = 0
    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            continue
        for cam in cam_list:
            folder = CAM_FOLDERS.get(cam, "data_rect")
            img_dir = seq_dir / cam / folder
            if not img_dir.exists():
                continue
            for png in sorted(img_dir.glob("*.png")):
                total += 1
                if total % 5000 == 0 or total == 1:
                    print(f"  检查进度: 已扫描 {total} 个文件...")
                ok, err = check_png(png)
                if not ok:
                    corrupted.append((png, err or "unknown"))
                    print(f"[损坏] {png}  {err}")
                elif verbose:
                    print(f"[OK] {png}")

    return corrupted


def scan_split_only(base_path: Path, split_path: Path, cams: list[str] | None = None, seqs: list[str] | None = None, verbose: bool = False) -> list[tuple[Path, str]]:
    """
    仅检查 train split 中出现的 sequence 下的 PNG。
    cams/seqs 为 None 表示全部。
    """
    sequences = set()
    with open(split_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                sequences.add(parts[0])

    raw_dir = base_path / "data_2d_raw"
    if not raw_dir.exists():
        print(f"[ERROR] 目录不存在: {raw_dir}")
        return []

    seq_list = sorted(sequences) if not seqs else [s for s in seqs if s in sequences]
    cam_list = cams if cams else list(CAM_FOLDERS.keys())

    corrupted = []
    total = 0
    for seq_name in seq_list:
        seq_dir = raw_dir / seq_name
        if not seq_dir.exists():
            continue
        for cam in cam_list:
            folder = CAM_FOLDERS.get(cam, "data_rect")
            img_dir = seq_dir / cam / folder
            if not img_dir.exists():
                continue
            for png in sorted(img_dir.glob("*.png")):
                total += 1
                if total % 5000 == 0 or total == 1:
                    print(f"  检查进度: 已扫描 {total} 个文件...")
                ok, err = check_png(png)
                if not ok:
                    corrupted.append((png, err or "unknown"))
                    print(f"[损坏] {png}  {err}")
                elif verbose:
                    print(f"[OK] {png}")

    print(f"  共检查 {len(seq_list)} 个 sequence, {total} 个文件")
    return corrupted


def main():
    parser = argparse.ArgumentParser(description="查找 KITTI-360 数据集中损坏的 PNG 文件")
    parser.add_argument("--data-path", type=str, default="/mnt/disk2/lmh/KITTI360",
                        help="KITTI-360 数据根目录")
    parser.add_argument("--scope", choices=["all", "split"], default="all",
                        help="all: 扫描所有 PNG; split: 仅扫描 train split 会加载的文件（更快）")
    parser.add_argument("--split-path", type=str,
                        default="datasets/kitti_360/splits/sscbench/train_files.txt",
                        help="train split 文件路径（相对项目根或绝对路径）")
    parser.add_argument("--cam", type=str, nargs="+", default=None,
                        help="仅检查指定相机，如: --cam image_00。报错指向 image_00 时可缩小范围")
    parser.add_argument("--seq", type=str, nargs="+", default=None,
                        help="仅检查指定序列，如: --seq 2013_05_28_drive_0000_sync")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每个检查的文件")
    args = parser.parse_args()

    base_path = Path(args.data_path)
    if not base_path.exists():
        print(f"[ERROR] 数据路径不存在: {base_path}")
        sys.exit(1)

    # 项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if not args.split_path.startswith("/"):
        split_path = project_root / args.split_path
    else:
        split_path = Path(args.split_path)

    print(f"数据路径: {base_path}")
    print(f"扫描范围: {args.scope}")
    if args.cam:
        print(f"限定相机: {args.cam}")
    if args.seq:
        print(f"限定序列: {args.seq}")
    if args.scope == "split":
        print(f"Split 文件: {split_path}")
        if not split_path.exists():
            print(f"[ERROR] Split 文件不存在: {split_path}")
            sys.exit(1)

    print("开始检查 PNG 文件...")
    if args.scope == "all":
        corrupted = scan_directory(base_path, cams=args.cam, seqs=args.seq, verbose=args.verbose)
    else:
        corrupted = scan_split_only(base_path, split_path, cams=args.cam, seqs=args.seq, verbose=args.verbose)

    print("\n" + "=" * 60)
    if corrupted:
        print(f"发现 {len(corrupted)} 个损坏文件:")
        for p, err in corrupted:
            print(f"  {p}")
        # 输出到文件便于后续处理
        out_file = project_root / "corrupted_files.txt"
        with open(out_file, "w") as f:
            for p, _ in corrupted:
                f.write(str(p) + "\n")
        print(f"\n损坏文件列表已保存到: {out_file}")
        sys.exit(1)
    else:
        print("未发现损坏的 PNG 文件。")
        sys.exit(0)


if __name__ == "__main__":
    main()
