#!/usr/bin/env python3
"""Select and render SSC qualitative-result candidates.

The script first selects candidate IDs from GT/reference PLYs without requiring
all repositories to already contain those PLYs.  It writes one shared PLY_IDS
list and a missing-file manifest.  After the missing PLYs have been generated,
run it again to render all methods with the same Blender script, apply one fixed
crop, and create four-column panels plus an overview contact sheet.

PLY vertex count is not an evaluation metric.  It is used only to avoid
spending Blender time on extremely sparse or obviously uninformative frames.
Final frame selection must be made visually and checked against quantitative
results.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


FRAME_RE = re.compile(r"^(\d+)(?:_gt)?\.ply$")
METHODS = ("gt", "s4c", "dhs", "rt")
LABELS = {"gt": "GT", "s4c": "S4C", "dhs": "DHS", "rt": "DHS+RT"}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    gt_vertices: int
    s4c_vertices: int
    dhs_vertices: int
    rt_vertices: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-screen and render four-column SSC qualitative candidates."
    )
    parser.add_argument("--s4c-dir", required=True, type=Path)
    parser.add_argument("--dhs-dir", required=True, type=Path)
    parser.add_argument("--rt-dir", required=True, type=Path)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Directory containing <frame>_gt.ply; defaults to --s4c-dir.",
    )
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path("~/blender-4.2.0-linux-x64/blender"),
    )
    parser.add_argument("--out", type=Path, default=Path("out/qualitative_candidates"))
    parser.add_argument(
        "--num",
        type=int,
        default=12,
        help="Number of candidates to render after pre-screening.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        help="Explicit frame IDs. If set, automatic pre-screening is skipped.",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0, 250, 2400, 1100),
        help="One fixed crop box applied to every 2400x1350 render.",
    )
    parser.add_argument("--panel-width", type=int, default=700)
    parser.add_argument("--overview-columns", type=int, default=2)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Reuse existing PNG files and rebuild panels only.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write selected frame IDs and missing-PLY manifest without rendering.",
    )
    return parser.parse_args()


def expanded(path: Path) -> Path:
    return path.expanduser().resolve()


def collect_prediction_ids(directory: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in directory.glob("*.ply"):
        match = FRAME_RE.match(path.name)
        if match and not path.stem.endswith("_gt"):
            result[int(match.group(1))] = path
    return result


def collect_gt_ids(directory: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in directory.glob("*_gt.ply"):
        match = FRAME_RE.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def ply_vertex_count(path: Path) -> int:
    """Read `element vertex N` from an ASCII or binary PLY header."""
    with path.open("rb") as handle:
        header = handle.read(65536).decode("ascii", errors="ignore")
    match = re.search(r"^element vertex\s+(\d+)\s*$", header, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find vertex count in PLY header: {path}")
    return int(match.group(1))


def normalized_count_error(pred: int, gt: int) -> float:
    return abs(pred - gt) / max(gt, 1)


def build_records(
    ids: Iterable[int], paths: Dict[str, Dict[int, Path]]
) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for frame_id in sorted(ids):
        counts = {name: ply_vertex_count(paths[name][frame_id]) for name in METHODS}
        gt = counts["gt"]
        s4c_error = normalized_count_error(counts["s4c"], gt)
        dhs_error = normalized_count_error(counts["dhs"], gt)
        rt_error = normalized_count_error(counts["rt"], gt)

        # Prefer frames where the proposed stages move total occupancy closer
        # to GT, while suppressing nearly empty scenes.  Spatial correctness is
        # intentionally left to visual inspection of the generated panels.
        improvement = s4c_error - 0.5 * (dhs_error + rt_error)
        occupancy_support = min(1.0, gt / 200_000.0)
        score = improvement + 0.15 * occupancy_support
        records.append(
            FrameRecord(
                frame_id=frame_id,
                gt_vertices=counts["gt"],
                s4c_vertices=counts["s4c"],
                dhs_vertices=counts["dhs"],
                rt_vertices=counts["rt"],
                score=score,
            )
        )
    return records


def diverse_top_records(records: Sequence[FrameRecord], num: int) -> List[FrameRecord]:
    """Mix high-scoring and uniformly spread records.

    Half of the pool is selected by the cheap occupancy-count heuristic; the
    other half is sampled across the shared sequence.  This keeps the contact
    sheet useful without silently reducing it to cherry-picked best cases.
    """
    if num <= 0:
        return []
    ranked = sorted(records, key=lambda item: item.score, reverse=True)
    if len(ranked) <= num:
        return sorted(ranked, key=lambda item: item.frame_id)

    all_ids = sorted(item.frame_id for item in records)
    span = max(all_ids[-1] - all_ids[0], 1)
    min_gap = max(1, span // max(num * 3, 1))
    selected: List[FrameRecord] = []
    top_quota = max(1, num // 2)
    for record in ranked:
        if all(abs(record.frame_id - old.frame_id) >= min_gap for old in selected):
            selected.append(record)
        if len(selected) == top_quota:
            break

    chronological = sorted(records, key=lambda item: item.frame_id)
    uniform_quota = num - len(selected)
    if uniform_quota > 0:
        for index in range(uniform_quota):
            position = round((index + 0.5) * len(chronological) / uniform_quota - 0.5)
            candidate = chronological[min(max(position, 0), len(chronological) - 1)]
            if candidate.frame_id not in {item.frame_id for item in selected}:
                selected.append(candidate)
    if len(selected) < num:
        selected_ids = {item.frame_id for item in selected}
        for record in ranked:
            if record.frame_id not in selected_ids:
                selected.append(record)
                selected_ids.add(record.frame_id)
            if len(selected) == num:
                break
    return sorted(selected, key=lambda item: item.frame_id)


def write_csv(path: Path, records: Sequence[FrameRecord], selected: set[int]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_id", "selected", "score", "gt_vertices", "s4c_vertices",
             "dhs_vertices", "rt_vertices"]
        )
        for record in sorted(records, key=lambda item: item.score, reverse=True):
            writer.writerow(
                [record.frame_id, int(record.frame_id in selected), f"{record.score:.6f}",
                 record.gt_vertices, record.s4c_vertices, record.dhs_vertices,
                 record.rt_vertices]
            )


def render_one(blender: Path, renderer: Path, ply: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(blender), "--background", "--python", str(renderer), "--",
        "--ply", str(ply), "--out", str(output),
    ]
    print("[render]", " ".join(command))
    subprocess.run(command, check=True)


def font(size: int = 28) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def crop_and_resize(path: Path, crop: Tuple[int, int, int, int], width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    left, top, right, bottom = crop
    left = max(0, min(left, image.width - 1))
    top = max(0, min(top, image.height - 1))
    right = max(left + 1, min(right, image.width))
    bottom = max(top + 1, min(bottom, image.height))
    image = image.crop((left, top, right, bottom))
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_panel(
    frame_id: int,
    rendered: Dict[str, Path],
    output: Path,
    crop: Tuple[int, int, int, int],
    panel_width: int,
) -> None:
    images = [crop_and_resize(rendered[name], crop, panel_width) for name in METHODS]
    label_height = 55
    canvas = Image.new(
        "RGB", (panel_width * len(images), images[0].height + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(28)
    for index, (name, image) in enumerate(zip(METHODS, images)):
        x = index * panel_width
        canvas.paste(image, (x, label_height))
        label = f"{LABELS[name]}   frame {frame_id:06d}"
        bbox = draw.textbbox((0, 0), label, font=title_font)
        text_width = bbox[2] - bbox[0]
        draw.text((x + (panel_width - text_width) / 2, 11), label, fill="black", font=title_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def make_overview(panel_paths: Sequence[Path], output: Path, columns: int) -> None:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    if not panels:
        return
    overview_width = 1800
    thumb_width = overview_width // columns
    thumbs: List[Image.Image] = []
    for panel in panels:
        thumb_height = round(panel.height * thumb_width / panel.width)
        thumbs.append(panel.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
    rows = math.ceil(len(thumbs) / columns)
    canvas = Image.new("RGB", (overview_width, rows * thumbs[0].height), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb.height
        canvas.paste(thumb, (x, y))
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    directories = {
        "s4c": expanded(args.s4c_dir),
        "dhs": expanded(args.dhs_dir),
        "rt": expanded(args.rt_dir),
    }
    gt_dir = expanded(args.gt_dir) if args.gt_dir else directories["s4c"]
    paths: Dict[str, Dict[int, Path]] = {
        "gt": collect_gt_ids(gt_dir),
        "s4c": collect_prediction_ids(directories["s4c"]),
        "dhs": collect_prediction_ids(directories["dhs"]),
        "rt": collect_prediction_ids(directories["rt"]),
    }
    # Candidate IDs come from GT/reference files, not from an intersection of
    # all repositories. Missing predictions are reported so they can be
    # generated with exactly the same PLY_IDS list.
    reference_ids = sorted(paths["gt"])
    if not reference_ids and not args.frames:
        raise SystemExit("No *_gt.ply reference frames were found in --gt-dir.")

    if args.frames:
        selected_ids_ordered = list(dict.fromkeys(args.frames))
    else:
        count = min(args.num, len(reference_ids))
        selected_ids_ordered = []
        for index in range(count):
            position = round((index + 0.5) * len(reference_ids) / count - 0.5)
            frame_id = reference_ids[min(max(position, 0), len(reference_ids) - 1)]
            if frame_id not in selected_ids_ordered:
                selected_ids_ordered.append(frame_id)

    output_dir = expanded(args.out)
    raw_dir = output_dir / "raw"
    panel_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = set(selected_ids_ordered)
    (output_dir / "selected_frames.txt").write_text(
        "\n".join(f"{frame_id:06d}" for frame_id in selected_ids_ordered) + "\n"
    )
    (output_dir / "ply_ids_python.txt").write_text(
        "PLY_IDS = [" + ", ".join(str(frame_id) for frame_id in selected_ids_ordered) + "]\n"
    )

    missing_lines: List[str] = []
    for frame_id in selected_ids_ordered:
        for name in METHODS:
            if frame_id not in paths[name]:
                suffix = "_gt.ply" if name == "gt" else ".ply"
                missing_lines.append(f"{name}\t{frame_id:06d}{suffix}")
    missing_path = output_dir / "missing_plys.txt"
    missing_path.write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""))

    if args.prepare_only or missing_lines:
        print(f"Selected IDs: {output_dir / 'ply_ids_python.txt'}")
        print(f"Missing PLYs: {missing_path}")
        if missing_lines and not args.prepare_only:
            print("Generate the missing PLYs in the indicated repositories, then rerun.")
        return

    if Image is None:
        raise SystemExit(
            "Pillow is required only for rendering contact sheets. Install it "
            "in the active environment with: conda install pillow"
        )

    records = build_records(selected_ids_ordered, paths)
    write_csv(output_dir / "candidate_stats.csv", records, selected_ids)

    blender = expanded(args.blender)
    renderer = expanded(args.renderer)
    panel_paths: List[Path] = []
    for frame_id in selected_ids_ordered:
        rendered: Dict[str, Path] = {}
        for name in METHODS:
            output = raw_dir / name / f"{frame_id:06d}.png"
            rendered[name] = output
            if not args.skip_render:
                render_one(blender, renderer, paths[name][frame_id], output)
            elif not output.exists():
                raise SystemExit(f"Missing cached render for --skip-render: {output}")
        panel_path = panel_dir / f"{frame_id:06d}.png"
        make_panel(frame_id, rendered, panel_path, tuple(args.crop), args.panel_width)
        panel_paths.append(panel_path)

    make_overview(panel_paths, output_dir / "overview.png", max(1, args.overview_columns))
    print(f"\nSelected frames: {', '.join(f'{item:06d}' for item in selected_ids_ordered)}")
    print(f"Overview: {output_dir / 'overview.png'}")
    print(f"Statistics: {output_dir / 'candidate_stats.csv'}")


if __name__ == "__main__":
    main()

