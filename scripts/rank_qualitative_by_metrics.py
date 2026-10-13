#!/usr/bin/env python3
"""Rank SSC qualitative candidates from official per-frame metric CSV files.

The three CSV files must be produced by the same SSCBench evaluation code,
split, masks, range and sigma threshold.  The script deliberately emits
separate geometry and semantic rankings instead of treating one opaque score
as proof that a frame is qualitatively better.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge S4C/DHS/RT per-frame metrics and rank qualitative candidates."
    )
    parser.add_argument("--s4c", required=True, type=Path, help="S4C per-frame CSV")
    parser.add_argument("--dhs", required=True, type=Path, help="DHS per-frame CSV")
    parser.add_argument("--rt", required=True, type=Path, help="DHS+RT per-frame CSV")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--range",
        dest="range_key",
        choices=("12p8", "25p6", "51p2"),
        default="51p2",
    )
    parser.add_argument("--top", type=int, default=12, help="Rows per ranking table")
    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=100,
        help="Suppress near-duplicate frames within each shortlist.",
    )
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=8.0,
        help="Maximum allowed Ours-vs-S4C recall drop in percentage points.",
    )
    parser.add_argument(
        "--max-iou-drop",
        type=float,
        default=3.0,
        help="Maximum allowed Ours-vs-S4C occupancy-IoU drop in percentage points.",
    )
    parser.add_argument(
        "--min-valid-classes",
        type=int,
        default=4,
        help="Ignore visually trivial frames with too few GT semantic classes.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> Dict[int, Dict[str, str]]:
    with path.expanduser().open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "frame_id" not in rows[0]:
        raise ValueError(f"Missing frame_id or no rows in {path}")
    return {int(row["frame_id"]): row for row in rows}


def number(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value.strip() == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def delta(a: float, b: float) -> float:
    return a - b if finite(a) and finite(b) else float("nan")


def diverse_top(rows: Sequence[dict], key: str, count: int, min_gap: int) -> List[dict]:
    ranked = sorted(
        (row for row in rows if finite(float(row[key]))),
        key=lambda row: float(row[key]),
        reverse=True,
    )
    selected: List[dict] = []
    for row in ranked:
        frame_id = int(row["frame_id"])
        if all(abs(frame_id - int(old["frame_id"])) >= min_gap for old in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    return selected


def write_rows(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return "nan" if not finite(value) else f"{value:+.2f}"


def markdown_table(title: str, rows: Sequence[dict]) -> List[str]:
    lines = [f"## {title}", "", "| Frame | ΔPrec | ΔRecall | ΔIoU | Δvalid-mIoU | DHS ΔPrec | RT ΔmIoU |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {frame_id:06d} | {d_prec} | {d_rec} | {d_iou} | {d_miou} | {dhs_prec} | {rt_miou} |".format(
                frame_id=int(row["frame_id"]),
                d_prec=fmt(row["ours_vs_s4c_precision_pp"]),
                d_rec=fmt(row["ours_vs_s4c_recall_pp"]),
                d_iou=fmt(row["ours_vs_s4c_iou_pp"]),
                d_miou=fmt(row["ours_vs_s4c_valid_miou_pp"]),
                dhs_prec=fmt(row["dhs_vs_s4c_precision_pp"]),
                rt_miou=fmt(row["rt_vs_dhs_valid_miou_pp"]),
            )
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    s4c = read_csv(args.s4c)
    dhs = read_csv(args.dhs)
    rt = read_csv(args.rt)
    shared = sorted(set(s4c) & set(dhs) & set(rt))
    if not shared:
        raise RuntimeError("The three CSV files have no shared frame IDs")

    prefix = args.range_key
    metric_keys = {
        "precision": f"precision_{prefix}_pct",
        "recall": f"recall_{prefix}_pct",
        "iou": f"iou_{prefix}_pct",
        "miou": f"valid_miou_{prefix}_pct",
        "valid_classes": f"valid_classes_{prefix}",
        "gt_occ": f"gt_occ_{prefix}",
    }
    rows: List[dict] = []
    class_columns = sorted(
        key for key in s4c[shared[0]]
        if key.startswith(f"iou_{prefix}_") and key.endswith("_pct")
        and key != f"iou_{prefix}_pct"
    )

    for frame_id in shared:
        values = {}
        for model_name, source in (("s4c", s4c), ("dhs", dhs), ("rt", rt)):
            for metric_name, csv_key in metric_keys.items():
                values[f"{model_name}_{metric_name}"] = number(source[frame_id], csv_key)

        row = {
            "frame_id": frame_id,
            **values,
            "ours_vs_s4c_precision_pp": delta(values["rt_precision"], values["s4c_precision"]),
            "ours_vs_s4c_recall_pp": delta(values["rt_recall"], values["s4c_recall"]),
            "ours_vs_s4c_iou_pp": delta(values["rt_iou"], values["s4c_iou"]),
            "ours_vs_s4c_valid_miou_pp": delta(values["rt_miou"], values["s4c_miou"]),
            "dhs_vs_s4c_precision_pp": delta(values["dhs_precision"], values["s4c_precision"]),
            "dhs_vs_s4c_recall_pp": delta(values["dhs_recall"], values["s4c_recall"]),
            "dhs_vs_s4c_iou_pp": delta(values["dhs_iou"], values["s4c_iou"]),
            "dhs_vs_s4c_valid_miou_pp": delta(values["dhs_miou"], values["s4c_miou"]),
            "rt_vs_dhs_precision_pp": delta(values["rt_precision"], values["dhs_precision"]),
            "rt_vs_dhs_recall_pp": delta(values["rt_recall"], values["dhs_recall"]),
            "rt_vs_dhs_iou_pp": delta(values["rt_iou"], values["dhs_iou"]),
            "rt_vs_dhs_valid_miou_pp": delta(values["rt_miou"], values["dhs_miou"]),
        }
        for column in class_columns:
            class_name = column[len(f"iou_{prefix}_"):-len("_pct")]
            row[f"ours_vs_s4c_{class_name}_iou_pp"] = delta(
                number(rt[frame_id], column), number(s4c[frame_id], column)
            )
            row[f"rt_vs_dhs_{class_name}_iou_pp"] = delta(
                number(rt[frame_id], column), number(dhs[frame_id], column)
            )

        # A transparent display score used only to order the balanced shortlist.
        # Semantic improvement leads; precision and IoU are supporting terms.
        # Recall loss beyond 3 pp is explicitly penalized.
        recall_excess = max(0.0, -row["ours_vs_s4c_recall_pp"] - 3.0)
        row["balanced_score"] = (
            row["ours_vs_s4c_valid_miou_pp"]
            + 0.35 * row["ours_vs_s4c_precision_pp"]
            + 0.15 * row["ours_vs_s4c_iou_pp"]
            - 0.35 * recall_excess
        )
        rows.append(row)

    eligible = [
        row for row in rows
        if row["s4c_valid_classes"] >= args.min_valid_classes
        and row["ours_vs_s4c_recall_pp"] >= -args.max_recall_drop
        and row["ours_vs_s4c_iou_pp"] >= -args.max_iou_drop
    ]
    geometry_pool = [
        row for row in eligible
        if row["dhs_vs_s4c_precision_pp"] > 0
    ]
    rt_pool = [
        row for row in eligible
        if row["rt_vs_dhs_valid_miou_pp"] > 0
    ]
    semantic_pool = [
        row for row in eligible
        if row["ours_vs_s4c_valid_miou_pp"] > 0
    ]

    geometry = diverse_top(geometry_pool, "dhs_vs_s4c_precision_pp", args.top, args.min_frame_gap)
    rt_semantic = diverse_top(rt_pool, "rt_vs_dhs_valid_miou_pp", args.top, args.min_frame_gap)
    overall_semantic = diverse_top(semantic_pool, "ours_vs_s4c_valid_miou_pp", args.top, args.min_frame_gap)
    balanced = diverse_top(eligible, "balanced_score", args.top, args.min_frame_gap)

    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    out = args.out.expanduser()
    all_fields = list(rows[0].keys())
    write_rows(out / "ranked_all.csv", sorted(rows, key=lambda row: row["balanced_score"], reverse=True), all_fields)
    write_rows(out / "geometry_candidates.csv", geometry, all_fields)
    write_rows(out / "rt_semantic_candidates.csv", rt_semantic, all_fields)
    write_rows(out / "overall_semantic_candidates.csv", overall_semantic, all_fields)
    write_rows(out / "balanced_candidates.csv", balanced, all_fields)

    selected_ids: List[int] = []
    for group in (geometry, rt_semantic, overall_semantic, balanced):
        for row in group:
            frame_id = int(row["frame_id"])
            if frame_id not in selected_ids:
                selected_ids.append(frame_id)
    with (out / "selected_frames.txt").open("w") as handle:
        for frame_id in selected_ids:
            handle.write(f"{frame_id:06d}\n")

    report = [
        "# Qualitative candidate report",
        "",
        f"Shared evaluated frames: {len(shared)}",
        f"Eligible after safeguards: {len(eligible)}",
        "",
        "Safeguards require sufficient GT classes and limit occupancy IoU/Recall loss. "
        "Final selection still requires inspecting the same-view GT/S4C/DHS/RT renders.",
        "",
    ]
    report += markdown_table("DHS geometry candidates", geometry)
    report += markdown_table("RT semantic candidates", rt_semantic)
    report += markdown_table("Ours vs S4C semantic candidates", overall_semantic)
    report += markdown_table("Balanced candidates", balanced)
    (out / "candidate_report.md").write_text("\n".join(report))

    print(f"Shared frames: {len(shared)}")
    print(f"Eligible frames: {len(eligible)}")
    print(f"Selected IDs: {out / 'selected_frames.txt'}")
    print(f"Report: {out / 'candidate_report.md'}")


if __name__ == "__main__":
    main()
