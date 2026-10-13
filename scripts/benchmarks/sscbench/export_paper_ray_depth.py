#!/usr/bin/env python3
"""Export metric expected ray-termination depth for paper figures.

Run this script inside each method's own repository so that its checkpoint is
constructed with the matching code and Hydra configuration.  The output is a
raw H x W float32 NumPy array in metres; no visualization normalization is
applied here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from torch import nn

sys.path.append(".")
sys.path.extend([".", "../../../"])

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer
from sscbench_dataset import SSCBenchDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unnormalized expected ray-termination depth."
    )
    parser.add_argument("--ssc", required=True, type=Path, help="SSCBench/KITTI-360 root.")
    parser.add_argument("--vgt", required=True, type=Path, help="SSCBench voxel-label root.")
    parser.add_argument("--checkpoint", "-cp", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("s4c", "ours"))
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    parser.add_argument("--sequence", default=9, type=int)
    parser.add_argument("--depth-root", type=Path, help="DA3 v3 root; required for Ours.")
    parser.add_argument(
        "--prior-dir",
        type=Path,
        help="Optional flat directory containing the selected depth-prior PNGs.",
    )
    parser.add_argument(
        "--confidence-dir",
        type=Path,
        help="Optional flat directory containing the selected confidence PNGs.",
    )
    parser.add_argument("--depth-scale", default=256.0, type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    matches = sorted(path.glob("training*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one training*.pt in {path}, found {len(matches)}")
    return matches[0]


def find_png(folder: Path, frame_id: int) -> Path:
    for name in (f"{frame_id:010d}.png", f"{frame_id:06d}.png"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing frame {frame_id} in {folder}")


def load_prior_and_confidence(
    depth_root: Path | None,
    prior_dir: Path | None,
    confidence_dir: Path | None,
    sequence: str,
    frame_id: int,
    height: int,
    width: int,
    depth_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    if prior_dir is not None or confidence_dir is not None:
        if prior_dir is None or confidence_dir is None:
            raise ValueError("Use --prior-dir and --confidence-dir together")
        depth_path = find_png(prior_dir.expanduser().resolve(), frame_id)
        conf_path = find_png(confidence_dir.expanduser().resolve(), frame_id)
    else:
        if depth_root is None:
            raise ValueError(
                "Ours requires either --depth-root or both --prior-dir and --confidence-dir"
            )
        camera_root = depth_root.expanduser().resolve() / sequence / "image_00"
        depth_folders = (camera_root / "data_rect", camera_root / "data_rgb")
        depth_path = None
        for folder in depth_folders:
            try:
                depth_path = find_png(folder, frame_id)
                break
            except FileNotFoundError:
                continue
        if depth_path is None:
            raise FileNotFoundError(
                f"Missing frame {frame_id} in both "
                + " and ".join(str(folder) for folder in depth_folders)
            )
        conf_path = find_png(camera_root / "data_rect_conf", frame_id)

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    conf_raw = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or conf_raw is None:
        raise RuntimeError(f"OpenCV could not read {depth_path} or {conf_path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if conf_raw.ndim == 3:
        conf_raw = conf_raw[..., 0]

    depth = depth_raw.astype(np.float32) / depth_scale
    confidence = conf_raw.astype(np.float32) / 10000.0
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    confidence[~np.isfinite(confidence)] = 1.0
    confidence = np.clip(confidence, 0.0, 6.5535)

    depth_tensor = torch.from_numpy(depth)[None, None, None].to(DEVICE)
    confidence_tensor = torch.from_numpy(confidence)[None, None, None].to(DEVICE)
    return depth_tensor, confidence_tensor, depth, confidence


def make_model(checkpoint: Path):
    initialize(version_base=None, config_path="../../../configs", job_name="export_paper_ray_depth")
    config = compose(config_name="exp_kitti_360", overrides=[])
    checkpoint_data = torch.load(checkpoint, map_location=DEVICE)
    state_dict = checkpoint_data.get("model", checkpoint_data)

    # Match the SSCBench evaluator's architecture overrides exactly.
    with open_dict(config):
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "z"
        config["model_conf"]["mlp_coarse"]["n_blocks"] = 0
        config["model_conf"]["mlp_coarse"]["d_hidden"] = 64
        config["model_conf"]["encoder"]["d_out"] = 64
        config["model_conf"]["encoder"]["type"] = "monodepth2"
        config["model_conf"]["grid_learn_empty"] = False
        config["model_conf"]["sample_color"] = True
        config["model_conf"]["segmentation_mode"] = "panoptic_deeplab"

        input_weight = state_dict.get("renderer.net.mlp_coarse.lin_in.weight")
        if input_weight is not None:
            config["model_conf"]["use_rbf_encoding"] = input_weight.shape[1] != 103

    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True
    renderer.renderer.hard_alpha_cap = True

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    incompatible = Wrapper().load_state_dict(state_dict, strict=False)
    print(f"Checkpoint: {checkpoint}")
    print(f"Missing keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    if incompatible.missing_keys:
        print("First missing keys:", incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        print("First unexpected keys:", incompatible.unexpected_keys[:10])

    renderer.to(DEVICE).eval()
    return config, net, renderer


def encode(net, images: torch.Tensor, projs: torch.Tensor, poses: torch.Tensor) -> None:
    net.compute_grid_transforms(projs, poses)
    parameters = inspect.signature(net.encode).parameters
    if "data" in parameters:
        net.encode({}, images, projs, poses, ids_encoder=[0], ids_render=[0])
    else:
        net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])
    net.set_scale(0)


def render_depth(
    data: dict,
    config,
    net,
    renderer,
    method: str,
    sequence: str,
    frame_id: int,
    depth_root: Path | None,
    prior_dir: Path | None,
    confidence_dir: Path | None,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    images = torch.stack(data["imgs"], dim=0).unsqueeze(0).to(DEVICE).float()
    poses = torch.from_numpy(np.stack(data["poses"], axis=0)).unsqueeze(0).to(DEVICE).float()
    projs = torch.from_numpy(np.stack(data["projs"], axis=0)).unsqueeze(0).to(DEVICE).float()
    poses = torch.inverse(poses[:, :1]) @ poses
    _, _, _, height, width = images.shape

    encode(net, images, projs, poses)
    sampler = ImageRaySampler(
        config["model_conf"]["z_near"],
        config["model_conf"]["z_far"],
        height,
        width,
        norm_dir=True,
    )

    prior_tensor = confidence_tensor = None
    prior_array = confidence_array = None
    adaptive = bool(getattr(net, "adaptive_sampling", False))
    if method == "ours":
        if not adaptive:
            raise RuntimeError("--method ours was requested, but the constructed network has adaptive_sampling=False")
        prior_tensor, confidence_tensor, prior_array, confidence_array = load_prior_and_confidence(
            depth_root,
            prior_dir,
            confidence_dir,
            sequence,
            frame_id,
            height,
            width,
            depth_scale,
        )

    sample_parameters = inspect.signature(sampler.sample).parameters
    sample_kwargs = {}
    if "da3_depths" in sample_parameters:
        sample_kwargs["da3_depths"] = prior_tensor
    if "da3_confs" in sample_parameters:
        sample_kwargs["da3_confs"] = confidence_tensor
    sample_result = sampler.sample(images * 0.5 + 0.5, poses, projs, **sample_kwargs)
    all_rays = sample_result[0]
    all_rgb = sample_result[1]
    all_prior = sample_result[2] if len(sample_result) >= 3 else None
    all_confidence = sample_result[3] if len(sample_result) >= 4 else None

    adaptive_config = None
    if adaptive:
        adaptive_config = {
            "n_surface": getattr(net, "n_surface_samples", 32),
            "n_global": getattr(net, "n_global_samples", 32),
            "absrel_prior": getattr(net, "absrel_prior", 0.12),
            "min_thickness": getattr(net, "min_thickness", 0.5),
            "conf_epsilon": getattr(net, "conf_epsilon", 0.1),
            "conf_threshold": getattr(net, "conf_threshold", 0.2),
        }
        if all_prior is None:
            raise RuntimeError("Adaptive model did not receive per-ray depth priors")

    render_parameters = inspect.signature(renderer.forward).parameters
    render_kwargs = {"want_weights": True, "want_alphas": True}
    if "da3_depths" in render_parameters:
        render_kwargs["da3_depths"] = all_prior
    if "adaptive_sampling_config" in render_parameters:
        render_kwargs["adaptive_sampling_config"] = adaptive_config
    if "conf_values" in render_parameters:
        render_kwargs["conf_values"] = all_confidence

    rendered = renderer(all_rays, **render_kwargs)
    if "fine" not in rendered:
        rendered["fine"] = dict(rendered["coarse"])
    rendered["rgb_gt"] = all_rgb
    rendered = sampler.reconstruct(rendered)
    depth = rendered["fine"]["depth"][0, 0].detach().cpu().numpy().astype(np.float32)
    return depth, prior_array, confidence_array


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = SSCBenchDataset(
        data_path=str(args.ssc.expanduser().resolve()),
        voxel_gt_path=str(args.vgt.expanduser().resolve()),
        sequences=(args.sequence,),
        target_image_size=(192, 640),
        return_stereo=False,
        frame_count=1,
        color_aug=False,
        load_all=False,
    )
    index_by_frame = {
        frame_id: index
        for index, (_, frame_id, is_right) in enumerate(dataset._datapoints)
        if not is_right
    }
    missing = sorted(set(args.frames) - set(index_by_frame))
    if missing:
        raise RuntimeError(f"Requested frames are absent from the SSCBench split: {missing}")

    config, net, renderer = make_model(checkpoint)
    sequence = f"2013_05_28_drive_00{args.sequence:02d}_sync"

    with torch.inference_mode():
        for frame_id in args.frames:
            # Reset the random stream per frame so a frame has the same
            # stratified samples regardless of list order.
            random.seed(args.seed + frame_id)
            np.random.seed(args.seed + frame_id)
            torch.manual_seed(args.seed + frame_id)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + frame_id)

            dataset_index = index_by_frame[frame_id]
            data = dataset[dataset_index]
            depth, prior, confidence = render_depth(
                data,
                config,
                net,
                renderer,
                args.method,
                sequence,
                frame_id,
                args.depth_root,
                args.prior_dir,
                args.confidence_dir,
                args.depth_scale,
            )
            prefix = args.out / f"{frame_id:06d}_{args.method}"
            np.save(prefix.with_name(prefix.name + "_render_depth.npy"), depth)
            if prior is not None:
                np.save(args.out / f"{frame_id:06d}_prior_depth.npy", prior.astype(np.float32))
            if confidence is not None:
                np.save(args.out / f"{frame_id:06d}_confidence.npy", confidence.astype(np.float32))

            # Sparse projected LiDAR is an optional measured reference for the
            # depth figure.  Its absence must not block model-depth export.
            try:
                sequence_name, _, is_right = dataset._datapoints[dataset_index]
                lidar_depth = dataset.load_depth(sequence_name, frame_id, is_right)[0]
                np.save(
                    args.out / f"{frame_id:06d}_lidar_depth.npy",
                    lidar_depth.astype(np.float32),
                )
            except FileNotFoundError as error:
                print(f"Warning: sparse LiDAR depth unavailable for {frame_id:06d}: {error}")

            valid = np.isfinite(depth) & (depth > 0)
            metadata = {
                "frame": frame_id,
                "method": args.method,
                "checkpoint": str(checkpoint),
                "shape": list(depth.shape),
                "valid_fraction": float(valid.mean()),
                "min_m": float(depth[valid].min()) if valid.any() else None,
                "median_m": float(np.median(depth[valid])) if valid.any() else None,
                "max_m": float(depth[valid].max()) if valid.any() else None,
                "seed": args.seed + frame_id,
            }
            prefix.with_name(prefix.name + "_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
