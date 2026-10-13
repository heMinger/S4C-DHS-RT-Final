import argparse
import csv
import inspect
import sys
from datetime import datetime

from omegaconf import open_dict

import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.extend([".", "../../../"])

from scripts.benchmarks.sscbench.generate_ply_sequence import get_cam_k
from scripts.benchmarks.sscbench.point_utils import read_calib, generate_point_grid, get_fov_mask

from scripts.voxel.gen_voxelgrid_npy import save_as_voxel_ply

import logging

from pathlib import Path
import subprocess
import yaml

import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from hydra import compose, initialize

import matplotlib.pyplot as plt

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer

from sscbench_dataset import SSCBenchDataset
from pathlib import Path

RELOAD_DATASET = True
DATASET_LENGTH = 2000  # must cover every id listed in PLY_IDS
FULL_EVAL = False  # 是否对完整数据集进行评估
SAMPLE_EVERY = None
SAMPLE_OFFSET = 2
# SAMPLE_RANGE = list(range(1000, 1600))
SAMPLE_RANGE = None

SIZE = 51.2 # Can be: 51.2, 25.6, 12.8
SIZES = (12.8, 25.6, 51.2)
# VOXEL_SIZE = 0.2 # Needs: 0.2 % VOXEL_SIZE == 0
VOXEL_SIZE = 0.1

# USE_ADDITIONAL_INVALIDS = True  # True = 排除遮挡区域，IoU 更高
USE_ADDITIONAL_INVALIDS = False

TEST_ALPHA_CUTOFFS = False
SEARCH_VALUES = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7]

SIGMA_CUTOFF = 0.25  # default; can be overridden via CLI

USE_ALPHA_WEIGHTING = True
# USE_GROW = True  # True = 膨胀预测，填补空洞，IoU 更高
USE_GROW = False

CREATE_SIGMA_TRADEOFF_PLOT = False  # headless PLY generation must not open matplotlib
SIGMA_VALUES = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

PLOT_ALL_IMAGES = False

# GENERATE_PLY_FILES = False  # True 需要同时设置 USE_GROW=False, USE_ADDITIONAL_INVALIDS=False
GENERATE_PLY_FILES = True
PLY_ONLY_FOV = True
PLY_IDS = [200]  # 先用一帧诊断；定稿时替换为同一列表
PLY_PATH = Path("out/voxel_outputs_vis/model_name")  # 改成 s4c / dhs / dhs_rt
PLY_SIZES = [51.2]  # 主定性图只保留一个空间范围

GENERATE_STATISTICS = False

if GENERATE_PLY_FILES:
    # assert (not USE_GROW) and (not USE_ADDITIONAL_INVALIDS) and VOXEL_SIZE == 0.1
    assert VOXEL_SIZE == 0.1


# Setup of CUDA device and logging

os.system("nvidia-smi")

device = f'cuda:0'

# DO NOT TOUCH OR YOU WILL BREAK RUNS (should be None)
gpu_id = None

if gpu_id is not None:
    print("GPU ID: " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)


def safe_divide(numerator, denominator):
    """Scalar division that returns NaN when a per-frame metric is undefined."""
    return float(numerator / denominator) if denominator != 0 else float("nan")


def safe_divide_array(numerator, denominator):
    """Element-wise division without RuntimeWarning for absent semantic classes."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def remove_small_components(occupied, classes, min_size=0):
    """Remove isolated predicted components for visualization only.

    This deliberately does not dilate the prediction.  Use min_size=0 for
    the raw result; if needed, use the same positive value for every method.
    """
    if min_size <= 0:
        return occupied, classes
    from scipy import ndimage
    labels, n = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(occupied, labels, index=range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = np.asarray(sizes) >= min_size
    keep_mask = keep[labels]
    return occupied & keep_mask, np.where(keep_mask, classes, 0)


def main():
    parser = argparse.ArgumentParser("SSCBenchmark Output generation")
    parser.add_argument("--sscbench_data_root", "-ssc", type=str)
    parser.add_argument("--voxel_gt_path", "-vgt", type=str)
    parser.add_argument("--resolution", "-r", default=(192, 640))
    parser.add_argument("--checkpoint", "-cp", type=str, required=True)
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument(
        "--sigma_cutoff",
        type=float,
        default=None,
        help="Override SIGMA_CUTOFF used to threshold occupancy (default: 0.25).",
    )
    parser.add_argument(
        "--sweep_sigma",
        type=str,
        default=None,
        help="Comma-separated sigma cutoffs to sweep for occupancy PR/IoU (e.g. '0.1,0.2,0.25,0.3'). "
             "If omitted, no sweep is performed.",
    )
    parser.add_argument(
        "--per_frame_csv",
        type=str,
        default=None,
        help="Write official per-frame occupancy and semantic metrics to this CSV.",
    )
    parser.add_argument(
        "--no_ply",
        action="store_true",
        help="Disable PLY generation and its PLY_IDS filter for metric-only runs.",
    )
    parser.add_argument(
        "--frame_ids",
        type=str,
        default=None,
        help="Optional comma-separated frame IDs. Omit to evaluate the selected dataset split.",
    )
    parser.add_argument(
        "--frame_ids_file",
        type=str,
        default=None,
        help="Optional text file containing one frame ID per line.",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=9,
        help="KITTI-360 sequence number (default: 9, the SSCBench test sequence).",
    )
    parser.add_argument(
        "--prediction_only",
        action="store_true",
        help="Run inference without voxel ground truth. Use this for arbitrary RGB frames such as sequence 06 frame 146.",
    )
    parser.add_argument(
        "--ply_output",
        type=str,
        default=None,
        help="Root directory for generated PLY files. Defaults to the PLY_PATH constant.",
    )

    args = parser.parse_args()

    sscbench_data_root = args.sscbench_data_root
    voxel_gt_path = args.voxel_gt_path
    resolution = args.resolution
    cp_path = args.checkpoint
    full_evaluation = args.full
    sigma_cutoff = args.sigma_cutoff
    sweep_sigma = args.sweep_sigma
    per_frame_csv = Path(args.per_frame_csv).expanduser() if args.per_frame_csv else None
    generate_ply_files = GENERATE_PLY_FILES and not args.no_ply
    sequence_number = int(args.sequence)
    prediction_only = bool(args.prediction_only)
    ply_path = Path(args.ply_output).expanduser() if args.ply_output else PLY_PATH
    if args.frame_ids and args.frame_ids_file:
        raise ValueError("Use either --frame_ids or --frame_ids_file, not both")
    requested_frame_ids = None
    if args.frame_ids_file:
        frame_ids_path = Path(args.frame_ids_file).expanduser()
        requested_frame_ids = {
            int(line.strip())
            for line in frame_ids_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    elif args.frame_ids:
        requested_frame_ids = {
            int(value.strip()) for value in args.frame_ids.split(",") if value.strip()
        }
    if requested_frame_ids is not None and not requested_frame_ids:
        raise ValueError("The requested frame-ID selection is empty")
    if prediction_only and requested_frame_ids is None:
        raise ValueError("--prediction_only requires --frame_ids or --frame_ids_file to avoid unintentionally processing an entire sequence")
    if prediction_only and not generate_ply_files:
        raise ValueError("--prediction_only is intended to export predictions and cannot be combined with --no_ply")
    if prediction_only and per_frame_csv is not None:
        raise ValueError("--per_frame_csv requires voxel ground truth and cannot be used with --prediction_only")

    if generate_ply_files:
        for size in PLY_SIZES:
            (ply_path / str(int(size))).mkdir(parents=True, exist_ok=True)

    global SIGMA_CUTOFF
    if sigma_cutoff is not None:
        SIGMA_CUTOFF = float(sigma_cutoff)
    logging.info(f"Using a sigma cutoff of {SIGMA_CUTOFF}")
    logging.info("Setting up dataset")

    global SIGMA_VALUES
    if sweep_sigma is not None:
        try:
            SIGMA_VALUES = [float(x.strip()) for x in sweep_sigma.split(",") if x.strip() != ""]
        except Exception as e:
            raise ValueError(f"Failed to parse --sweep_sigma='{sweep_sigma}'. Expected comma-separated floats.") from e
        if len(SIGMA_VALUES) == 0:
            raise ValueError("Parsed --sweep_sigma is empty. Provide at least one value, e.g. --sweep_sigma '0.25'.")

    with open("scripts/benchmarks/sscbench/label_maps.yaml", "r") as f:
        label_maps = yaml.safe_load(f)

    # Keep caches split-specific so a sequence-09 test cache can never be
    # mistaken for an arbitrary-frame sequence-06 dataset.
    dataset_cache_path = Path(
        f"dataset_seq{sequence_number:02d}_{'all' if prediction_only else 'gt'}.pkl"
    )
    if dataset_cache_path.is_file() and not RELOAD_DATASET:
        logging.info("Loading dataset from %s", dataset_cache_path)
        with dataset_cache_path.open("rb") as f:
            dataset = pickle.load(f)
    else:
        logging.info("Generating sequence %02d dataset (prediction_only=%s)", sequence_number, prediction_only)
        dataset = SSCBenchDataset(
            data_path=sscbench_data_root,
            voxel_gt_path=voxel_gt_path,
            sequences=(sequence_number,),
            target_image_size=resolution,
            return_stereo=False,
            frame_count=1,
            color_aug=False,
            load_all=prediction_only,
        )
        with dataset_cache_path.open("wb") as f:
            pickle.dump(dataset, f)

    if requested_frame_ids is not None:
        dataset._datapoints = [
            datapoint for datapoint in dataset._datapoints
            if datapoint[1] in requested_frame_ids
        ]
        dataset.length = len(dataset._datapoints)
        if dataset.length == 0:
            raise RuntimeError(
                f"None of the requested frame IDs {sorted(requested_frame_ids)} "
                f"were found in sequence {sequence_number:02d}. Check --sscbench_data_root "
                "and whether image filenames are present in image_00/data_rect."
            )
    elif not full_evaluation:
        dataset.length = min(int(DATASET_LENGTH), dataset.length)

    logging.info("Setting up the model...")

    config_path = "exp_kitti_360"


    cp_path = Path(cp_path)
    # cp_path = next(cp_path.glob("training*.pt"))

    if cp_path.is_file():
        # 如果你直接指定了文件（如 .pt），就直接使用它
        final_cp_path = cp_path
    elif cp_path.is_dir():
        # 如果你指定的是文件夹，则保持原有的搜索逻辑
        try:
            final_cp_path = next(cp_path.glob("training*.pt"))
        except StopIteration:
            raise FileNotFoundError(f"在文件夹 {cp_path} 中找不到 training*.pt 文件")
    else:
        raise FileNotFoundError(f"路径不存在: {cp_path}")

    cp_path = final_cp_path

    initialize(version_base=None, config_path="../../../configs", job_name="gen_sscbench_outputs")
    config = compose(config_name=config_path, overrides=[])

    logging.info('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    with open_dict(config):
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "z"
        # config["model_conf"]["z_near"] = 8
        config["model_conf"]["mlp_coarse"]["n_blocks"] = 0
        config["model_conf"]["mlp_coarse"]["d_hidden"] = 64
        config["model_conf"]["encoder"]["d_out"] = 64
        config["model_conf"]["encoder"]["type"] = "monodepth2"
        config["model_conf"]["grid_learn_empty"] = False
        config["model_conf"]["sample_color"] = True

        # stuff for segmentation
        config["model_conf"]["segmentation_mode"] = 'panoptic_deeplab'
        
        # 自动检测 checkpoint 是否使用 RBF encoding
        # 通过检查 mlp_coarse.lin_in.weight 的维度来判断
        state_dict = cp["model"] if "model" in cp else cp
        if "renderer.net.mlp_coarse.lin_in.weight" in state_dict:
            mlp_in_dim = state_dict["renderer.net.mlp_coarse.lin_in.weight"].shape[1]
            # d_in=103 表示没有 RBF (features=64 + xyz_code=39)
            # d_in=107 表示有 RBF (features=64 + xyz_code=39 + rbf_code=4)
            if mlp_in_dim == 103:
                logging.info(f"[Auto-detect] Checkpoint uses d_in={mlp_in_dim}, disabling RBF encoding")
                config["model_conf"]["use_rbf_encoding"] = False
            else:
                logging.info(f"[Auto-detect] Checkpoint uses d_in={mlp_in_dim}, keeping RBF encoding")
                config["model_conf"]["use_rbf_encoding"] = True

    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    # 兼容两种权重格式：training_checkpoint (包含"model"键) 和 best_model (直接是state_dict)
    if "model" in cp:
        state_dict = cp["model"]
    else:
        state_dict = cp
    _wrapper.load_state_dict(state_dict, strict=False)
    renderer.to(device)
    renderer.eval()

    logging.info("Loading the Lidar to Camera matrices...")

    calib = read_calib()
    T_velo_2_cam = calib["Tr"]

    logging.info("Generating the point cloud...")

    pts, _ = generate_point_grid(vox_origin=np.array([0, -25.6, -2]),
                              scene_size=(51.2, 51.2, 6.4),
                              voxel_size=VOXEL_SIZE,
                              cam_E=T_velo_2_cam,
                              cam_k=get_cam_k())

    fov_mask = get_fov_mask()

    pts = torch.tensor(pts).to(device).reshape(1, -1, 3).float()
    fov_mask = fov_mask.reshape(256, 256, 32)

    logging.info("Setting up folders...")

    downsample_factor = int(0.2 // VOXEL_SIZE)

    results = {}
    for size in SIZES:
        results[size] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_seg": np.zeros(15),
            "fp_seg": np.zeros(15),
            "tn_seg": np.zeros(15),
            "fn_seg": np.zeros(15),
        }

    # for the sigma tradeoff plots
    trade_off_values = np.zeros([len(SIGMA_VALUES), 4])

    cutoff_results = {i: {sv: {"tp":0, "fp": 0, "tn": 0, "fn": 0} for sv in SEARCH_VALUES} for i in range(1, 16)}

    pbar = tqdm(range(len(dataset)))
    per_frame_rows = []
    processed_frame_ids = set()

    images = {"ids": [], "images": []}

    ids = [125, 280, 960, 1000, 1150, 1325, 2300, 3175, 3750, 4300, 5155, 5475, 5750, 6475, 6525, 6670, 6775, 7500, 7860, 8000, 8350, 9000, 9350, 10975]

    ids = [60, 250, 455, 690, 835, 2235, 2385, 2495, 3385, 4235, 4360, 4550, 4875, 5550, 6035, 7010, 7110, 8575, 9010, 9410, 11260, 11460, 11885]

    # for our statistics
    tframeIds = []
    tinval = []
    ttp = []
    tfp = []
    ttn = []
    tfn = []

    # plot_image_at_frame_id(dataset, 952)
    for i in pbar:
        if SAMPLE_EVERY:
            if (i - SAMPLE_OFFSET) % SAMPLE_EVERY != 0:
                continue

        sequence, id, is_right = dataset._datapoints[i]

        if requested_frame_ids is not None and id not in requested_frame_ids:
            continue

        if SAMPLE_RANGE:
            if id not in SAMPLE_RANGE:
                continue

        # An explicit CLI frame selection takes precedence over the legacy
        # PLY_IDS constant near the top of this file.
        if (generate_ply_files and requested_frame_ids is None
                and PLY_IDS is not None and id not in PLY_IDS):
            continue

        if GENERATE_STATISTICS:
            tframeIds.append(id)

        data = dataset[i]

        # downsample the sigmas
        sigmas, segs = downsample_and_predict(data, net, pts, downsample_factor)

        # convert both to the right format
        segs = convert_voxels(segs, label_maps["cityscapes_to_label"])
        target = None
        if not prediction_only:
            target = convert_voxels(data["voxel_gt"][0].astype(int), label_maps["sscbench_to_label"])


        if PLOT_ALL_IMAGES:
            images["ids"].append(id)
            images["images"].append(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))

            if len(images["ids"]) == 6:
                plot_images(images)
                images = {"images": [], "ids": []}

        # print(f"Image_Id: {id}")
        #
        # plt.imshow(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))
        # plt.show()
        #
        # out_dict = {"sigmas": sigmas, "segs": segs.copy(), "gt": target, "fov_mask": fov_mask}
        #
        # with open(f'plots10_40/{id:06d}.pkl', 'wb') as f:
        #     pickle.dump(out_dict, f)

        if generate_ply_files:
            _segs = segs.copy()
            _segs[sigmas < SIGMA_CUTOFF] = 0

            if prediction_only:
                # No voxel GT exists for an arbitrary RGB frame. Restrict the
                # visualization by camera FOV only, never by a fabricated GT.
                mask = fov_mask if PLY_ONLY_FOV else np.ones_like(_segs, dtype=bool)
                _target = None
            else:
                _target = target.copy()
                mask = target != 255
                if PLY_ONLY_FOV:
                    mask = mask & fov_mask

            _segs[~mask] = 0
            if _target is not None:
                _target[~mask] = 0
            # Keep the main result faithful to the prediction.  If isolated
            # speckles are unacceptable, set the same MIN_COMPONENT_SIZE in
            # all three repositories; never use grow for only one method.
            MIN_COMPONENT_SIZE = 0
            pred_occ = _segs > 0
            pred_occ, _segs = remove_small_components(
                pred_occ, _segs, min_size=MIN_COMPONENT_SIZE
            )
            is_occupied_seg = torch.tensor(_segs > 0)

            for size in PLY_SIZES:
                num_voxels = int(size // 0.2)
                save_as_voxel_ply(ply_path / str(int(size)) / f"{id:06d}.ply",
                                  is_occupied_seg[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:],
                                  classes=torch.tensor(_segs[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:]))
                if _target is not None:
                    is_occupied_gt = torch.tensor(_target > 0)
                    save_as_voxel_ply(ply_path / str(int(size)) / f"{id:06d}_gt.ply",
                                      is_occupied_gt[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:],
                                      classes=torch.tensor(_target[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:]))

        processed_frame_ids.add(int(id))
        if prediction_only:
            logging.info("Prediction-only PLY saved for sequence %02d frame %06d under %s", sequence_number, id, ply_path)
            continue

        if USE_ADDITIONAL_INVALIDS:
            invalids = identify_additional_invalids(target)
            # logging.info(np.mean(invalids))
            target[invalids == 1] = 255

            if GENERATE_STATISTICS:
                tinval.append(np.mean(invalids))

        # test and summarize different alpha cutoffs
        if TEST_ALPHA_CUTOFFS:
            for i in range(1, 16):
                for search_value in SEARCH_VALUES:
                    _tmp = segs.copy()
                    _tmp[np.logical_and(segs == i, sigmas < search_value)] = 0
                    _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                        y_pred=_tmp, y_true=target, fov_mask=fov_mask, labels=label_maps["labels"])
                    cutoff_results[i][search_value]["tp"] += _tp_seg[i-1]
                    cutoff_results[i][search_value]["fp"] += _fp_seg[i-1]
                    cutoff_results[i][search_value]["tn"] += _tn_seg[i-1]
                    cutoff_results[i][search_value]["fn"] += _fn_seg[i-1]

        if CREATE_SIGMA_TRADEOFF_PLOT:
            for i, val in enumerate(SIGMA_VALUES):
                _tmp = segs.copy()
                _tmp[sigmas < val] = 0
                _tp, _fp, _tn, _fn = compute_occupancy_numbers(y_pred=_tmp, y_true=target, fov_mask=fov_mask)
                trade_off_values[i] += np.array([_tp, _fp, _tn, _fn])

        segs[sigmas < SIGMA_CUTOFF] = 0

        frame_row = {
            "frame_id": int(id),
            "sequence": str(sequence),
            "is_right": int(bool(is_right)),
            "sigma_cutoff": float(SIGMA_CUTOFF),
        }

        for size in SIZES:
            num_voxels = int(size // 0.2)

            # resize to right scene size
            _segs = segs[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _target = target[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _fov_mask = fov_mask[:num_voxels, (128 - num_voxels // 2):(128 + num_voxels // 2), :]

            _tp, _fp, _tn, _fn = compute_occupancy_numbers(y_pred=_segs, y_true=_target, fov_mask=_fov_mask)
            _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                y_pred=_segs, y_true=_target, fov_mask=_fov_mask, labels=label_maps["labels"])

            size_key = str(size).replace(".", "p")
            frame_precision = safe_divide(_tp, _tp + _fp)
            frame_recall = safe_divide(_tp, _tp + _fn)
            frame_iou = safe_divide(_tp, _tp + _fp + _fn)
            frame_class_iou = safe_divide_array(
                _tp_seg, _tp_seg + _fp_seg + _fn_seg
            )
            frame_gt_present = (_tp_seg + _fn_seg) > 0
            frame_valid_miou = (
                float(np.mean(frame_class_iou[frame_gt_present]))
                if np.any(frame_gt_present) else float("nan")
            )
            frame_row.update({
                f"tp_{size_key}": int(_tp),
                f"fp_{size_key}": int(_fp),
                f"fn_{size_key}": int(_fn),
                f"gt_occ_{size_key}": int(_tp + _fn),
                f"iou_{size_key}_pct": 100.0 * frame_iou,
                f"precision_{size_key}_pct": 100.0 * frame_precision,
                f"recall_{size_key}_pct": 100.0 * frame_recall,
                f"valid_miou_{size_key}_pct": 100.0 * frame_valid_miou,
                f"valid_classes_{size_key}": int(np.sum(frame_gt_present)),
            })
            for class_index, class_name in label_maps["labels"].items():
                if int(class_index) == 0:
                    continue
                normalized_name = str(class_name).replace("-", "_")
                frame_row[f"iou_{size_key}_{normalized_name}_pct"] = (
                    100.0 * frame_class_iou[int(class_index) - 1]
                    if frame_gt_present[int(class_index) - 1] else float("nan")
                )

            if size == 51.2 and GENERATE_STATISTICS:
                ttp += [_tp]
                tfp += [_fp]
                ttn += [_fn]
                tfn += [_fn]

            results[size]["tp"] += _tp
            results[size]["fp"] += _fp
            results[size]["tn"] += _tn
            results[size]["fn"] += _fn

            results[size]["tp_seg"] += _tp_seg
            results[size]["fp_seg"] += _fp_seg
            results[size]["tn_seg"] += _tn_seg
            results[size]["fn_seg"] += _fn_seg

            recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
            precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
            iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        pbar.set_postfix_str(f"IoU: {iou*100:.2f} Prec: {precision*100:.2f} Rec: {recall*100:.2f}")
        per_frame_rows.append(frame_row)

    if requested_frame_ids is not None:
        missing_ids = sorted(requested_frame_ids - processed_frame_ids)
        if missing_ids:
            logging.warning("Requested frame IDs not found in the evaluated split: %s", missing_ids)

    if prediction_only:
        if not processed_frame_ids:
            raise RuntimeError("Prediction-only run completed without processing any frames")
        logging.info(
            "Prediction-only export complete: sequence %02d, frames=%s, output=%s",
            sequence_number,
            sorted(processed_frame_ids),
            ply_path,
        )
        return

    if per_frame_csv is not None:
        if not per_frame_rows:
            raise RuntimeError("No frames were evaluated; per-frame CSV would be empty")
        per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(per_frame_rows[0].keys())
        with per_frame_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_frame_rows)
        logging.info("Per-frame metrics saved to: %s", per_frame_csv)

    results_table = np.zeros((19, 3), dtype=np.float32)

    # Here we compute all the metrics
    for size_i, size in enumerate(SIZES):
        recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
        precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
        iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        results_table[0, size_i] = iou
        results_table[1, size_i] = precision
        results_table[2, size_i] = recall

        logging.info(f"#" * 50)
        logging.info(f"Results for size {size}. ")
        logging.info(f"#" * 50)

        logging.info("Occupancy metrics")
        logging.info(f"Recall: {recall*100:.2f}%")
        logging.info(f"Precision: {precision*100:.2f}%")
        logging.info(f"IoU: {iou*100:.2f}")

        recall_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fn_seg"])
        precision_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"])
        iou_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"])

        weights = label_maps["weights"]
        weights_val = np.array(list(weights.values()))
        weighted_mean_iou = np.sum(weights_val * np.nan_to_num(iou_seg)) / np.sum(weights_val)

        mean_iou = np.mean(np.nan_to_num(iou_seg))
        # "Valid-class" mIoU: only classes that appear in GT for this evaluation set (avoid penalizing missing rare classes in small subsets)
        gt_present = (results[size]["tp_seg"] + results[size]["fn_seg"]) > 0
        valid_mean_iou = np.mean(np.nan_to_num(iou_seg)[gt_present]) if np.any(gt_present) else float("nan")

        results_table[3, size_i] = mean_iou
        results_table[4:, size_i] = iou_seg

        logging.info("Occupancy metrics segmentation")
        for i in range(15):
            logging.info(f"{label_maps['labels'][i+1]}; IoU: {iou_seg[i]*100:.2f}; Precision: {precision_seg[i]*100:.2f}%; Recall: {recall_seg[i]*100:.2f}%")

    logging.info(f"Results table for copying.")

    results_table_str = ""
    for i in range(19):
        results_table_str += f"{results_table[i, 0]*100:.2f}\t{results_table[i, 1]*100:.2f}\t{results_table[i, 2]*100:.2f}\n"
    print(results_table_str)

    logging.info(f"Mean IoU: {mean_iou*100:.2f}")
    logging.info(f"Valid-class Mean IoU: {valid_mean_iou*100:.2f}" if not np.isnan(valid_mean_iou) else "Valid-class Mean IoU: nan")
    logging.info(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")

    if sweep_sigma is not None and CREATE_SIGMA_TRADEOFF_PLOT:
        # Print all sweep points (what you typically want to see in tmux logs)
        logging.info("[Sigma Sweep] Occupancy trade-off over provided SIGMA_VALUES:")
        logging.info("  sigma_cutoff\tIoU(%)\tPrec(%)\tRec(%)\tTP\tFP\tFN")
        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, _, fn = trade_off_values[i]
            pres = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            _iou = tp / (tp + fp + fn + 1e-9)
            logging.info(
                f"  {val}\t{_iou*100:.2f}\t{pres*100:.2f}\t{rec*100:.2f}\t{int(tp)}\t{int(fp)}\t{int(fn)}"
            )

        best = None  # (iou, sigma, precision, recall)
        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, _, fn = trade_off_values[i]
            pres = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            _iou = tp / (tp + fp + fn + 1e-9)
            cand = (_iou, val, pres, rec)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            biou, bsigma, bpres, brec = best
            logging.info(
                f"[Sigma Sweep] Best occupancy IoU={biou*100:.2f} at SIGMA_CUTOFF={bsigma} "
                f"(Prec={bpres*100:.2f} Rec={brec*100:.2f})"
            )

    # 自动保存评估结果到与权重一致的目录
    save_evaluation_results(
        cp_path=cp_path,
        results=results,
        results_table=results_table,
        mean_iou=mean_iou,
        valid_mean_iou=valid_mean_iou,
        weighted_mean_iou=weighted_mean_iou,
        label_maps=label_maps,
        dataset_length=len(dataset),
        sizes=SIZES
    )

    if TEST_ALPHA_CUTOFFS:
        cutoff_metrics = \
            {i: {sv: {"precision": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fp"])),
                       "recall": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fn"])),
                       "IoU": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fn"] + cutoff_results[i][sv]["fp"]))}
                      for sv in SEARCH_VALUES} for i in range(1, 16)}

        best_values = {i: SEARCH_VALUES[torch.argmax(torch.tensor([cutoff_metrics[i][sv]["IoU"] for sv in SEARCH_VALUES]))] for i in range(1, 16)}

        print(best_values)

    if CREATE_SIGMA_TRADEOFF_PLOT:
        plt.figure(figsize=(10, 8))
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.xlim([10, 70])
        # plt.ylim([0, 100])

        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, tn, fn = trade_off_values[i]
            pres = 100*tp / (tp + fp)
            recall = 100*tp/ (tp + fn)
            plt.scatter(pres, recall)
            plt.annotate(f"Sigma: {val}; IoU: {100*tp / (tp + fp + fn):.2f}", (pres, recall))

        identifier = os.path.basename(cp_path)
        if FULL_EVAL:
            path = f"figures/inv{str(USE_ADDITIONAL_INVALIDS)}_{VOXEL_SIZE:.1f}_mp{str(USE_GROW)}_{identifier}.png"
        else:
            path = f"figures/inv{str(USE_ADDITIONAL_INVALIDS)}_{DATASET_LENGTH}_{VOXEL_SIZE:.1f}_mp{str(USE_GROW)}_{identifier}.png"

        if os.path.isfile(path):
            os.remove(path)
        plt.savefig(path)

        plt.show()

    if GENERATE_STATISTICS:
        statistics_raw = {"frameId": tframeIds, "TP": ttp, "FP": tfp, "TN": ttn, "FN": tfn, "invalids": tinval}
        with open("stats.pkl", "wb") as f:
            pickle.dump(statistics_raw, f)
        logging.info("Saved the statistics for further analysis.")


def downsample_and_predict(data, net, pts, factor):
    pts = pts.reshape(256*factor, 256*factor, 32*factor, 3)

    sigmas = torch.zeros(256, 256, 32).numpy()
    segs = torch.zeros(256, 256, 32).numpy()

    chunk_size_x = chunk_size_y = 256
    chunk_size_z = 32

    n_chunks_x = int(256*factor / chunk_size_x)
    n_chunks_y = int(256*factor / chunk_size_y)
    n_chunks_z = int(32*factor / chunk_size_z)


    b_x = chunk_size_x // factor # size of the mini blocks
    b_y = chunk_size_y // factor
    b_z = chunk_size_z // factor


    for i in range(n_chunks_x):
        for j in range(n_chunks_y):
            for k in range(n_chunks_z):
                pts_block = pts[i * chunk_size_x:(i + 1) * chunk_size_x, j * chunk_size_y:(j + 1) * chunk_size_y, k * chunk_size_z:(k + 1) * chunk_size_z]
                sigmas_block, segs_block = predict_grid(data, net, pts_block)
                sigmas_block = sigmas_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z)
                segs_block = segs_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z, 19)

                if USE_ALPHA_WEIGHTING:
                    alphas = 1 - torch.exp(- VOXEL_SIZE * sigmas_block)
                    segs_block = (alphas.unsqueeze(-1) * segs_block).unsqueeze(0)
                else:
                    segs_block = (sigmas_block.unsqueeze(-1) * segs_block).unsqueeze(0)

                segs_pool_list = [F.avg_pool3d(segs_block[..., i], kernel_size=factor, stride=factor, padding=0) for i in
                                  range(segs_block.shape[-1])]
                segs_pool = torch.stack(segs_pool_list, dim=-1).unsqueeze(0)
                segs_pool = torch.argmax(segs_pool, dim=-1).detach().cpu().numpy()

                # pool the observations
                sigmas_block = F.max_pool3d(sigmas_block.unsqueeze(0), kernel_size=factor, stride=factor, padding=0).squeeze(0).detach().cpu().numpy()

                sigmas[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = sigmas_block
                segs[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = segs_pool

                torch.cuda.empty_cache()

    if USE_GROW:
        sigmas = F.max_pool3d(torch.tensor(sigmas).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).numpy()

    return sigmas, segs

def use_custom_maxpool(_sigmas):
    sigmas = torch.zeros(258, 258, 34)
    sigmas[1:257, 1:257, 1:33] = torch.tensor(_sigmas)
    sigmas_pooled = torch.zeros(256, 256, 32)

    for i in range(256):
        for j in range(256):
            for k in range(32):
                sigmas_pooled[i, j, k] = max(sigmas[i+1, j+1, k+1],
                                             sigmas[i, j+1, k+1], sigmas[i+1, j, k+1],sigmas[i+1, j+1, k],
                                             sigmas[i+2, j+1, k+1], sigmas[i+1, j+2, k+1],sigmas[i+1, j+1, k+2])
    return sigmas_pooled

def plot_images(images_dict):
    """The images dict should include six images and six corresponding ids"""
    images = images_dict["images"]
    ids = images_dict["ids"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 6))

    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        axes[i].set_title(f"FrameId: {ids[i]}")

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

def plot_image_at_frame_id(dataset, frame_id):

    for i in range(len(dataset)):
        sequence, id, is_right = dataset._datapoints[i]
        if id == frame_id:
            data = dataset[i]
            plt.figure(figsize=(10, 4))
            plt.imshow(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))
            plt.gca().set_axis_off()
            plt.show()
            return



def identify_additional_invalids(target):
    # Note: The Numpy implementation is a bit faster (about 0.1 seconds per iteration)

    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    # _t = torch.cat([torch.zeros([256, 256, 1], device=device, dtype=torch.int32), torch.tensor(target, dtype=torch.int32).to(device)], dim=2)
    # invalids = torch.cumsum((_t != 255) & (_t != 0), axis=2)[:,:, :32] == 0
    # height cut-off (z > 6 ==> no invalid)
    invalids[: , :, 7:] = 0
    # only empty voxels matter
    invalids[target != 0] = 0

    # return invalids.cpu().numpy()
    return invalids

PSEUDO_RAY_K = 64

def predict_grid(data_batch, net, points):
    images = torch.stack(data_batch["imgs"], dim=0).unsqueeze(0).to(device).float()
    poses = torch.tensor(np.stack(data_batch["poses"], 0)).unsqueeze(0).to(device).float()
    projs = torch.tensor(np.stack(data_batch["projs"], 0)).unsqueeze(0).to(device).float()

    poses = torch.inverse(poses[:, :1]) @ poses

    n, nv, c, h, w = images.shape

    net.compute_grid_transforms(projs, poses)
    # RT/DHS branches accept an additional leading `data` dictionary, whereas
    # the original S4C implementation starts directly with `images`. Inspect
    # the bound method explicitly instead of catching TypeError, which could
    # otherwise hide a genuine error raised inside encode().
    encode_parameters = inspect.signature(net.encode).parameters
    if "data" in encode_parameters:
        net.encode({}, images, projs, poses, ids_encoder=[0], ids_render=[0])
    else:
        net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])

    net.set_scale(0)

    use_pseudo_rays = (hasattr(net, 'ray_attention_enabled') and net.ray_attention_enabled
                       and PSEUDO_RAY_K is not None)

    if not use_pseudo_rays:
        points = points.reshape(1, -1, 3)
        _, invalid, sigmas, segs = net.forward(points, predict_segmentation=True)
        return sigmas, segs

    X, Y, Z, _ = points.shape
    K = PSEUDO_RAY_K
    n_segments = X // K
    remainder = X % K

    sigmas_all = []
    segs_all = []

    for seg_i in range(n_segments):
        pts_seg = points[seg_i * K:(seg_i + 1) * K]
        pts_seg = pts_seg.permute(1, 2, 0, 3).reshape(1, Y * Z * K, 3)
        _, _, sigmas_seg, segs_seg = net.forward(
            pts_seg, predict_segmentation=True, ray_K=K)
        sigmas_seg = sigmas_seg.reshape(Y, Z, K).permute(2, 0, 1)
        segs_seg = segs_seg.reshape(Y, Z, K, -1).permute(2, 0, 1, 3)
        sigmas_all.append(sigmas_seg)
        segs_all.append(segs_seg)

    if remainder > 0:
        pts_rem = points[n_segments * K:]
        pts_rem = pts_rem.permute(1, 2, 0, 3).reshape(1, Y * Z * remainder, 3)
        _, _, sigmas_rem, segs_rem = net.forward(
            pts_rem, predict_segmentation=True)
        sigmas_rem = sigmas_rem.reshape(Y, Z, remainder).permute(2, 0, 1)
        segs_rem = segs_rem.reshape(Y, Z, remainder, -1).permute(2, 0, 1, 3)
        sigmas_all.append(sigmas_rem)
        segs_all.append(segs_rem)

    sigmas = torch.cat(sigmas_all, dim=0)
    segs = torch.cat(segs_all, dim=0)

    sigmas = sigmas.reshape(-1).unsqueeze(0).unsqueeze(-1)
    segs = segs.reshape(-1, segs.shape[-1]).unsqueeze(0)

    return sigmas, segs


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def compute_occupancy_numbers_segmentation(y_pred, y_true, fov_mask, labels):
    label_ids = list(labels.keys())[1:]
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()[mask]
    y_true = y_true.flatten()[mask]

    tp = np.zeros(len(label_ids))
    fp = np.zeros(len(label_ids))
    fn = np.zeros(len(label_ids))
    tn = np.zeros(len(label_ids))

    for label_id in label_ids:
        tp[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred == label_id))
        fp[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred == label_id))
        fn[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred != label_id))
        tn[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred != label_id))

    return tp, fp, tn, fn


def compute_occupancy_numbers(y_pred, y_true, fov_mask):
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    occ_true = y_true[mask] > 0
    occ_pred = y_pred[mask] > 0

    tp = np.sum(np.logical_and(occ_true == 1, occ_pred == 1))
    fp = np.sum(np.logical_and(occ_true == 0, occ_pred == 1))
    fn = np.sum(np.logical_and(occ_true == 1, occ_pred == 0))
    tn = np.sum(np.logical_and(occ_true == 0, occ_pred == 0))

    return tp, fp, tn, fn


def save_evaluation_results(cp_path, results, results_table, mean_iou, valid_mean_iou, weighted_mean_iou, label_maps, dataset_length, sizes):
    """
    保存评估结果到与权重一致的目录，文件名以权重名+时间命名
    """
    # 获取权重文件所在目录和文件名
    cp_path = Path(cp_path)
    weight_dir = cp_path.parent
    weight_name = cp_path.stem  # 不带扩展名的文件名
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成输出文件名
    output_filename = f"evaluation_{weight_name}_{timestamp}.txt"
    output_path = weight_dir / output_filename
    
    # 构建结果内容
    content_lines = []
    content_lines.append("# SSCBench Evaluation Results")
    content_lines.append(f"# Checkpoint: {cp_path.name}")
    content_lines.append(f"# Model Directory: {weight_dir}")
    content_lines.append(f"# Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append("")
    content_lines.append("=" * 80)
    content_lines.append("                           SSCBench Evaluation Results")
    content_lines.append("=" * 80)
    content_lines.append("")
    content_lines.append(f"Checkpoint: {cp_path.name}")
    content_lines.append(f"Dataset: SSCBench KITTI ({dataset_length} samples)")
    content_lines.append("")
    
    labels = label_maps["labels"]
    
    for size in sizes:
        content_lines.append("-" * 80)
        content_lines.append(f"Results for size {size}")
        content_lines.append("-" * 80)
        
        recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
        precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
        iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])
        
        content_lines.append("Occupancy Metrics:")
        content_lines.append(f"  - IoU:       {iou*100:.2f}")
        content_lines.append(f"  - Precision: {precision*100:.2f}%")
        content_lines.append(f"  - Recall:    {recall*100:.2f}%")
        content_lines.append("")
        
        recall_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fn_seg"])
        precision_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"])
        iou_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"])
        
        seg_miou = np.mean(np.nan_to_num(iou_seg))
        
        content_lines.append(f"Segmentation Metrics (mIoU: {seg_miou*100:.2f}):")
        content_lines.append("  | Class          | IoU    | Precision | Recall |")
        content_lines.append("  |----------------|--------|-----------|--------|")
        
        for i in range(15):
            class_name = labels[i+1]
            p = precision_seg[i]*100 if not np.isnan(precision_seg[i]) else float('nan')
            r = recall_seg[i]*100 if not np.isnan(recall_seg[i]) else float('nan')
            content_lines.append(f"  | {class_name:<14} | {iou_seg[i]*100:6.2f} | {p:8.2f}% | {r:6.2f}% |")
        
        content_lines.append("")
    
    # 添加汇总表
    content_lines.append("=" * 80)
    content_lines.append("                               Summary Table")
    content_lines.append("=" * 80)
    content_lines.append("")
    content_lines.append("Results table (for copying):")
    content_lines.append("             | 12.8   | 25.6   | 51.2   |")
    content_lines.append("-------------|--------|--------|--------|")
    
    row_names = ["IoU", "Precision", "Recall", "mIoU"] + [labels[i+1] for i in range(15)]
    for i, row_name in enumerate(row_names):
        content_lines.append(f"{row_name:<13}| {results_table[i, 0]*100:6.2f} | {results_table[i, 1]*100:6.2f} | {results_table[i, 2]*100:6.2f} |")
    
    content_lines.append("")
    content_lines.append(f"Mean IoU: {mean_iou*100:.2f}")
    if valid_mean_iou is not None and not np.isnan(valid_mean_iou):
        content_lines.append(f"Valid-class Mean IoU: {valid_mean_iou*100:.2f}")
    else:
        content_lines.append("Valid-class Mean IoU: nan")
    content_lines.append(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(content_lines))
    
    logging.info(f"Evaluation results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    with torch.no_grad():
        main()
