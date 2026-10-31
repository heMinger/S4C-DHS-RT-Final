#!/usr/bin/env python
"""
逐体素漏检定位 + RT 密度诊断 (固定检查点, 不训练)
================================================
目的: 回答 "Ours 相比 S4C 丢掉的正确占据, 到底在哪个环节变为空".
使用与 evaluate_model_sscbench.py 完全一致的体素网格 / 有效掩码 / 阈值,
在同一批验证帧上导出三个固定模型的逐体素占据预测.

三个固定模型 (均在 eval 协议 SIGMA_CUTOFF=0.25 / USE_GROW / 排除遮挡下):
  s4c : S4C 基线 (参照, 无 DHS/RT)
        /home/lmh/s4c/out/training_checkpoint_176000.pt
  ours: 论文 Ours (S4C + DHS + RT, 原实现 da3_z_to_distance=false)
        out/s4c_dhs_rt_e1/s4c_dhs_rt_e1_backend-None-1_20261024-010315/training_checkpoint_53500.pt
  e1  : 修复后 E1 (da3_z_to_distance=true, step 55000)
        out/s4c_dhs_rt_e1/s4c_dhs_rt_e1_backend-None-1_20261029-225544/best_2_model_55000_abs_rel_lidar=-0.1188.pt

占据判定与 evaluate_model_sscbench.py 完全一致:
  occ = grow(sigma) >= SIGMA_CUTOFF(0.25)  且  分割头 argmax 映射后 label > 0
  (分割头预测 sky/rider -> cityscapes_to_label=0 视为空; 见 label_maps.yaml)

四组体素集合 (全部限制在有效掩码 eff = FOV & (target!=255) 内):
  set1  s4c✓ ours✗  = gt_occ & occ_s4c & ~occ_ours  -> 原设计丢掉了什么
  set2  ours✓ s4c✗  = gt_occ & occ_ours & ~occ_s4c  -> 原设计改善了什么
  set3  ours✗ e1✓   = gt_occ & ~occ_ours & occ_e1   -> 当前修复恢复了什么
  set4  GT空  e1✓   = gt_empty & occ_e1             -> 修复付出的误报

对 set1 额外导出 ours 模型的 RT 前密度 (softplus(raw) 修正前) / RT 残差
(sigma_correction) / RT 后密度 (最终 sigma), 用于区分:
  - RT前 > 阈值 且 RT后 <= 阈值 => RT 抑制了本应占据的体素
  - RT前 <= 阈值               => RT 前密度已低于阈值 (原因待分析, 不能直接归为采样)

用法 (在 s4c conda 环境下, 从仓库根目录运行):
  python scripts/diagnose_voxel_miss_rt.py \
      -ssc /mnt/disk1/lmh/sscbench/squashfs-root/sscbench-kitti \
      -vgt /mnt/disk1/lmh/KITTI360/labels \
      [--max_frames 50] [--out_dir out/voxel_miss_diag]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import open_dict
from hydra import compose, initialize
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmarks" / "sscbench"))

from scripts.benchmarks.sscbench.generate_ply_sequence import get_cam_k
from scripts.benchmarks.sscbench.point_utils import read_calib, generate_point_grid, get_fov_mask
from models.bts.model import BTSNet
from models.common.render import NeRFRenderer
from sscbench_dataset import SSCBenchDataset

device = "cuda:0"

VOXEL_SIZE = 0.2
SIGMA_CUTOFF = 0.25
USE_ADDITIONAL_INVALIDS = True
USE_GROW = True
PSEUDO_RAY_K = 64
GRID = (256, 256, 32)  # (x: depth 0..51.2m, y: lateral -25.6..25.6m, z: height)

MODELS = {
    "s4c": {
        "cp": "/home/lmh/s4c/out/training_checkpoint_176000.pt",
        "desc": "S4C baseline (no DHS/RT)",
        # S4C 检查点没有 ray_attention 权重; 必须关掉 RT, 否则会用随机初始化的
        # RaySigmaCorrector 给 S4C 施加随机密度修正, 不再是真正的 S4C 基线.
        "rt_enabled": False,
    },
    "ours": {
        "cp": "out/s4c_dhs_rt_e1/s4c_dhs_rt_e1_backend-None-1_20261024-010315/training_checkpoint_53500.pt",
        "desc": "paper Ours (S4C+DHS+RT, da3_z_to_distance=false)",
        "rt_enabled": True,
    },
    "e1": {
        # 55000 步的 training_checkpoint 已被后续训练覆盖删除; best_2 是该步的
        # 非 EMA 权重 (与 evaluation_training_checkpoint_55000 的 IoU 52.42 对应).
        "cp": "out/s4c_dhs_rt_e1/s4c_dhs_rt_e1_backend-None-1_20261029-225544/best_2_model_55000_abs_rel_lidar=-0.1188.pt",
        "desc": "fixed E1 (da3_z_to_distance=true, step 55000)",
        "rt_enabled": True,
    },
}

DIST_BUCKETS = [("near <12.8m", 0.0, 12.8), ("mid 12.8-25.6m", 12.8, 25.6), ("far >25.6m", 25.6, np.inf)]


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def identify_additional_invalids(target):
    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    invalids[:, :, 7:] = 0
    invalids[target != 0] = 0
    return invalids


_hydra_initialized = False


def _init_hydra():
    global _hydra_initialized
    if not _hydra_initialized:
        # hydra 1.3 要求 config_path 相对且相对调用文件 (scripts/) 解析
        initialize(version_base=None, config_path="../configs", job_name="voxel_miss_diag")
        _hydra_initialized = True


def load_model(cp_path, return_rt=False, rt_enabled=True):
    """加载固定检查点, 复刻 evaluate_model_sscbench.py 的模型重建逻辑."""
    _init_hydra()
    config = compose(config_name="exp_kitti_360", overrides=[])

    cp = torch.load(cp_path, map_location=device)

    with open_dict(config):
        # 复刻 eval 的 RBF 自动检测: 依据 mlp_coarse.lin_in.weight 输入维判断.
        # 103 = 64 features + 39 xyz_code (无 RBF); 107 = 另有 4 个 rbf_code.
        state_dict_peek = cp["model"] if "model" in cp else cp
        if "renderer.net.mlp_coarse.lin_in.weight" in state_dict_peek:
            _d_in = state_dict_peek["renderer.net.mlp_coarse.lin_in.weight"].shape[1]
            config["model_conf"]["use_rbf_encoding"] = (_d_in != 103)
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "z"
        config["model_conf"]["mlp_coarse"]["n_blocks"] = 0
        config["model_conf"]["mlp_coarse"]["d_hidden"] = 64
        config["model_conf"]["encoder"]["d_out"] = 64
        config["model_conf"]["encoder"]["type"] = "monodepth2"
        config["model_conf"]["grid_learn_empty"] = False
        config["model_conf"]["sample_color"] = True
        config["model_conf"]["segmentation_mode"] = "panoptic_deeplab"
        # S4C 基线没有 RayTransformer, 关闭 RT 以复刻其真实前向 (否则随机权重)
        if not rt_enabled:
            if "ray_attention" in config["model_conf"]:
                config["model_conf"]["ray_attention"]["enabled"] = False

    net = BTSNet(config["model_conf"])
    net.return_rt_diagnostics = return_rt

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()
    state_dict = cp["model"] if "model" in cp else cp
    _wrapper.load_state_dict(state_dict, strict=False)
    renderer.to(device)
    renderer.eval()
    torch.cuda.empty_cache()
    return net


def predict_occupancy(net, data, pts_grid, return_rt=False, cityscapes_lut=None):
    """
    对整帧体素网格做前向, 返回:
      sigma    : (256,256,32) 最终密度 (RT 后, 未膨胀)
      seg_label: (256,256,32) 分割 argmax -> SSCBench label (cityscapes_to_label 映射后)
      base/residual/post: 仅 return_rt=True 时返回 RT 前密度 / RT 残差 / RT 后密度 (未膨胀)

    占据判定 (与 evaluate_model_sscbench.py 一致) 需同时满足:
      grow(sigma) >= SIGMA_CUTOFF  且  seg_label > 0 (分割头预测了有效类别)
    """
    images = torch.stack(data["imgs"], dim=0).unsqueeze(0).to(device).float()
    poses = torch.tensor(np.stack(data["poses"], 0)).unsqueeze(0).to(device).float()
    projs = torch.tensor(np.stack(data["projs"], 0)).unsqueeze(0).to(device).float()
    poses = torch.inverse(poses[:, :1]) @ poses

    net.compute_grid_transforms(projs, poses)
    net.encode({}, images, projs, poses, ids_encoder=[0], ids_render=[0])
    net.set_scale(0)

    # eval 里 predict_grid 收到的 points 是 device 上的 torch 张量; 这里把 numpy 网格
    # 转成同款张量, 使 .permute/.unsqueeze 等 torch 语义可用 (否则 numpy 会 AttributeError)
    if not isinstance(pts_grid, torch.Tensor):
        pts_grid = torch.from_numpy(np.asarray(pts_grid)).to(device).float()
    else:
        pts_grid = pts_grid.to(device).float()

    X, Y, Z, _ = pts_grid.shape
    K = PSEUDO_RAY_K
    use_pseudo = (hasattr(net, "ray_attention_enabled") and net.ray_attention_enabled
                  and K is not None)

    def _to_grid(t, d0, d1, d2):
        # t: (1, n_pts, 1) -> (d0, d1, d2) numpy float32
        return t.detach().cpu().numpy().reshape(d0, d1, d2).astype(np.float32)

    sigma_all = []
    seg_all = []
    base_all = []
    residual_all = []
    post_all = []

    if not use_pseudo:
        # 无 RT 时逐点独立, 分块前向与单次前向数学等价 (避免 2M 点一次 OOM)
        pts_flat = pts_grid.reshape(-1, 3)  # (X*Y*Z, 3)
        CHUNK = 262144
        sig_parts, seg_parts, base_parts, res_parts, post_parts = [], [], [], [], []
        for s in range(0, pts_flat.shape[0], CHUNK):
            pts_chunk = pts_flat[s:s + CHUNK].unsqueeze(0)  # (1, C, 3)
            _, _, sigma, segs = net.forward(pts_chunk, predict_segmentation=True)
            sig_parts.append(sigma.detach().cpu().numpy())
            seg_parts.append(segs.detach().cpu().numpy())
            if return_rt:
                d = net._rt_diagnostics
                base_parts.append(d["base_sigma"].detach().cpu().numpy())
                res_parts.append(d["residual"].detach().cpu().numpy())
                post_parts.append(d["post_sigma"].detach().cpu().numpy())
        sigma_all.append(np.concatenate(sig_parts, axis=1).reshape(X, Y, Z).astype(np.float32))
        seg_all.append(np.concatenate(seg_parts, axis=1).argmax(-1).reshape(X, Y, Z).astype(np.int64))
        if return_rt:
            base_all.append(np.concatenate(base_parts, axis=1).reshape(X, Y, Z).astype(np.float32))
            residual_all.append(np.concatenate(res_parts, axis=1).reshape(X, Y, Z).astype(np.float32))
            post_all.append(np.concatenate(post_parts, axis=1).reshape(X, Y, Z).astype(np.float32))
    else:
        n_segments = X // K
        remainder = X % K
        for seg_i in range(n_segments):
            pts_seg = pts_grid[seg_i * K:(seg_i + 1) * K]
            pts_seg = pts_seg.permute(1, 2, 0, 3).reshape(1, Y * Z * K, 3)
            _, _, sigma, segs = net.forward(pts_seg, predict_segmentation=True, ray_K=K)
            sigma_all.append(_to_grid(sigma, Y, Z, K).transpose(2, 0, 1))
            seg_all.append(segs.detach().cpu().numpy().argmax(-1).reshape(Y, Z, K).transpose(2, 0, 1).astype(np.int64))
            if return_rt:
                d = net._rt_diagnostics
                base_all.append(_to_grid(d["base_sigma"], Y, Z, K).transpose(2, 0, 1))
                residual_all.append(_to_grid(d["residual"], Y, Z, K).transpose(2, 0, 1))
                post_all.append(_to_grid(d["post_sigma"], Y, Z, K).transpose(2, 0, 1))
        if remainder > 0:
            pts_rem = pts_grid[n_segments * K:]
            pts_rem = pts_rem.permute(1, 2, 0, 3).reshape(1, Y * Z * remainder, 3)
            _, _, sigma, segs = net.forward(pts_rem, predict_segmentation=True)
            sigma_all.append(_to_grid(sigma, Y, Z, remainder).transpose(2, 0, 1))
            seg_all.append(segs.detach().cpu().numpy().argmax(-1).reshape(Y, Z, remainder).transpose(2, 0, 1).astype(np.int64))
            if return_rt:
                d = net._rt_diagnostics
                base_all.append(_to_grid(d["base_sigma"], Y, Z, remainder).transpose(2, 0, 1))
                residual_all.append(_to_grid(d["residual"], Y, Z, remainder).transpose(2, 0, 1))
                post_all.append(_to_grid(d["post_sigma"], Y, Z, remainder).transpose(2, 0, 1))

    sigma = np.concatenate(sigma_all, axis=0)
    seg_argmax = np.concatenate(seg_all, axis=0)  # (X, Y, Z) 0..18
    seg_label = cityscapes_lut[seg_argmax] if cityscapes_lut is not None else seg_argmax
    if return_rt:
        base = np.concatenate(base_all, axis=0)
        residual = np.concatenate(residual_all, axis=0)
        post = np.concatenate(post_all, axis=0)
        return sigma, seg_label, base, residual, post
    return sigma, seg_label, None, None, None


def grow_sigma(sigma):
    """与 evaluate_model_sscbench.py 的 USE_GROW 一致: 3x3x3 max-pool 膨胀."""
    return F.max_pool3d(
        torch.from_numpy(sigma).float().unsqueeze(0),
        kernel_size=3, stride=1, padding=1,
    ).squeeze(0).numpy()


def analyze(targets, effs, occ, ours_rt, ours_seg, dist_grid, out_dir):
    """汇总四组体素集合统计 + set1 的 RT 密度统计 + BEV 定位图."""
    n = len(targets)

    # 全局体素计数
    gt_occ_total = 0
    gt_empty_total = 0
    # 各模型 TP/FP/FN (占据, 有效掩码内)
    tp = {m: 0 for m in MODELS}
    fp = {m: 0 for m in MODELS}
    fn = {m: 0 for m in MODELS}

    # 四组集合计数
    set_cnt = {"set1_s4c_hit_ours_miss": 0,
               "set2_ours_hit_s4c_miss": 0,
               "set3_ours_miss_e1_hit": 0,
               "set4_gt_empty_e1_occ": 0}

    # set1 的 RT 密度统计 (raw, 未膨胀)
    rt_base_all = []
    rt_post_all = []
    rt_res_all = []
    rt_dist_all = []
    n_rt_suppressed = 0   # RT前>阈值 且 RT后<=阈值
    n_mlp_empty = 0       # RT前<=阈值 (RT 前密度已低于阈值)
    n_seg_gate = 0        # RT前/后都>阈值 但分割头预测空 (sky/rider -> label 0)

    # 全局 RT 翻转统计 (grow + seg 口径, 与 [A] 一致)
    rt_off_occ = 0   # GT-occ   : base 占 -> post 空 (RT 压掉正确占据)
    rt_on_occ = 0    # GT-occ   : base 空 -> post 占 (RT 救回漏检)
    rt_fp_off = 0    # GT-empty : base 误报 -> post 空 (RT 消除误报)
    rt_fp_on = 0     # GT-empty : base 空 -> post 误报 (RT 新增误报)

    # [F] base(关RT假想) vs post(实际ours) 同口径混淆矩阵 (grow + seg gate)
    base_tp = base_fp = base_fn = 0
    post_tp = post_fp = post_fn = 0

    # [G] Ours -> E1 双向转移 (g_recovered 复用 set3)
    g_lost = 0        # GT-occ   : ours 占 -> e1 空
    g_new_fp = 0      # GT-empty : ours 空 -> e1 占 (新增误报)
    g_removed_fp = 0  # GT-empty : ours 占 -> e1 空 (消除误报)

    # 距离桶分桶计数 (set1) + 每桶 GT-occ 总数 (算漏检率)
    bucket_cnt = {b[0]: 0 for b in DIST_BUCKETS}
    gt_occ_bucket = {b[0]: 0 for b in DIST_BUCKETS}

    # BEV 累加图 (sum over z)
    bev_set1 = np.zeros((256, 256), dtype=np.float32)
    bev_set3 = np.zeros((256, 256), dtype=np.float32)
    bev_set2 = np.zeros((256, 256), dtype=np.float32)
    bev_set4 = np.zeros((256, 256), dtype=np.float32)

    for fi in range(n):
        target = targets[fi]
        eff = effs[fi]
        gt_occ = (target > 0) & eff
        gt_empty = (target == 0) & eff
        gt_occ_total += int(gt_occ.sum())
        gt_empty_total += int(gt_empty.sum())

        A = occ["s4c"][fi] & eff
        B = occ["ours"][fi] & eff
        C = occ["e1"][fi] & eff

        for m, O in (("s4c", A), ("ours", B), ("e1", C)):
            tp[m] += int((O & gt_occ).sum())
            fp[m] += int((O & gt_empty).sum())
            fn[m] += int((~O & gt_occ).sum())

        s1 = A & ~B & gt_occ
        s2 = B & ~A & gt_occ
        s3 = ~B & C & gt_occ
        s4 = C & gt_empty

        set_cnt["set1_s4c_hit_ours_miss"] += int(s1.sum())
        set_cnt["set2_ours_hit_s4c_miss"] += int(s2.sum())
        set_cnt["set3_ours_miss_e1_hit"] += int(s3.sum())
        set_cnt["set4_gt_empty_e1_occ"] += int(s4.sum())

        bev_set1 += s1.sum(axis=2)
        bev_set2 += s2.sum(axis=2)
        bev_set3 += s3.sum(axis=2)
        bev_set4 += s4.sum(axis=2)

        # [G] Ours -> E1 双向转移 (g_recovered == set3)
        g_lost += int((B & ~C & gt_occ).sum())
        g_new_fp += int((~B & C & gt_empty).sum())
        g_removed_fp += int((B & ~C & gt_empty).sum())

        # set1 距离桶 (同时累计每桶 GT-occ 总数, 用于漏检率)
        for bname, lo, hi in DIST_BUCKETS:
            in_bucket = (dist_grid >= lo) & (dist_grid < hi)
            bucket_cnt[bname] += int((s1 & in_bucket).sum())
            gt_occ_bucket[bname] += int((gt_occ & in_bucket).sum())

        # set1 的 RT 密度诊断 (ours raw)
        if ours_rt is not None and s1.sum() > 0:
            base = ours_rt[fi]["base"][s1]
            post = ours_rt[fi]["post"][s1]
            res = ours_rt[fi]["residual"][s1]
            d = dist_grid[s1]
            rt_base_all.append(base)
            rt_post_all.append(post)
            rt_res_all.append(res)
            rt_dist_all.append(d)

            base_occ = base >= SIGMA_CUTOFF
            post_occ = post >= SIGMA_CUTOFF
            n_rt_suppressed += int((base_occ & ~post_occ).sum())
            n_mlp_empty += int((~base_occ).sum())
            seg_empty = ours_seg[fi][s1] == 0
            n_seg_gate += int((base_occ & post_occ & seg_empty).sum())

        # [E][F] base(关RT假想) vs post(实际ours) 同口径 (grow + seg gate)
        if ours_rt is not None:
            base_g = grow_sigma(ours_rt[fi]["base"])
            base_occ_full = (base_g >= SIGMA_CUTOFF) & (ours_seg[fi] > 0) & eff
            post_occ_full = B  # = occ["ours"][fi] & eff
            base_tp += int((base_occ_full & gt_occ).sum())
            base_fp += int((base_occ_full & gt_empty).sum())
            base_fn += int((~base_occ_full & gt_occ).sum())
            post_tp += int((post_occ_full & gt_occ).sum())
            post_fp += int((post_occ_full & gt_empty).sum())
            post_fn += int((~post_occ_full & gt_occ).sum())
            rt_off_occ += int((base_occ_full & ~post_occ_full & gt_occ).sum())
            rt_on_occ += int((~base_occ_full & post_occ_full & gt_occ).sum())
            rt_fp_off += int((base_occ_full & ~post_occ_full & gt_empty).sum())
            rt_fp_on += int((~base_occ_full & post_occ_full & gt_empty).sum())

    # ---- 打印统计表 ----
    lines = []
    lines.append("=" * 96)
    lines.append("逐体素漏检定位 + RT 密度诊断结果")
    lines.append("=" * 96)
    lines.append(f"有效体素 (eff): GT-occ {gt_occ_total}, GT-empty {gt_empty_total}")

    lines.append("")
    lines.append("[A] 各模型占据召回/精度/IoU (eval 协议, 有效掩码内)")
    lines.append(f"  {'model':<6}{'Recall%':>10}{'Precision%':>12}{'IoU%':>10}")
    for m in ("s4c", "ours", "e1"):
        rec = 100 * tp[m] / (tp[m] + fn[m] + 1e-9)
        pre = 100 * tp[m] / (tp[m] + fp[m] + 1e-9)
        iou = 100 * tp[m] / (tp[m] + fp[m] + fn[m] + 1e-9)
        lines.append(f"  {m:<6}{rec:>10.2f}{pre:>12.2f}{iou:>10.2f}")

    lines.append("")
    lines.append("[B] 四组体素集合统计")
    lines.append(f"  {'集合':<26}{'体素数':>10}{'占GT-occ%':>12}   含义")
    lines.append(f"  {'set1 s4c✓ ours✗':<26}{set_cnt['set1_s4c_hit_ours_miss']:>10}"
                 f"{100*set_cnt['set1_s4c_hit_ours_miss']/(gt_occ_total+1e-9):>12.2f}   原设计丢掉")
    lines.append(f"  {'set2 ours✓ s4c✗':<26}{set_cnt['set2_ours_hit_s4c_miss']:>10}"
                 f"{100*set_cnt['set2_ours_hit_s4c_miss']/(gt_occ_total+1e-9):>12.2f}   原设计改善")
    lines.append(f"  {'set3 ours✗ e1✓':<26}{set_cnt['set3_ours_miss_e1_hit']:>10}"
                 f"{100*set_cnt['set3_ours_miss_e1_hit']/(gt_occ_total+1e-9):>12.2f}   修复恢复")
    lines.append(f"  {'set4 GT空 e1✓':<26}{set_cnt['set4_gt_empty_e1_occ']:>10}"
                 f"{100*set_cnt['set4_gt_empty_e1_occ']/(gt_empty_total+1e-9):>12.2f}   修复误报(占GT空)")

    lines.append("")
    lines.append("[C] set1 (s4c✓ ours✗) 距离桶分布 (含每桶漏检率)")
    lines.append(f"  {'距离桶':<16}{'set1数':>9}{'占set1%':>9}{'GT-occ数':>11}{'漏检率%':>9}")
    for bname, lo, hi in DIST_BUCKETS:
        n_set1 = bucket_cnt[bname]
        n_gtocc = gt_occ_bucket[bname]
        lines.append(f"  {bname:<16}{n_set1:>9}"
                     f"{100*n_set1/(set_cnt['set1_s4c_hit_ours_miss']+1e-9):>9.1f}"
                     f"{n_gtocc:>11}"
                     f"{100*n_set1/(n_gtocc+1e-9):>9.2f}")

    lines.append("")
    lines.append("[D] set1 的 RT 密度诊断 (raw 未膨胀密度口径; [F] 为 grow+seg 同口径)")
    if len(rt_base_all) > 0:
        base = np.concatenate(rt_base_all)
        post = np.concatenate(rt_post_all)
        res = np.concatenate(rt_res_all)
        dd = np.concatenate(rt_dist_all)
        delta = post - base
        lines.append(f"  set1 体素数: {len(base)}")
        lines.append(f"  RT前 base: mean={base.mean():.4f} median={np.median(base):.4f} "
                     f"p90={np.percentile(base,90):.4f}")
        lines.append(f"  RT后 post: mean={post.mean():.4f} median={np.median(post):.4f}")
        lines.append(f"  Δ(post-base): mean={delta.mean():.4f} median={np.median(delta):.4f}")
        lines.append(f"  RT残差(σ修正,logit空间): mean={res.mean():.4f} median={np.median(res):.4f}")
        lines.append(f"  RT前>=阈值(base>={SIGMA_CUTOFF}) 但 RT后<阈值 => RT 抑制: "
                     f"{n_rt_suppressed} ({100*n_rt_suppressed/len(base):.1f}%)")
        lines.append(f"  RT前<阈值 => RT 前密度已低于阈值 (原因待分析): "
                     f"{n_mlp_empty} ({100*n_mlp_empty/len(base):.1f}%)")
        lines.append(f"  RT前/后都>=阈值 但分割头预测空 (sky/rider): "
                     f"{n_seg_gate} ({100*n_seg_gate/len(base):.1f}%)")
        # 按距离桶细分 (raw 口径)
        lines.append("  按距离桶细分 (raw 口径):")
        for bname, lo, hi in DIST_BUCKETS:
            m = (dd >= lo) & (dd < hi)
            if m.sum() == 0:
                continue
            b_occ = base[m] >= SIGMA_CUTOFF
            p_occ = post[m] >= SIGMA_CUTOFF
            supp = int((b_occ & ~p_occ).sum())
            mlp_empty = int((~b_occ).sum())
            lines.append(f"    {bname:<16}: n={m.sum():>6}, RT抑制={supp:>5} ({100*supp/m.sum():.1f}%), "
                         f"RT前空={mlp_empty:>5} ({100*mlp_empty/m.sum():.1f}%)")
    else:
        lines.append("  (set1 为空, 无 RT 诊断样本)")

    lines.append("")
    lines.append("[E] 全局 RT 翻转 (ours, grow+seg 口径, 与 [A] 一致)")
    lines.append(f"  GT-occ  : RT 占据->空 (flip_off) {rt_off_occ},  空->占据 (flip_on) {rt_on_occ}, "
                 f"净 {rt_on_occ - rt_off_occ:+d}")
    lines.append(f"  GT-empty: RT 误报->空 (消除误报) {rt_fp_off},  空->误报 (新增误报) {rt_fp_on}, "
                 f"净 {rt_fp_on - rt_fp_off:+d}")

    lines.append("")
    lines.append("[F] base(关RT假想) vs post(实际ours) 完整混淆矩阵 (grow+seg 口径)")
    lines.append(f"  {'':<6}{'TP':>10}{'FP':>10}{'FN':>10}{'Recall%':>10}{'Precision%':>12}{'IoU%':>10}")
    for label, (tp_, fp_, fn_) in (("base", (base_tp, base_fp, base_fn)),
                                   ("post", (post_tp, post_fp, post_fn))):
        rec = 100 * tp_ / (tp_ + fn_ + 1e-9)
        pre = 100 * tp_ / (tp_ + fp_ + 1e-9)
        iou = 100 * tp_ / (tp_ + fp_ + fn_ + 1e-9)
        lines.append(f"  {label:<6}{tp_:>10}{fp_:>10}{fn_:>10}{rec:>10.2f}{pre:>12.2f}{iou:>10.2f}")
    lines.append("  注: RT 只修正密度, 不改变分割头输出, 故 base/post 的 seg 门控完全相同;")
    lines.append("      [F] 即 '含语义过滤+后处理' 的最终指标对比 (RT 是否有益的直接判据)。")

    lines.append("")
    lines.append("[G] Ours -> E1 双向转移 (grow+seg 口径, eff 内)")
    g_recovered = set_cnt["set3_ours_miss_e1_hit"]
    lines.append(f"  GT-occ  : 恢复(ours空->e1占) {g_recovered},  丢失(ours占->e1空) {g_lost}, "
                 f"净 {g_recovered - g_lost:+d}")
    lines.append(f"  GT-empty: 新增误报(ours空->e1占) {g_new_fp},  消除误报(ours占->e1空) {g_removed_fp}, "
                 f"净 {g_new_fp - g_removed_fp:+d}")

    text = "\n".join(lines)
    print(text)
    with open(Path(out_dir) / "miss_recovery_stats.txt", "w") as f:
        f.write(text + "\n")

    # ---- BEV 定位图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        (bev_set1, "set1: s4c✓ ours✗ (lost)", "Reds"),
        (bev_set2, "set2: ours✓ s4c✗ (improved)", "Greens"),
        (bev_set3, "set3: ours✗ e1✓ (recovered)", "Blues"),
        (bev_set4, "set4: GT-empty e1✓ (FP)", "Oranges"),
    ]
    for ax, (bev, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(bev.T, origin="lower", extent=[0, 51.2, -25.6, 25.6],
                       aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("depth x (m)")
        ax.set_ylabel("lateral y (m)")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Per-voxel miss / recovery localization (BEV, sum over z)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "bev_localization.png", dpi=120)
    plt.close(fig)

    # ---- 保存原始数据 ----
    np.savez_compressed(
        Path(out_dir) / "set_maps.npz",
        set1=bev_set1, set2=bev_set2, set3=bev_set3, set4=bev_set4,
        set_cnt=np.array(list(set_cnt.values())),
        set_names=np.array(list(set_cnt.keys())),
    )
    print(f"\nSaved: {out_dir}/miss_recovery_stats.txt, bev_localization.png, set_maps.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ssc", "--ssc", "--ssc_path", dest="ssc_path", required=True)
    parser.add_argument("-vgt", "--vgt", "--voxel_gt_path", dest="voxel_gt_path", required=True)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--out_dir", default="out/voxel_miss_diag")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 类别映射 (与 evaluate_model_sscbench.py 一致)
    with open(REPO_ROOT / "scripts" / "benchmarks" / "sscbench" / "label_maps.yaml") as f:
        label_maps = yaml.safe_load(f)
    # cityscapes (19类) -> SSCBench label 的查找表, 用于分割头的 argmax 转换
    cityscapes_lut = np.zeros(256, dtype=np.int64)
    for k, v in label_maps["cityscapes_to_label"].items():
        cityscapes_lut[int(k)] = int(v)

    # ---- 数据 (构造参数与 evaluate_model_sscbench.py 完全一致) ----
    print("Loading SSCBench dataset ...")
    dataset = SSCBenchDataset(
        data_path=args.ssc_path,
        voxel_gt_path=args.voxel_gt_path,
        sequences=(9,),
        target_image_size=(192, 640),
        return_stereo=False,
        frame_count=1,
        color_aug=False,
    )
    dataset.length = min(args.max_frames, dataset.length)

    # ---- 网格 + FOV 掩码 (与 evaluate_model_sscbench.py 一致) ----
    calib = read_calib()
    T_velo_2_cam = calib["Tr"]
    cam_pts, _ = generate_point_grid(
        vox_origin=np.array([0, -25.6, -2]),
        scene_size=(51.2, 51.2, 6.4),
        voxel_size=VOXEL_SIZE,
        cam_E=T_velo_2_cam,
        cam_k=get_cam_k(),
    )
    fov_mask = get_fov_mask()  # (256, 256, 32) bool

    # cam_pts: (N,3) 相机系坐标; 展平顺序为 (x 深度, y 横向, z 高度), z 最快
    pts_grid = cam_pts.reshape(256, 256, 32, 3).astype(np.float32)  # (X, Y, Z, 3)
    # 距离网格: 到相机原点的欧氏距离 (米), 用于距离桶
    dist_grid = np.sqrt((cam_pts ** 2).sum(axis=-1)).reshape(256, 256, 32).astype(np.float32)

    print(f"诊断帧数: {dataset.length}")

    # ---- 预计算 GT / eff / 帧数据 (模型无关, 只做一次) ----
    targets = []
    effs = []
    frames = []
    for fi in range(dataset.length):
        data = dataset[fi]
        # GT 转换到 label 空间, 并应用额外无效掩码 (与 eval 顺序一致)
        target = convert_voxels(data["voxel_gt"][0].astype(int),
                                label_maps["sscbench_to_label"])
        if USE_ADDITIONAL_INVALIDS:
            invalids = identify_additional_invalids(target)
            target[invalids == 1] = 255
        targets.append(target)
        eff = (target != 255) & fov_mask
        effs.append(eff)
        frames.append(data)

    # ---- 逐模型加载一次, 逐帧前向 (每次只驻留一个模型, 省显存) ----
    occ = {m: [None] * dataset.length for m in MODELS}
    ours_rt = [None] * dataset.length  # 每帧 dict(base, post, residual)
    ours_seg = [None] * dataset.length

    for m in MODELS:
        net = load_model(MODELS[m]["cp"], return_rt=(m == "ours"),
                         rt_enabled=MODELS[m]["rt_enabled"])
        for fi in range(dataset.length):
            sigma_raw, seg_label, base, residual, post = predict_occupancy(
                net, frames[fi], pts_grid, return_rt=(m == "ours"),
                cityscapes_lut=cityscapes_lut)
            sigma_grown = grow_sigma(sigma_raw)
            # 占据判定与 eval 一致: grow(sigma)>=cutoff 且 分割头预测有效类别
            occ[m][fi] = (sigma_grown >= SIGMA_CUTOFF) & (seg_label > 0)
            if m == "ours":
                ours_rt[fi] = {"base": base, "post": post, "residual": residual}
                ours_seg[fi] = seg_label
            print(f"  {m} frame {fi} done")
        del net
        torch.cuda.empty_cache()

    # ---- 分析 ----
    analyze(targets, effs, occ, ours_rt, ours_seg, dist_grid, out_dir)


if __name__ == "__main__":
    main()
