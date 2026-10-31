#!/usr/bin/env python
"""
采样点误差诊断脚本 (纯 CPU, 无需模型/GPU)
=============================================

目的:
    定量分解三维 Recall 低的原因 —— 是"采样点误差"还是"预测深度误差"。
    本脚本严格复现训练中的 DHS 自适应采样逻辑
    (models/common/render/nerf.py `sample_adaptive` + ray_sampler.py 的
     DA3 轴向 z → 沿射线距离转换), 确保统计口径与训练一致。

三种深度:
    d_z    : DA3 先验深度 (轴向 z-buffer, 单位米) —— data["depths"]
    d_gt_z : LiDAR 真值深度 (轴向 z-buffer, 单位米) —— data["lidar_depths"]
    d_ray  : 沿射线距离 = d_z / cos(theta) —— 训练 da3_z_to_distance=true 时的采样中心

复现的采样逻辑 (与 nerf.py `sample_adaptive` 逐行一致):
    1. norm_conf        = clip((conf - 1) / 5.5535, conf_epsilon=0.1, 1.0)
    2. effective_absrel = absrel_prior(0.12) / norm_conf
    3. sigma            = max(min_thickness(0.5), effective_absrel * d_ray)
    4. 表面采样: n_surface 点均匀铺在 [d_ray-2σ, d_ray+2σ], 裁剪到 [z_near, z_far]
    5. 全局采样: n_global 点逆深度均匀铺在 [z_near, z_far]
    6. 回退 (fallback): norm_conf < conf_threshold(0.2) 或 d_ray < z_near+0.01
       → 全部 n_surface+n_global 个点走全局逆深度采样 (sample_coarse), 无表面窗口

输出 (按沿射线距离 d_ray_gt 分桶 <12.8 / 12.8-25.6 / >25.6 米):
    A. 采样中心误差: 像素数 / 回退率 / abs_rel / RMSE / a1(1.25, 标准对称) / 2σ命中率
    B. 采样点分辨率: 表面窗口宽 / 表面点间距 / 全局点间距 / 表面/全局达标率

相对误差 (abs_rel / a1) 在轴向 z 与沿射线距离下等价 (同一像素 cosθ 相同, 分子分母
同乘 cosθ 抵消); RMSE / σ / 间距 是绝对量, 必须换算到沿射线方向 (乘除 cosθ)。

修正旧版误区:
    * a1 用标准对称公式 ratio = max(d1/d2, d2/d1) < 1.25, 而非 rel_err < 0.25
      (后者在低估方向把 ratio<1.333 以内也误判为通过, 阈值不对称)。
    * 表面窗口按 [z_near, z_far] 裁剪, 实际层宽 = d_max - d_min, 而非恒为 4σ。
    * 低置信度像素走 64 点全局采样, 不进入表面窗口统计 (旧版仍统计其 32 点窗口)。
    * 无置信图时 effective_absrel = absrel_prior(0.12), 而非 ones→norm_conf=0.1→1.2
      (旧版会把窗口放大 10 倍)。
    * "surface 达标率" 是训练采样分辨率的度量, 不是 Recall 的结构性上限:
      SSCBench 评估在稠密体素网格上直接查询 MLP (downsample_and_predict),
      训练未采到的位置也能被预测; 该指标刻画的是训练信号(梯度)是否覆盖到体素级
      分辨率, 而非评估时的查询可达性。

用法:
    python scripts/diagnose_sampling_depth_error.py [--max-samples N] [--no-conf]
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_util import make_datasets

ABSREL_PRIOR = 0.12
MIN_THICKNESS = 0.5
CONF_EPSILON = 0.1
CONF_SCALE = 5.5535
CONF_THRESHOLD = 0.2
DIST_BINS = [(0.0, 12.8), (12.8, 25.6), (25.6, float("inf"))]


def bin_name(lo, hi):
    if lo == 0:
        return f"<{hi}"
    if hi == float("inf"):
        return f">{lo}"
    return f"{lo}-{hi}"


def norm_conf(conf):
    return np.clip((conf - 1.0) / CONF_SCALE, CONF_EPSILON, 1.0)


def cos_theta_map(K, H, W):
    """复现 models/common/util/util.py `unproj_map(...)[..., 2]`:
    K 为归一化内参 ([-1,1] 坐标), fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]。
    返回 (H, W) 的 cos(theta), theta 为射线与光轴(+z)夹角。
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)  # (W,)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)  # (H,)
    X = (x[None, :] - cx) / fx  # (1, W)
    Y = (y[:, None] - cy) / fy  # (H, 1)
    return 1.0 / np.sqrt(X * X + Y * Y + 1.0)  # (H, W)


def standard_a1(d_pred, d_gt):
    """标准 a1 (threshold 1.25): ratio = max(d_pred/d_gt, d_gt/d_pred) < 1.25"""
    ratio = np.maximum(d_pred / d_gt, d_gt / d_pred)
    return ratio < 1.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多诊断多少样本 (默认全部 test split, 上限 256)")
    parser.add_argument("--n-surface", type=int, default=32,
                        help="表面采样点数 (默认 32, 对应当前 DHS 配置)")
    parser.add_argument("--n-global", type=int, default=32,
                        help="全局采样点数 (默认 32)")
    parser.add_argument("--voxel-size", type=float, default=0.2,
                        help="体素尺寸/米 (默认 0.2, 对应 SSC VOXEL)")
    parser.add_argument("--z-near", type=float, default=3.0)
    parser.add_argument("--z-far", type=float, default=80.0)
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD,
                        help="norm_conf 低于此值走全局回退 (默认 0.2)")
    parser.add_argument("--no-conf", action="store_true",
                        help="模拟 conf_values=None: effective_absrel=absrel_prior, 无回退")
    args = parser.parse_args()

    data_conf = OmegaConf.load("configs/data/kitti_360.yaml")
    _, test_dataset = make_datasets(data_conf)

    # 只加载诊断所需的 depth / lidar / conf / projs, 关闭无关分支以加速
    test_dataset.return_depth = True
    test_dataset.return_lidar_depth = True
    test_dataset.return_stereo = False
    test_dataset.return_fisheye = False
    test_dataset.return_segmentation = False
    test_dataset.frame_count = 1
    test_dataset.color_aug = False
    test_dataset.length = min(256, test_dataset.length)
    if args.max_samples is not None:
        test_dataset.length = min(args.max_samples, test_dataset.length)

    n = test_dataset.length
    print(f"诊断样本数: {n}")

    n_total = args.n_surface + args.n_global

    # 每桶累积原始量 (逐像素), 最后统一聚合
    acc = {bin_name(lo, hi): {"abs_err": [], "rel_err": [], "a1_ok": [],
                              "hit": [], "fb": [], "win": [], "surf_sp": [],
                              "glob_sp": [], "fb_sp": [], "n": 0}
           for lo, hi in DIST_BINS}
    total_px = 0
    total_hit = 0
    total_fb = 0
    total_nonfb = 0

    # 逆深度均匀采样的点间距因子: dz/dt = z^2 * (1/near - 1/far)
    inv_disp = (1.0 / args.z_near) - (1.0 / args.z_far)

    for i in range(n):
        data = test_dataset[i]
        da3 = np.asarray(data["depths"][0], dtype=np.float32)          # 轴向 z
        lidar = np.asarray(data["lidar_depths"][0], dtype=np.float32)  # 轴向 z
        da3 = da3.reshape(da3.shape[-2], da3.shape[-1])
        lidar = lidar.reshape(lidar.shape[-2], lidar.shape[-1])
        H, W = da3.shape

        valid = (lidar > 0) & (da3 > 0)
        if valid.sum() == 0:
            continue

        # cos(theta) 每像素 (训练 da3_z_to_distance=true 时 /cosθ 转沿射线距离)
        projs = data.get("projs", [])
        if len(projs) and projs[0] is not None:
            K = np.asarray(projs[0], dtype=np.float32)
            cos_th = cos_theta_map(K, H, W)
            cos_th = np.maximum(cos_th, 1e-6)
        else:
            cos_th = np.ones((H, W), dtype=np.float32)

        d_da3 = da3[valid] / cos_th[valid]   # 沿射线距离
        d_gt = lidar[valid] / cos_th[valid]  # 沿射线距离

        abs_err = np.abs(d_da3 - d_gt)
        rel_err = abs_err / d_gt
        a1_ok = standard_a1(d_da3, d_gt)

        if args.no_conf:
            # 复现 sample_adaptive 的 conf_values=None 分支
            effective_absrel = np.full_like(d_da3, ABSREL_PRIOR)
            low_conf = np.zeros_like(d_da3, dtype=bool)
        else:
            conf = np.asarray(data["confs"][0], dtype=np.float32)
            conf = conf.reshape(conf.shape[-2], conf.shape[-1])[valid]
            nc = norm_conf(conf)
            effective_absrel = ABSREL_PRIOR / nc
            low_conf = nc < args.conf_threshold

        invalid_depth = d_da3 < (args.z_near + 0.01)
        fb = low_conf | invalid_depth  # 回退: 走 64 点全局逆深度采样

        sigma = np.maximum(MIN_THICKNESS, effective_absrel * d_da3)

        # 表面窗口 [d-2σ, d+2σ] 裁剪到 [z_near, z_far]
        d_min = np.maximum(d_da3 - 2.0 * sigma, args.z_near)
        d_max = np.minimum(d_da3 + 2.0 * sigma, args.z_far)
        win = np.maximum(d_max - d_min, 0.0)
        surf_sp = win / args.n_surface
        hit = (d_gt >= d_min) & (d_gt <= d_max)

        # 全局逆深度间距 (沿射线方向, 米)
        glob_sp = (d_gt ** 2) * inv_disp / args.n_global  # 非回退: n_global 点
        fb_sp = (d_gt ** 2) * inv_disp / n_total          # 回退: n_surface+n_global 点

        total_px += d_gt.size
        total_hit += int(hit[~fb].sum())
        total_fb += int(fb.sum())
        total_nonfb += int((~fb).sum())

        for lo, hi in DIST_BINS:
            mask = (d_gt > lo) & (d_gt <= hi)
            if mask.sum() == 0:
                continue
            key = bin_name(lo, hi)
            acc[key]["abs_err"].append(abs_err[mask])
            acc[key]["rel_err"].append(rel_err[mask])
            acc[key]["a1_ok"].append(a1_ok[mask])
            acc[key]["hit"].append(hit[mask])
            acc[key]["fb"].append(fb[mask])
            acc[key]["win"].append(win[mask])
            acc[key]["surf_sp"].append(surf_sp[mask])
            acc[key]["glob_sp"].append(glob_sp[mask])
            acc[key]["fb_sp"].append(fb_sp[mask])
            acc[key]["n"] += int(mask.sum())

    # ============================================================
    # A. 采样中心误差
    # ============================================================
    print("\n" + "=" * 100)
    print("A. 采样中心误差 (DA3 vs LiDAR GT, 沿射线距离)")
    print(f"{'距离桶(米)':<14} {'像素数':>9} {'回退率':>8} {'abs_rel':>9} "
          f"{'RMSE':>9} {'a1(1.25)':>9} {'2σ命中率':>9}")
    print("-" * 100)
    for lo, hi in DIST_BINS:
        key = bin_name(lo, hi)
        a = acc[key]
        if a["n"] == 0:
            print(f"{key:<14} {0:>9} {'-':>8} {'-':>9} {'-':>9} {'-':>9} {'-':>9}")
            continue
        abs_err = np.concatenate(a["abs_err"])
        rel_err = np.concatenate(a["rel_err"])
        a1_ok = np.concatenate(a["a1_ok"])
        hit = np.concatenate(a["hit"])
        fb = np.concatenate(a["fb"])
        nonfb = ~fb
        rmse = float(np.sqrt((abs_err ** 2).mean()))
        a1 = float(a1_ok.mean())
        abs_rel = float(rel_err.mean())
        fb_rate = float(fb.mean())
        hit_rate = float(hit[nonfb].mean()) if nonfb.any() else float("nan")
        print(f"{key:<14} {a['n']:>9} {fb_rate:>8.4f} {abs_rel:>9.4f} "
              f"{rmse:>9.3f} {a1:>9.4f} {hit_rate:>9.4f}")
    print("=" * 100)
    print(f"整体: 像素数 {total_px}, 回退率 {total_fb / max(total_px, 1):.4f}, "
          f"2σ命中率(非回退) {total_hit / max(total_nonfb, 1):.4f}")
    print()

    # ============================================================
    # B. 采样点分辨率 (沿射线方向, 米) vs 体素尺寸
    # ============================================================
    print("=" * 100)
    print("B. 采样点分辨率 (沿射线方向, 米; voxel = {:.2f}m)".format(args.voxel_size))
    print(f"{'距离桶(米)':<14} {'窗口宽':>8} {'表面间距':>8} {'全局间距':>8} "
          f"{'回退间距':>8} {'表面达标':>8} {'全局达标':>8}")
    print("-" * 100)
    for lo, hi in DIST_BINS:
        key = bin_name(lo, hi)
        a = acc[key]
        if a["n"] == 0:
            print(f"{key:<14} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        win = np.concatenate(a["win"])
        surf_sp = np.concatenate(a["surf_sp"])
        glob_sp = np.concatenate(a["glob_sp"])
        fb_sp = np.concatenate(a["fb_sp"])
        fb = np.concatenate(a["fb"])
        nonfb = ~fb
        win_mean = float(win[nonfb].mean()) if nonfb.any() else float("nan")
        surf_mean = float(surf_sp[nonfb].mean()) if nonfb.any() else float("nan")
        glob_mean = float(glob_sp[nonfb].mean()) if nonfb.any() else float("nan")
        fb_mean = float(fb_sp[fb].mean()) if fb.any() else float("nan")
        surf_ok = float((surf_sp[nonfb] <= args.voxel_size).mean()) if nonfb.any() else float("nan")
        glob_ok = float((glob_sp[nonfb] <= args.voxel_size).mean()) if nonfb.any() else float("nan")
        print(f"{key:<14} {win_mean:>8.3f} {surf_mean:>8.3f} {glob_mean:>8.3f} "
              f"{fb_mean:>8.3f} {surf_ok:>8.4f} {glob_ok:>8.4f}")
    print("=" * 100)
    print()
    print("判读:")
    print("  - 2σ命中率(非回退) 高 + abs_rel 大 => 采样带盖住表面但带过宽(sigma∝距离),")
    print("    表面点铺在宽带上分辨率不足 —— 采样点稀疏是主因。")
    print("  - 2σ命中率(非回退) 低 => DA3 先验系统性偏, 采样中心错失表面。")
    print("  - 回退率 高 => 大量像素 norm_conf<阈值, 根本没用表面窗口(走 64 点全局采样),")
    print("    表面窗口再宽也与之无关; 此时应先收紧 conf 的 σ 放大(见 P2)。")
    print("  - '表面/全局达标率' 是训练采样分辨率的度量(间距<=voxel 的像素占比), 不是")
    print("    Recall 的结构性上限: SSCBench 评估在稠密体素网格上直接查询 MLP, 训练未")
    print("    采到的位置也能被预测。达标率<1 说明训练信号在该桶未覆盖到体素级分辨率。")
    print("  - 结合训练 validate 的 a1_lidar / abs_rel_lidar (预测 vs LiDAR GT) 即可")
    print("    定量区分 '采样误差' vs '预测误差'。")


if __name__ == "__main__":
    main()
