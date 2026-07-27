# DA3 在 KITTI-360 上的表现与设置分析

> 基于 DA3 论文 [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/pdf/2511.10647) (arXiv:2511.10647) 与本地在 KITTI-360 上的使用/评估文档整理。  
> **重要**：论文官方 benchmark **未包含 KITTI-360**，仅包含标准 **KITTI**（Eigen 等）及 NYU、ETH3D、DIODE 等；KITTI-360 上的数值来自本项目本地评估。

---

## 一、论文中的深度估计设定与指标（与 KITTI-360 无关）

### 1.1 论文里"KITTI"指什么

- 论文中出现的 **KITTI** 均指**标准 KITTI 深度 benchmark**（如 Eigen split、常用单目深度评估协议），**不是 KITTI-360**。
- 官方 Visual Geometry Benchmark 使用的数据集为：**HiRoom, ETH3D, DTU, 7Scenes, ScanNet++**，同样**不包含 KITTI-360**。

因此，**"DA3 在 KITTI-360 上的表现"在论文中没有任何表格或实验**；下文先归纳论文中的深度指标与输入设置，再说明在 KITTI-360 上如何用 DA3、以及本地已观测到的表现。

---

### 1.2 深度估计相关指标（论文中）

| 类型 | 指标 | 含义 |
|------|------|------|
| **相对深度** | **δ1 (δ<1.25)** | 预测与 GT 比值落在 (1/1.25, 1.25) 的像素比例，↑ 越好 |
| | **δ2 / δ3** | 阈值 1.25²、1.25³ 下的比例 |
| **误差** | **AbsRel** | 平均 \|pred−gt\|/gt，↓ 越好 |
| | **SqRel** | 平均 ((pred−gt)²/gt)，↓ 越好 |
| | **RMSE / RMSE_log** | 均方根误差（线性/对数），↓ 越好 |
| **几何 benchmark** | **F1 (reconstruction)** | 重建点云与 GT 点云的 F1（precision/recall @ 距离阈值），↑ 越好 |
| | **CD (Chamfer Distance)** | 仅 DTU 使用，单位 mm，↓ 越好 |
| **位姿** | **Auc3 / Auc30** | 相对旋转/平移精度曲线下面积 @ 3°、30°，↑ 越好 |

论文中**单目/公制深度表格**主要报告：**δ1↑, AbsRel↓, SqRel↓**；部分表格带 RMSE。

---

### 1.3 单目深度（Monocular）—— 1 帧输入

- **输入**：**单张图像**（Nv=1）。多视图架构下 Nv=1 时退化为单目，无额外计算开销。
- **输出**：相对深度（scale-shift 不变），评估时按常规协议做 median scaling 等与 GT 对齐。
- **训练**：Teacher–Student；Teacher 仅在合成数据上训练，Student 用 Teacher 伪标签 + 无标签数据。
- **论文报告**（Tab. 4, Tab. 10，均为**标准 KITTI / NYU / SINTEL / ETH3D / DIODE**，非 KITTI-360）：

| 方法 | KITTI δ1↑ | NYU δ1↑ | SINTEL δ1↑ | ETH3D δ1↑ | DIODE δ1↑ |
|------|-----------|----------|------------|-----------|-----------|
| DA2 | 94.6 | 97.9 | 77.2 | 86.5 | 95.2 |
| VGGT | 91.7 | 97.9 | 67.9 | 97.5 | 95.3 |
| **DA3** | **95.3** | 97.4 | 75.5 | **98.6** | 95.4 |
| Teacher | 97.2 | 97.9 | 81.4 | 99.8 | 96.6 |

| 方法 | KITTI δ1↑ | NYU δ1↑ | SINTEL δ1↑ | ETH3D δ1↑ | DIODE δ1↑ |
|------|-----------|----------|------------|-----------|-----------|
| DA2 | 94.6 | 97.9 | 77.2 | 86.5 | 95.2 |
| **mono-student** | **97.1** | **98.0** | **82.3** | **98.8** | **96.5** |

- **结论**：单目设定下，**1 帧输入**，DA3 在标准 KITTI 等数据集上优于 DA2/VGGT；**论文未给出 KITTI-360**。

---

### 1.4 公制深度（Metric）—— 单目 + 尺度对齐

- **输入**：单张图；训练时用 **canonical camera 变换**（fc=300）和 GT 公制深度做 scale-shift 对齐。
- **训练**：14 个数据集（含 Taskonomy, DIML, DDAD, Argoverse, Waymo, ScanNet++, ArkitScenes, Cityscapes 等）；batch 64，160K iters；分辨率 base 504，多种宽高比；20% 概率用原始 GT、80% 用 Teacher 对齐后的预测做监督。
- **论文报告**（Tab. 11，**KITTI 为标准 KITTI，非 KITTI-360**）：

| 方法 | NYUv2 δ1↑/AbsRel↓ | KITTI δ1↑/AbsRel↓ | ETH3D δ1↑/AbsRel↓ | SUN-RGBD δ1↑/AbsRel↓ | DIODE δ1↑/AbsRel↓ |
|------|-------------------|-------------------|-------------------|----------------------|-------------------|
| DepthPro | 0.932/0.093 | 0.843/0.121 | 0.386/0.349 | 0.950/0.126 | 0.734/0.173 |
| Metric3D v2 | 0.971/0.067 | **0.976/0.051** | 0.830/0.138 | 0.954/0.132 | - |
| UniDepthv1 | **0.980/0.061** | 0.978/0.051 | 0.234/0.464 | 0.971/0.113 | 0.570/0.266 |
| UniDepthv2 | 0.968/0.064 | 0.968/0.076 | 0.863/0.152 | **0.977/0.111** | 0.856/0.123 |
| **DA3-metric** | 0.963/0.070 | 0.953/0.086 | **0.917/0.104** | 0.973/**0.105** | **0.838/0.128** |

- DA3-metric 在 **ETH3D** 上领先明显（δ1=0.917），KITTI/NYU 上为竞争水平；**所有结果均为标准 KITTI，非 KITTI-360**。

---

### 1.5 多视图（Any-view）—— 帧数与分辨率

- **训练**（论文 Sec. 3.4）：  
  - 基础分辨率 **504×504**，宽高比多变（如 504×378, 504×336, 896×504 等）。  
  - **视图数**：在 504×504 下从 **[2, 18]** 均匀采样。  
  - 128×H100，200k steps；约 120k steps 后从 GT 深度过渡到 Teacher 伪标签；pose conditioning 随机开启概率 0.2。
- **Benchmark 评估**（pose / reconstruction）：  
  - 每场景最多 **100 张图**（若超过则随机采样）；**未固定"多少帧"**，按场景可用图像数而定。  
  - 重建：用预测位姿 + 预测深度做 TSDF 融合，再与 GT 点云比 F1/CD。
- **NVS（novel view synthesis）**（Tab. 5）：  
  - **12 张** context views，farthest point sampling；target 每 8 帧采 1 帧；分辨率 **270×480**。

**小结**：  
- **单目**：1 帧；指标 δ1, AbsRel, SqRel 等；**KITTI 指标准 KITTI**。  
- **多视图**：训练 2–18 帧，benchmark 最多 100 帧/场景，NVS 固定 12 帧输入。  
- **全文未出现 KITTI-360**。

---

## 二、DA3 模型架构与 Metric Depth 机制

### 2.1 DA3NESTED-GIANT-LARGE 双分支架构

DA3NESTED-GIANT-LARGE（1.40B 参数）是一个 **Nested（嵌套）** 架构，内部包含两个独立分支：

```
NestedDepthAnything3Net
├── self.da3          # Giant 分支（any-view）: 相对深度 + 位姿估计 + Pose Conditioning
└── self.da3_metric   # Large 分支（metric）:  直接输出 metric depth + 天空分割
```

**Forward 流程**（`da3.py` NestedDepthAnything3Net.forward）：

1. **Giant 分支**：`output = self.da3(x, extrinsics, intrinsics, ...)` → 相对深度 + 估计位姿
2. **Metric Large 分支**：`metric_output = self.da3_metric(x)` → **仅用图像，直接预测 metric depth**
3. **Metric Scaling**：`apply_metric_scaling(metric_depth, intrinsics)` → `depth * (focal / 300.0)`
4. **Depth Alignment**：`least_squares_scale_scalar(metric_depth, giant_depth)` → 用 metric 分支的绝对尺度对齐 Giant 的相对深度
5. **Sky Handling**：天空区域设为最大深度（默认 200m）

**关键：Metric depth 不需要 LiDAR 或外部深度 GT**。Large 分支从训练数据中学到了绝对尺度（canonical camera + GT metric depth 联合训练），推理时仅需图像。

### 2.2 DA3METRIC-LARGE 独立模型

DA3METRIC-LARGE（0.35B 参数，[HuggingFace](https://huggingface.co/depth-anything/DA3METRIC-LARGE)）是 Nested 中 metric 分支的独立版本：

| 属性 | DA3METRIC-LARGE | DA3NESTED-GIANT-LARGE |
|------|-----------------|----------------------|
| 参数量 | 0.35B | 1.40B |
| 权重大小 | 1.3 GB | 6.8 GB |
| Backbone | ViT-Large | Giant + Large（双分支） |
| Metric Depth | ✅（直接输出） | ✅（内部 metric 分支） |
| Pose Estimation | ❌ | ✅ |
| Pose Conditioning | ❌ | ✅ |
| 3D Gaussians | ❌ | ✅ |
| License | **Apache 2.0** | CC BY-NC 4.0 |

### 2.3 Pose Conditioning 与 Umeyama 对齐

当用户提供 extrinsics（相机位姿）时，`api.py` 在模型推理之后执行额外的对齐步骤：

```python
# api.py: _align_to_input_extrinsics_intrinsics
_, _, scale, aligned_extrinsics = align_poses_umeyama(
    prediction.extrinsics,    # 模型预测的位姿
    extrinsics.numpy(),       # 用户提供的位姿（GT或SLAM等）
    ransac=..., random_state=42,
)
if align_to_input_ext_scale:
    prediction.depth /= scale  # 将深度缩放到用户位姿的坐标系
```

这一步将模型预测的位姿与用户提供的位姿做 Umeyama Sim(3) 对齐，**用位姿的平移尺度来进一步校准深度的绝对尺度**。注意：
- 这里用的是**相机位姿**（来自 GT、SLAM 或视觉里程计），**不是 LiDAR**
- 这是可选的后处理步骤，不提供位姿时模型仍能输出 metric depth（来自 metric 分支）

---

## 三、DA3 在 KITTI-360 上的使用方式与本地表现

论文未做 KITTI-360 实验；以下为**本项目在 KITTI-360 上使用 DA3 的设定与已观测表现**。

### 3.1 四种用法与评估结果

基于 10 组帧、40 个透视视图（image_00 + image_01）在 KITTI-360 seq 0000 上的本地评估（`test_metric_vs_nested.py`）。

#### median_ratio 定义

```python
median_ratio = np.median(gt_lidar / pred_depth)
```

- **= 1.0**：预测深度尺度与 LiDAR 真值完全一致
- **< 1.0**（如 0.748）：预测值系统性偏大，pred ≈ gt / 0.748 ≈ 1.34 × gt
- **> 1.0**（如 1.5）：预测值系统性偏小

#### RAW 指标（绝对尺度，不做任何对齐）

| 方法 | N | abs_rel↓ | δ<1.25↑ | RMSE↓ | median_ratio | 速度 |
|------|---|---------|---------|-------|-------------|------|
| **NESTED MV+pose** | 40 | **0.186** | **0.781** | **4.19** | **0.917** | 0.09 s/img |
| NESTED 单图 | 40 | 0.371 | 0.328 | 6.34 | 1.042±0.34 | 0.16 s/img |
| **DA3METRIC-LARGE** | 40 | 0.371 | 0.251 | 5.72 | 0.748 | **0.08 s/img** |
| OLD 离线 (data_2d_depth_da3) | 40 | 0.371 | 0.252 | 5.72 | 0.748 | — |

#### ALIGNED 指标（对齐后，反映纯深度结构质量）

| 方法 | N | abs_rel↓ | δ<1.25↑ | RMSE↓ | align_scale |
|------|---|---------|---------|-------|------------|
| **NESTED MV+pose** | 40 | **0.105** | **0.882** | 3.97 | 0.917 |
| DA3METRIC-LARGE | 40 | 0.110 | 0.883 | 3.94 | 0.748 |
| OLD 离线 | 40 | 0.110 | 0.883 | 3.94 | 0.748 |
| NESTED 单图 | 40 | 0.121 | 0.853 | 4.20 | 1.042 |

#### 鱼眼视图（image_02/03，重采样为 pinhole 后，80 视图评估）

| 方法 | abs_rel (RAW)↓ | δ<1.25 (RAW)↑ | median_ratio | abs_rel (ALIGNED)↓ |
|------|---------------|---------------|-------------|-------------------|
| **NESTED MV+pose** | **0.181** | **0.766** | 1.092 | **0.139** |
| NESTED 单图 | 0.410 | 0.087 | 1.591 | 0.153 |

### 3.2 关键发现

1. **DA3METRIC-LARGE ≈ OLD 离线**：两者指标几乎完全一致（abs_rel 0.371, ratio 0.748），证实当前 `data_2d_depth_da3/` 本质上就是 DA3METRIC-LARGE（或 Nested 中等效 metric 分支）的输出。

2. **Metric depth 不需要 focal 二次缩放**：DA3METRIC-LARGE 的 raw 输出已经过内部 `apply_metric_scaling`（乘 focal/300），直接就是 metric depth（median_ratio 0.748）。手动再乘 focal 反而变差（ratio 0.406）。

3. **对齐后所有方法精度相近**：abs_rel 均在 0.105~0.121，说明 DA3 的**相对深度结构一致可靠**，差异主要在**绝对尺度**。

4. **NESTED MV+pose 绝对尺度最好**：ratio 0.917，最接近理想值 1.0，但仍有 ~8% 的偏差。

5. **DA3METRIC-LARGE 速度最快**：0.08 s/img（0.35B），比 NESTED 单图（0.16 s/img, 1.40B）快一倍，且**深度结构质量几乎相同**。

6. **鱼眼重采样后 DA3 可用**：MV+pose 的鱼眼 abs_rel 0.181（RAW），远好于当前训练中的全零深度。

---

## 四、对训练管线的影响：absrel_prior 设置

S4C 训练中 DA3 深度用于**自适应采样**（`sample_adaptive`），采样窗口为 `[d - 2σ, d + 2σ]`，其中 `σ = max(min_thickness, absrel_prior × d)`。

### 4.1 当前配置的问题

当前 `data_2d_depth_da3/` 的 median_ratio = 0.748（预测偏大 34%），配合 `absrel_prior = 0.10`：

```
真实深度 d_gt = 20m 时:
  预测 d_pred ≈ 26.7m (= 20 / 0.748)
  σ = 0.10 × 26.7 = 2.67m
  采样窗口 = [21.3, 32.0]m
  真实表面在 20m → 在窗口外！
```

**真实表面可能落在采样窗口外**，导致自适应采样效果受限。

### 4.2 不同方案的 absrel_prior 推荐

| 深度来源 | median_ratio | 尺度偏差 | 推荐 absrel_prior | 理由 |
|---------|-------------|---------|-------------------|------|
| OLD (METRIC-LARGE) | 0.748 | 偏大 34% | 0.35 或 乘 0.748 校正后用 0.12 | 需覆盖 34% 系统偏差 |
| NESTED MV+pose | 0.917 | 偏大 8% | **0.15** | 实测 abs_rel ≈ 0.12，留余量 |
| NESTED MV+pose (鱼眼) | 1.092 | 偏小 8% | **0.20** | 实测 abs_rel ≈ 0.18 |

### 4.3 最高效训练策略

```
┌─────────────────────────────────────────────────────────────────┐
│ 时间线                                                           │
├──────────┬──────────────────────────────────────────────────────┤
│ 立即开始 │ 方案B: 加载 OLD 深度时乘 0.748 校正尺度               │
│          │ + absrel_prior = 0.12 → 可直接开始训练                 │
│ 同时后台 │ 用 NESTED MV+pose 生成新深度图 data_2d_depth_da3_v3    │
│ 预处理完 │ 切换到 v3 深度 + absrel_prior = 0.15 重新/继续训练      │
└──────────┴──────────────────────────────────────────────────────┘
```

方案B 的修改位置：`kitti_360_dataset.py` 的 `load_depth` 中：

```python
depth = depth / self.depth_da3_scale     # uint16 → float (÷256)
depth = depth * 0.748                    # 校正 METRIC-LARGE 的系统性偏差
```

---

## 五、设置汇总表（便于复现与对比）

| 维度 | 论文单目 | 论文公制 | 论文多视图（训练） | 论文 NVS 评估 | KITTI-360 单图（OLD/v2） | KITTI-360 多视图 v3 |
|------|----------|----------|--------------------|---------------|---------------------------|----------------------|
| **输入帧数** | 1 | 1 | 2–18（随机） | 12 | 1 | 多帧（如 7） |
| **分辨率** | 随 benchmark | base 504，多宽高比 | 504×504 等 | 270×480 | 192×640 等 | 同左 |
| **位姿** | 无 | 无 | 可选 0.2 概率 | GT pose | 无 | GT 位姿+内参 |
| **尺度** | 相对 | 公制（canonical+GT） | 统一 scale | - | 相对/逐帧不稳 | 公制（Umeyama） |
| **主要指标** | δ1, AbsRel, SqRel | δ1, AbsRel | F1, Auc3/30, CD | PSNR/SSIM/LPIPS | abs_rel, rmse, δ1, median_ratio | 同上 |

---

## 六、结论与使用建议

1. **论文**：DA3 的深度指标与"单目/多视图"设置在 **KITTI（标准）、NYU、ETH3D、DIODE、SUN-RGBD** 等上报告；**未包含 KITTI-360**。  
2. **DA3 可直接输出 metric depth**：Nested 模型的 metric Large 分支从训练数据中学到绝对尺度，**不需要 LiDAR 或外部深度 GT**；提供位姿做 Umeyama 对齐是可选的精度提升手段。
3. **DA3METRIC-LARGE vs NESTED**：两者深度结构质量几乎相同（对齐后 abs_rel ~0.11）；METRIC-LARGE 更快（0.08 vs 0.16 s/img）且使用 Apache 2.0 许可；NESTED 提供更好的绝对尺度（ratio 0.917 vs 0.748）和位姿估计能力。
4. **KITTI-360 上**：  
   - **单图**：相对深度可用（ALIGNED abs_rel~0.11, δ1~0.88），绝对尺度偏大 ~34%。  
   - **多视图 + GT 位姿**：RAW abs_rel 0.186, ratio 0.917，绝对尺度和时间一致性最佳。  
   - **鱼眼重采样**：MV+pose 的 abs_rel 0.181（RAW），可用于替代当前训练中的全零深度。
5. **训练建议**：当前 OLD 深度（ratio 0.748）配合 absrel_prior=0.10 存在采样窗口不覆盖真实表面的风险；建议校正尺度或使用 MV+pose 深度。

---

## 参考文献

- DA3 论文：[*Depth Anything 3: Recovering the Visual Space from Any Views*](https://arxiv.org/pdf/2511.10647) (arXiv:2511.10647).  
- DA3METRIC-LARGE：[HuggingFace Model Card](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
- DA3NESTED-GIANT-LARGE-1.1：[HuggingFace Model Card](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)
- 本地评估脚本：`test_fisheye_da3_depth.py`、`test_metric_vs_nested.py`、`test_depth_vs_lidar.py`
- 本地文档：《各目录深度图准确度对比分析》《更准确深度图获取方式与效果对比》《ViPOcc深度预测机制与实现细节》等。
