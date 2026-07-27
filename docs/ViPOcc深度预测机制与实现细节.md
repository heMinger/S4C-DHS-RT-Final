# ViPOcc 深度预测机制与实现细节

> 基于官方仓库 [fengyi233/ViPOcc](https://github.com/fengyi233/ViPOcc) 与论文 [ViPOcc (AAAI-25)](https://arxiv.org/pdf/2412.11210) 的代码与实现分析总结。

---

## 1. 概述

ViPOcc 采用**双深度预测架构**：

- **Rendered depth**：NeRF 体渲染得到的深度（alpha-compositing）
- **Predicted depth**：独立的 metric depth 分支，利用 VFM 伪深度 + 逆深度对齐得到

两者通过 **depth reconstruction consistency loss** 与 **temporal alignment loss** 互相约束。

---

## 2. 伪深度生成（离线预处理）

- **模型**：Depth Anything V2（ViT-Large，在 vKITTI 上训练的 metric 版本）
- **输入**：KITTI-360 透视相机图像（image_00 / image_01）
- **输出**：metric 深度图，最大深度 80m，保存为 `.npy` 与 `.png`

脚本：`data_preprocess/pseudo_depth/gen_pseudo_depth_kitti360.py`：

```python
depth_anything = DepthAnythingV2(encoder='vitl', features=256,
                                  out_channels=[256, 512, 1024, 1024], max_depth=80)
depth = depth_anything.infer_image(raw_image)
```

训练时以 `pseudo_depth` 形式加载，**不参与梯度回传**。

---

## 3. 特征编码器（NeRF 分支）

- **结构**：Monodepth2（ResNet50 encoder + 修改版 decoder）
- **作用**：从关键帧提取 64 维像素级特征，供 NeRF 密度预测使用
- **输入**：单关键帧 `(B, 3, 192, 640)`
- **输出**：特征图 `(B, 64, 192, 640)`，**不直接输出深度**

文件：`models/common/backbones/monodepth2.py` → `Monodepth2`。

---

## 4. 深度预测分支（核心）

入口：`BTSWrapper.predict_depth(images, pseudo_depth)`（`trainer.py`）。

支持三种模式（`rid_ablation`）：

| 模式 | 说明 |
|------|------|
| **rid**（默认） | RID 网络预测残差逆深度 + IDA 与 pseudo_depth 融合 |
| **res_depth** | ResidualDepth 预测残差深度，直接与 pseudo_depth 相加 |
| **depth** | Monodepth2 直接预测深度，不用 pseudo_depth |

### 4.1 RID 网络（`res_inv_depth.py`）

- **结构**：ResNet18 encoder + U-Net 风格 decoder，输出 1 通道
- **输出**：残差逆深度图（无 sigmoid，可为任意实数）
- **输入**：图像 `[0,1]`（代码中 `x * 0.5 + 0.5`）

与 NeRF 的 ResNet50 编码器**完全独立**。

### 4.2 Inverse Depth Alignment（IDA）模块

文件：`utils/modules.py` → `DepthRefinement`。

```python
def forward(self, pseudo_depth, res_inv_depth):
    pseudo_depth = pseudo_depth.clamp(min=0.1, max=80)
    res_inv_depth /= 10
    inv_pseudo_depth = 1.0 / pseudo_depth
    depth = 1.0 / (inv_pseudo_depth + res_inv_depth + 1e-8)
    return depth
```

**数学形式**：

\[
d_{pred} = \frac{1}{\frac{1}{d_{pseudo}} + \frac{\Delta_{inv}}{10}}
\]

**设计要点**：

- VFM 提供相对/序数深度，metric 尺度有偏差；在**逆深度空间**做残差可保留局部微分性质。
- 残差除以 10 使网络只学习小幅修正，不破坏 VFM 先验。
- 近处逆深度大、远处小，修正对不同距离的影响更均衡。

---

## 5. NeRF 渲染深度

- **采样**：每射线 64 点，**线性视差**（`lindisp=True`），深度范围 3–80m。
- **流程**：3D 点 → 投影到 2D 特征图采特征 → 位置编码 → MLP 预测 σ → alpha-compositing。
- **渲染深度**：`depth = Σ (weights * z_samp)`。

文件：`models/common/render/nerf.py` → `NeRFRenderer.sample_coarse` / `composite` / `composite_depth`。

---

## 6. 深度分支的监督方向（澄清）

- ViPOcc 学的是一套**在 VFM 伪深度基础上的 refinement 网络**（RID + IDA），得到 **predicted_depth**。
- **predicted_depth 不是“直接作为监督信号”去监督别的模块**，而是**被监督的对象**：  
  深度分支通过以下 loss 被训练，使 predicted_depth 既符合 NeRF 几何（rendered_depth），又满足时序一致性， optionally 符合 GT 深度。
- 总结：**“很好的深度”是深度分支的输出目标**，监督来自 (1) NeRF 渲染深度、(2) 时序 warp 一致性、(3) 可选的 GT 深度。

---

## 7. 深度相关 Loss 公式

| Loss | 配置键 | 默认权重 | 作用 |
|------|--------|----------|------|
| 深度重建一致性 | `lambda_depth_recon` | 0.005 | 对齐 predicted depth 与 rendered depth |
| 时序对齐 | `lambda_temporal_alignment` | 0.01 | 用 predicted depth warp 参考帧，光度 + 几何一致性 |
| 深度监督 | `lambda_depth_supervision` | 0 | 与 GT 深度 L1，默认关闭 |

### 7.1 深度监督（可选，exp_kitti_360 中 = 0）

仅在 `lambda_depth_supervision > 0` 时启用，且需提供 GT 深度 `D_gt`（如 LiDAR 投影）。在有效像素集合 Ω（`depth_gt > 0`）上：

$$
\mathcal{L}_{\text{depth\_super}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \big| D_{\text{gt}}(p) - D_{\text{pred}}(p) \big|
$$

代码：`loss_depth_super = (depth_gt - pred_depth).abs().mean()`（mask 后 concat 两帧一起算）。

### 7.2 深度重建一致性（核心：让预测深度对齐 NeRF）

在随机 patch 上，对每条射线得到 **predicted_depth** 对应的深度值 `D_pred`，以及 NeRF 体渲染的 **rendered_depth**（经射线方向 scaling 校正）`D_rend`。配置 `depth_recon_version` 控制形式：

- **Version 1（默认）**：
  $$\mathcal{L}_{\text{depth\_recon}} = \frac{1}{N} \sum_{i} \big| D_{\text{pred},i} - D_{\text{rend},i} \big|$$

- **Version 2（远距离加权）**：先对 `D_pred` 做 min-max 归一化得 \(\hat{D}_{\text{pred}}\)，权重 \(w = 1 - \hat{D}_{\text{pred}}\)，则
  $$\mathcal{L}_{\text{depth\_recon}} = \frac{1}{N} \sum_{i} w_i \cdot \big| D_{\text{pred},i} - D_{\text{rend},i} \big|$$

- **Version 3（分布对齐）**：对 pred 与 rend 分别 min-max 归一化后做 L1：
  $$\mathcal{L}_{\text{depth\_recon}} = \frac{1}{N} \sum_{i} \big| \hat{D}_{\text{pred},i} - \hat{D}_{\text{rend},i} \big|$$

代码中均为 `F.l1_loss` 或等价的 `.abs().mean()`。

### 7.3 时序对齐（光度 + 几何）

用 **predicted_depth** 做帧间 warp。记 target 帧为 \(t\)、reference 帧为 \(r\)，\(D_t, D_r\) 为两帧的 predicted depth，\(I_r\) 为 reference 图像，\(T_{t \to r}\) 为 target 到 reference 的位姿（3×4），\(K\) 为内参。先由 \(D_t\) 和 \(K\) 反投影得 3D 点，用 \(T_{t \to r}\) 和 \(K\) 投影到 reference 图像，得到：
- **warped 图像** \(\hat{I}_{r \to t}\)
- **warped 的 reference 深度** `projected_depth`
- **由几何算出的深度** `computed_depth`（投影后的 z）
- **有效 mask** `valid_mask`（投影在图像内）

则：

- **光度项**（在 valid 上、用几何一致性加权）：
  $$\mathcal{L}_{\text{photo}} = \frac{1}{\sum m} \sum_{p} m(p) \cdot \big| I_t(p) - \hat{I}_{r \to t}(p) \big| \cdot \big(1 - \text{diff\_depth}(p)\big)$$
  其中 \(\text{diff\_depth} = \frac{|D_{\text{computed}} - D_{\text{projected}}|}{D_{\text{computed}} + D_{\text{projected}}}}\)，用于降权不一致区域。

- **几何一致性项**：
  $$\mathcal{L}_{\text{geom}} = \frac{1}{\sum m} \sum_{p} m(p) \cdot \text{diff\_depth}(p)$$

**总时序 loss**：\(\mathcal{L}_{\text{temp}} = \mathcal{L}_{\text{photo}} + \mathcal{L}_{\text{geom}}\)，再乘 `lambda_temporal_alignment`。

实现：`utils/warp.py`（`depth_to_3d` + 投影 + `grid_sample`），`loss.py` 中 `mean_on_mask`。

---

### 7.4 深度重建一致性（实现细节）

- 使用 `DepthRaySampler` 在 predicted depth 上随机采 patch（约 2048 条射线）。
- 对这些射线做 NeRF 渲染得到 `rendered_depth`，并按射线方向做 scaling 校正（与相机 z 的夹角）。
- 上述 L_depth_recon 在 patch 内逐像素/逐射线计算，再 mean。

---

## 8. 训练数据流简图

```
离线：
  Image → [Depth Anything V2] → pseudo_depth
  Image → [Grounded-SAM 等]   → sampling anchors（SNOG 用）

训练时：
  NeRF 分支：Image → ResNet50+Decoder → 64ch → MLP → σ → volume render → rendered_rgb, rendered_depth
  深度分支：Image → RID(ResNet18) → res_inv_depth → IDA(pseudo_depth, res_inv_depth) → predicted_depth

Loss = L_rgb + 0.005 * L_depth_recon + 0.01 * L_temporal_align
```

---

## 9. 关键实现参数（exp_kitti_360）

| 项 | 值 |
|----|-----|
| 伪深度 | Depth Anything V2 ViT-L, metric, max_depth=80 |
| RID | ResNet18 + U-Net decoder, 1ch |
| NeRF 编码器 | ResNet50, 64ch |
| 深度范围 | z_near=3, z_far=80 |
| 射线采样 | patch 8×8，64 patch → 4096 rays |
| 深度采样 | 额外 2048 rays（depth recon） |
| 分辨率 | 192×640 |
| 帧采样 | kitti360-mono（随机一半帧 loss / 一半 render） |

---

## 10. 与本文项目的对比要点

- 本地项目（s4c-adaptive-sampling1）使用 **DA3 深度** 做自适应射线采样与监督，**没有** ViPOcc 的 RID+IDA 逆深度对齐分支。
- ViPOcc 的 **predicted depth** 由「伪深度 + 残差逆深度」得到，并和 **rendered depth** 通过 loss 对齐，形成「VFM 先验 + NeRF 几何」的双分支深度预测。

---

## 11. 参考

- 代码：<https://github.com/fengyi233/ViPOcc>
- 论文：ViPOcc: Leveraging Visual Priors from Vision Foundation Models for Single-View 3D Occupancy Prediction (AAAI-25 Oral)

---

## 12. 补充分析：监督逻辑、时序 Warp 与为何需要 RID

以下为基于代码与训练逻辑的归纳与合理推断，用于澄清常见疑问。

### 12.1 为什么 NeRF 的 rendered_depth 可以监督 predicted_depth？

**疑问**：rendered_depth 来自预测的 σ，σ 本身也是学出来的，为何能当“真值”去监督 predicted_depth？

**要点**：

1. **NeRF 的 σ 有独立的强监督**：主 loss 是**多视图 RGB 重建**（L_rgb = L1+SSIM(rendered_rgb, rgb_gt)），不是深度。当多视角渲染的 RGB 与 GT 一致时，隐式要求 3D 几何（σ 场）正确，否则颜色对不齐。因此 **rendered_depth 的“权威性”来自多视图光度一致性**，而不是“用猜测监督猜测”。
2. **监督是双向的**：L_depth_recon 的梯度也会反传到 NeRF。predicted_depth 携带 VFM 先验，通过该 loss 把先验反哺给 NeRF，帮助几何更快收敛。
3. **总结**：rendered_depth 是由**多视图 RGB 约束**隐式得到的几何量，用它监督 predicted_depth 等价于用“多视图一致的几何”去约束“单视图深度分支”；同时深度分支的先验通过同一 loss 反传，改善 NeRF。

### 12.2 时序对齐：输入、Warp 含义与具体操作

**输入（来自 `loss.py`）**：

- `depth_t`, `depth_r`：两帧的 **predicted_depth**（深度分支输出），shape [B,1,H,W]
- `img_t`, `img_r`：target 与 reference 的 RGB 图像
- `pose_t2r`：target → reference 的相对位姿 [B,3,4]
- `K`：内参（经代码中归一化与缩放至像素坐标）

**Warp 的含义**：用**一帧的深度 + 帧间位姿**，把另一帧的像素“搬”到当前帧的像素网格上，得到**合成图**。若深度正确，合成图应与当前帧观测一致，从而提供监督。

**具体操作（对应 `utils/warp.py`）**：

1. **Target 像素 → 3D**：对 target 每像素 \(p=(u,v,1)^T\)，用 \(D_t\) 与 \(K\) 反投影：\(P_{cam} = D_t(u,v) \cdot K^{-1} p\)（代码：`depth_to_3d(depth_t, K)`）。
2. **3D → Reference 像素**：用 \(T_{t \to r}\) 与 \(K\) 投影到 reference 得到 \((u', v')\)（代码：`KT @ p_cam`，再除以 z）。
3. **在 reference 上采样**：用 \((u', v')\) 对 `img_r` 做 `grid_sample`，得到 **warped 图像** \(\hat{I}_{r \to t}\)。
4. **监督**：若 \(D_t\) 正确，则 \(\hat{I}_{r \to t}(u,v) \approx I_t(u,v)\)。同时用 `projected_depth` 与 `computed_depth` 算几何一致性，对不一致区域降权光度项。

**图示**：Target 像素 + depth_t → 3D → 投影到 reference → 在 reference 上采颜色 → 与 target 图像比较；差异大则说明 depth_t 有误，loss 推动 RID 更新。

### 12.3 既然 NeRF 能通过多视图 RGB 得到 rendered_depth，为何还要 RID？

**原因概览**：

1. **训练时 NeRF 只渲染极稀疏射线**：每步仅 **4096 条射线**（64 个 8×8 patch），全图 192×640 = **122,880 像素**，覆盖率约 **3.3%**。没有“全图”的 rendered_depth 可用。
2. **时序对齐必须用密集深度图**：warp 需要对**每个像素**做反投影与投影，必须有一张完整的 [B,1,H,W] 深度图。只有 RID 一次前向得到全图深度，才能算 L_temporal。
3. **推理效率**：验证时若用 NeRF 出深度，需对 122,880 条射线做体渲染（每条 64 点 × MLP），极慢；有深度分支时用 RID 一次前向即可，速度快两个数量级（代码中 `use_depth_branch` 分支即为此设计）。

**对比小结**：

| | NeRF rendered_depth | RID predicted_depth |
|---|---------------------|----------------------|
| 训练时覆盖率 | 约 3.3% 像素 | 100% 全图 |
| 单张深度图成本 | 全图需 12 万+ 射线 × 64 采样 | 一次前向 |
| 几何来源 | 多视图 RGB 一致性 | VFM 先验 + 逆深度残差 |
| 主要用途 | RGB 重建、与 RID 做 recon loss | 时序 warp、推理时快速出深度 |

**结论**：两者互补。NeRF 提供多视图几何并透过 L_depth_recon 约束 RID；RID 提供**密集、快速**的深度图，使时序对齐可行，并把 VFM 先验反哺给 NeRF。缺 RID 则既做不了逐像素时序 loss，推理时也无法高效输出深度图。
