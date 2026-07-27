# Loss 震荡/平台现象分析

> 基于 `exp_kitti_360` 训练管线及多轮连续训练日志（out/kitti_360）的分析。
> **更新**：经与原始 S4C（[GitHub](https://github.com/ahayler/s4c) + [论文](https://arxiv.org/pdf/2310.07522)）对比验证，标注了各项原因是否为真正的问题。

---

## 一、现象描述

从 TensorBoard 与连续多次 resume 训练记录可见：

- **Loss 并非严格"震荡"**，而是**在高位平台区间内波动**，无明显长期下降趋势。
- 多轮训练（step 1 → 1848 → 3500 → 5294）中，`loss` 与 `loss_rgb_coarse` 始终在 **0.12～0.18** 区间内起伏。
- `loss_invalid_ratio` 在 **0.02～0.11** 间波动，说明每步被判定为 invalid 的射线比例在变，带来 step 间方差。
- 学习率在 5000 step 时仍为 `1e-4`（StepLR 在 120000 step 才衰减），因此**当前现象不是由 LR 过早衰减导致**。

---

## 二、原因分析

### 2.1 每步随机性导致的方差（表象）— ℹ️ 非问题，与原始 S4C 一致

以下因素会在**不改变期望梯度方向**的前提下，增大单步 loss 的波动：

| 因素 | 说明 |
|------|------|
| **Patch 随机采样** | `PatchRaySampler` 每步从 `ids_loss` 视图中随机采 64 个 8×8 patch，不同 step 覆盖的像素集合不同。 |
| **帧分配随机** | `kitti360-mono` 下 `frame_perm` 随机，`ids_loss` / `ids_render` 每步不同，loss 视图组合变化。 |
| **颜色增强** | `color_aug: true` 时每步对图像做随机变换，同一场景不同 step 的 RGB 目标不同。 |
| **invalid 比例波动** | `invalid_policy: weight_guided` 下，超过 90% 权重落在无效区域的射线被忽略，`loss_invalid_ratio` 随采样变化而波动。 |

> **与原始 S4C 对比**：原始 S4C 使用完全相同的 `PatchRaySampler`、`kitti360-mono` 帧分配、`color_aug: true`、`invalid_policy: weight_guided`。原始 S4C 的 loss 曲线也存在同样的抖动，这是 NeRF patch 采样训练的**正常现象**，不是 loss 不降的原因。

### 2.2 鱼眼视图 DA3 深度退化 — ✅ 真正的问题

- 鱼眼相机（image_02/03）的 DA3 离线深度为**全零**（DA3 未处理鱼眼）。
- 在 `sample_adaptive()` 中，深度被 `clamp` 到 `z_near=3`，于是：
  - σ = max(0.5, 0.10×3) = 0.5
  - Surface 采样范围被 clamp 到 **[3, 4] m**
  - **32 个 surface 采样点全部挤在 3～4 m**，32 个 global 点才覆盖 [3, 80] m。
- 结果：来自鱼眼视图的射线，**约一半采样点浪费在极窄近处**，深度分辨率极差，梯度质量差，且与透视视图的采样分布不一致，拖累整体收敛。

> **与原始 S4C 对比**：原始 S4C 使用**全程均匀/逆深度采样**（64 个 coarse 样本均匀分布在 [z_near, z_far]），所有视图一视同仁，鱼眼和透视图的采样质量相同。此退化是自适应采样引入的新问题。

### 2.3 置信度过滤未生效 — ✅ 真正的问题

- 配置写了 `depth_mode: "online"`，但 `data_util.py` 未向 `Kitti360Dataset` 传递 `depth_mode`，实际为 **offline**。
- `model_conf` 中未设置 `online_depth_generation`，Trainer 层 DA3 在线推理不触发，**始终使用离线 DA3 深度**。
- 离线分支**无置信度输出**，`low_conf_mask=None`，设计上"低置信度区域退化为均匀采样"的逻辑**完全未执行**。
- 因此所有射线（包括深度不可靠或鱼眼零深度）都走同一套自适应采样，无法对不可靠深度做降权或回退。

> **与原始 S4C 对比**：原始 S4C 没有 DA3 深度和置信度的概念，不存在此问题。这是新增功能的实现缺陷。

### 2.4 无学习率 Warmup — ℹ️ 非问题，与原始 S4C 一致

- `default.yaml` 中配置了 `warmup_steps: 10000`，但 `scheduler.py` 的 `make_scheduler` **不读取**该参数，仅支持 `fix` 与 `step`。
- 训练从 **step 1 起即为 lr=1e-4**，无 warmup。

> **与原始 S4C 对比**：原始 S4C 的 `default.yaml` 中也写了 `warmup_steps: 10000`，但同样的 `scheduler.py` 也只支持 `fix` 和 `step`，**原始 S4C 同样没有 warmup**，且训练成功收敛。因此无 warmup 不是 loss 不降的原因。

### 2.5 MLP 与 epoch 覆盖说明 — ℹ️ 非问题，与原始 S4C 一致

- **mlp_coarse**：`n_blocks=0, d_hidden=64` 对应 **1 个隐藏层**（103→64→1），与原始 S4C/BTS 在 KITTI-360 上一致，**不是** loss 不降的主因。
- **Epoch 覆盖**：数据集约 96901 样本，`batch_size=16` 时每 epoch 约 6056 iters；5000 step 约 82.5% 一个 epoch，属正常范围，不单独解释平台现象。

### 2.6 未加载预训练权重 — ✅ 最关键的遗漏原因

原始 S4C 的训练**并非从头开始**，而是从 BTS（BehindTheScenes）预训练权重出发微调：

- 原始 S4C 仓库 README 明确说明：*"For faster convergence and slightly better results, we use the pretrained model from BehindTheScenes as a backbone from which we start our training."*
- 原始 `exp_kitti_360.yaml` 中设置 `use_backbone: true`
- `base_trainer.py` 中的实现：从 `out/kitti_360/pretrained/training*.pt` 加载 BTS 完整模型权重（encoder + density MLP + decoder），以 `strict=False` 跳过不匹配的键（如新增的 segmentation MLP）

**而当前 s4c-adaptive-sampling1 的设置**：

```yaml
use_backbone: false   # ← 未加载任何预训练权重！
```

这意味着除了 ResNet-50 的 ImageNet 权重外，density MLP、decoder 等全部随机初始化，训练难度远大于原始 S4C 的微调策略。

> **影响**：这很可能是 loss 长期在 0.12~0.18 高位平台不下降的**最主要原因**。原始 S4C 的训练实质是在已学好几何的 BTS 权重基础上加入 segmentation，而非从零学习场景几何。

**建议**：设置 `use_backbone: true` 或加载 S4C 预训练权重。本项目中已有预训练文件：
`out/kitti_360/pretrained/training_checkpoint_176000.pt`

加载 S4C 完整预训练权重（encoder + density MLP + segmentation MLP 全部已训练好）效果更佳，因为 s4c-adaptive-sampling1 的模型架构与 S4C 完全相同（自适应采样只改变采样策略，不增加可学习参数）。

---

## 三、小结

| 类型 | 原因 | 对 loss 曲线的影响 | 是否为真正问题？ |
|------|------|---------------------|-----------------|
| 表象 | Patch/帧/颜色随机、invalid 比例波动 | 单步方差大、曲线抖动 | ℹ️ 否，原始 S4C 相同 |
| 结构 | 鱼眼 DA3 深度为零，表面采样集中在 3~4 m | 部分射线梯度质量差，拖累收敛 | ✅ 是，自适应采样引入 |
| 配置 | 置信度未接入，低置信度回退未生效 | 不可靠深度仍参与自适应采样 | ✅ 是，新功能实现缺陷 |
| 训练策略 | 无 warmup，LR 一步到位 | 早期可能不稳定 | ℹ️ 否，原始 S4C 也无 warmup |
| **初始化** | **未加载 BTS/S4C 预训练权重（`use_backbone: false`）** | **模型从零学习几何，收敛极慢** | **✅ 最关键原因** |

**结论**：Loss 的"震荡/平台"的**最主要原因**是未加载预训练权重（原始 S4C 从 BTS 权重出发微调，而非从头训练）。其次是鱼眼视图下 DA3 自适应采样的退化和置信度过滤缺失。每步随机性和无 warmup 与原始 S4C 行为一致，不是导致 loss 不降的原因。
