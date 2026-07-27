# data_2d_depth_da3_v3 质量检查与训练策略
(Depth map inconsistency analysis)
> 基于 v3 生成结果检查与既有文档（《DA3深度置信度分析与采样设计》《DA3在KITTI360上的表现与设置分析》《训练中鱼眼图像使用与无深度影响分析》）整理。

---

## 一、生成质量检查

### 1.1 目录结构与完整性

| 项目 | 状态 |
|------|------|
| 路径 | `/mnt/remote_data/lmh_data/KITTI360/data_2d_depth_da3_v3` |
| 结构 | `<seq>/image_0X/data_rect/` + `data_rect_conf/` |
| 相机 | image_00, 01（透视）, 02, 03（鱼眼）全部生成 |
| 格式 | 深度 uint16 = depth_m × 256；置信度 uint16 = conf × 10000 |

**帧数统计**（抽样检查）：

| 序列 | image_00 | image_01 | image_02 | image_03 |
|------|----------|----------|----------|----------|
| drive_0000 | 11518 | 11518 | 11518 | 11518 |
| drive_0002 | 14607 | 14607 | 14385 | 14385 |
| drive_0004 | 11587 | 11587 | 11587 | 11587 |
| drive_0005 | 6743 | 6743 | 6743 | 6743 |

drive_0002 的 02/03 为 14385 帧（源数据本身与 00/01 不同），非丢帧。

### 1.2 深度图质量（基于既有评估与文档）

**预期指标**（MV+pose 多视图推理，与 LiDAR 对比）：

| 相机类型 | abs_rel (RAW) | median_ratio | abs_rel (ALIGNED) |
|---------|---------------|--------------|-------------------|
| 透视 (00/01) | ~0.12 | ~0.92–1.0 | ~0.11 |
| 鱼眼 (02/03) | ~0.18 | ~1.09 | ~0.14 |

**验证方式**：运行 `dino-da3-adaptive-sampling1/scripts/eval_v3_vs_lidar.py` 获取实际数值。

### 1.3 分辨率与加载

| 项目 | 值 |
|------|-----|
| v3 深度图分辨率 | **140 × 504**（DA3 process_res=504 输出） |
| 训练 target_image_size | 通常 192×640 |
| 数据集处理 | `kitti_360_dataset.py` 中 `load_depth` 会对 `depth.shape != target_size` 做 `cv2.resize`，**无需额外预处理** |

### 1.4 置信度

- 约 90% 帧存在 conf clipping（>6.5535 被截断），对低置信阈值计算影响有限。
- 离线模式下当前 **未加载 confidence**，需按后文策略接入。

---

## 二、当前训练管线中的问题

### 2.1 鱼眼深度未加载

`kitti_360_dataset.py` 第 968–971 行：

```python
# Depth for fisheye is not directly supported; use zeros as placeholder
depths_f_left = [np.zeros((1, *fisheye_size), dtype=np.float32) for _ in img_ids_fish]
depths_f_right = [np.zeros((1, *fisheye_size), dtype=np.float32) for _ in img_ids_fish]
```

**问题**：v3 已生成 image_02/03 深度，但数据集仍返回全零，导致鱼眼射线继续走 nerf 的 zero-depth fallback（全局采样）。

**结论**：需修改数据集，从 v3 加载鱼眼深度。

### 2.2 深度目录未指向 v3

配置中 `depth_da3_path` 默认指向 `data_2d_depth_da3`，需改为 v3 路径。

### 2.3 置信度未使用

离线模式下 `low_conf_mask = None`，confidence PNG 已生成但未加载，低置信回退未生效。

### 2.4 absrel_prior 未针对 v3 调整

v3 透视 abs_rel ~0.12，建议 `absrel_prior = 0.15`；鱼眼 ~0.18，建议 0.18–0.20 或依赖 low_conf 回退。

---

## 三、训练策略（按优先级）

### P1：切换深度目录并加载鱼眼深度（必做）

**目标**：使用 v3 深度，并让鱼眼参与自适应采样。

| 改动 | 文件 | 内容 |
|------|------|------|
| 1 | config (如 exp_kitti_360.yaml) | `depth_da3_path: /mnt/remote_data/lmh_data/KITTI360/data_2d_depth_da3_v3` |
| 2 | `kitti_360_dataset.py` | 为 image_02/03 增加 `load_depth` 或等效加载逻辑，从 `depth_da3_path/seq/image_02/data_rect/` 等读取；若无文件则退回 zeros |

**实现要点**：  
- 透视沿用现有 `load_depth(seq, img_id, is_right)`。  
- 鱼眼需新增 `load_depth_fisheye(seq, img_id, cam)`，cam 为 `"image_02"` 或 `"image_03"`，路径 `{depth_da3_path}/{seq}/{cam}/data_rect/{img_id:010d}.png`，读取后 resize 到 `target_image_size`。

### P2：设置 absrel_prior（必做）

| 配置项 | 推荐值 | 依据 |
|--------|--------|------|
| absrel_prior | **0.15** | v3 透视 abs_rel ~0.12，略放大以覆盖误差 |
| （若区分鱼眼） | 鱼眼 0.18–0.20 | 鱼眼 abs_rel ~0.18 |

若不区分视图，统一用 **0.15** 即可。

### P3：接入置信度（高优先级）

**目标**：利用 v3 的 `data_rect_conf` 做低置信回退，减少错误先验对采样的影响。

| 改动 | 文件 | 内容 |
|------|------|------|
| 1 | `kitti_360_dataset.py` | `load_depth` 同时加载 `data_rect_conf/*.png`，返回 `(depth, conf)` 或扩展 data 结构 |
| 2 | `trainer.py` | 离线模式下构建 `conf` tensor，传入 sampler 和 renderer |
| 3 | `nerf.py` | 已有 `low_conf_mask` 支持；需在 trainer 中根据 conf 百分位（如 20%）生成 `low_conf_mask` 并传入 |

**置信度 scale**：`conf_float = uint16_value / 10000.0`。

### P4：鱼眼 zero-depth fallback 保留（已实现）

`nerf.py` 中 `zero_depth_mask = (da3_depths <= 1e-3)` 已实现，鱼眼在加载 v3 深度后，仅对真正无效（0 或缺失）的像素回退全局采样，逻辑正确。

### P5：可选——Warmup、batch、梯度裁剪

参考《更稳定训练策略分析》：
- **Warmup**：`warmup_steps: 5000–10000`
- **batch/ray_batch**：显存允许时 `batch_size=8`、`ray_batch_size=8192`
- **梯度裁剪**：`clip_grad_norm_(model.parameters(), max_norm=1.0)`

---

## 四、实施顺序建议

1. **立即**：P1（切换 v3 路径 + 加载鱼眼深度）+ P2（absrel_prior=0.15）
2. **短期**：P3（加载并使用 confidence 做 low_conf_mask）
3. **视训练曲线**：P5（Warmup、batch、梯度裁剪）

---

## 五、验证清单

- [ ] 配置中 `depth_da3_path` 指向 v3
- [ ] 数据集能加载 image_02/03 的 v3 深度
- [ ] `absrel_prior` 设为 0.15
- [ ] （可选）confidence 加载并传入 low_conf_mask
- [ ] 运行 `eval_v3_vs_lidar.py` 确认 v3 与 LiDAR 指标符合预期
