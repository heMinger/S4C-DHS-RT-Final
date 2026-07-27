# DA3 深度图 + 置信图：生成检查、机制分析与采样改进设计

---

## 一、生成结果检查（data_2d_depth_da3_v3）

### 1.1 生成命令

```bash
python generate_da3_depth_multiview.py \
  --data_path /data/lmh_data/KITTI360 \
  --output_root /data/lmh_data/KITTI360/data_2d_depth_da3_v3 \
  --cameras image_00 image_01 image_02 image_03 \
  --window_size 7 \
  --process_res 504 \
  --compare_lidar \
  --skip_existing
```

脚本：`~/generate_da3_depth_multiview.py`，远程服务器 `/data/lmh_data` 对应本地 `/mnt/remote_data/lmh_data`。

### 1.2 输出结构

```
data_2d_depth_da3_v3/
  <seq>/image_0X/data_rect/<frame>.png          # 深度 (uint16, depth_m × 256)
  <seq>/image_0X/data_rect_conf/<frame>.png     # 置信度 (uint16, conf × 10000)
```

### 1.3 帧数完整性

已完成的 7 个序列 + drive_0009 前 3 个相机，**深度帧数 = 置信帧数 = 源图像帧数**，100% 匹配：

| 序列 | image_00 | image_01 | image_02 | image_03 |
|------|----------|----------|----------|----------|
| drive_0000 | 11518 ✅ | 11518 ✅ | 11518 ✅ | 11518 ✅ |
| drive_0002 | 14607 ✅ | 14607 ✅ | 14385 ✅ | 14385 ✅ |
| drive_0003 | 1031 ✅ | 1031 ✅ | 1031 ✅ | 1031 ✅ |
| drive_0004 | 11587 ✅ | 11587 ✅ | 11587 ✅ | 11587 ✅ |
| drive_0005 | 6743 ✅ | 6743 ✅ | 6743 ✅ | 6743 ✅ |
| drive_0006 | 9699 ✅ | 9699 ✅ | 9699 ✅ | 9699 ✅ |
| drive_0007 | 3396 ✅ | 3396 ✅ | 3396 ✅ | 3396 ✅ |
| drive_0009 | 14056 ✅ | 14056 ✅ | 14056 ✅ | 生成中 |
| drive_0010 | — | — | — | — (排队中) |

drive_0002 的 image_02/03 源数据本身就是 14385 帧（与 image_00/01 的 14607 不同），不是丢帧。

### 1.4 深度图质量

**采样检查**（每 500 帧取一个）：

| 序列/相机 | 最小深度(m) | 最大深度(m) | 平均深度(m) |
|-----------|-----------|-----------|-----------|
| drive_0000/image_00 | 0.9~4.2 | 16.7~101.3 | 8.1~33.2 |
| drive_0000/image_01 | 0.9~4.0 | 18.3~101.3 | 6.7~24.7 |
| drive_0000/image_02 (鱼眼) | 0.7~10.3 | 7.1~108.1 | 3.8~42.7 |
| drive_0000/image_03 (鱼眼) | 0.9~4.7 | 7.3~82.2 | 4.0~32.3 |
| drive_0004/image_02 (鱼眼) | 1.9~9.4 | 9.2~**147.2** | 5.1~**88.0** |

- 全量检查 drive_0000/image_00 前 1000 帧：**0 个退化帧**（无全零、无常值、无读取失败）
- zero_pct = 0.0%，所有帧都有有效深度
- 透视相机（image_00/01）深度范围合理
- 鱼眼相机（image_02/03）偶尔出现偏高深度（如 drive_0004/image_02 最大 147m、均值 88m），与鱼眼→针孔重采样边缘区域有关

### 1.5 分辨率不匹配

| 项目 | 分辨率 |
|------|--------|
| 源图像 | **376 × 1408** |
| 深度图输出 | **140 × 504** |

深度图保存在 DA3 的推理分辨率（`process_res=504`），而非原始图像分辨率。脚本中 `save_depth_conf` 直接保存了模型输出，没有 resize 回原始尺寸的步骤。下游加载时需自行 resize。

### 1.6 置信度 clipping

置信度以 `conf_scale=10000` 保存为 uint16，最大可表示 `65535/10000 = 6.5535`。

| 检查结果 | 数据 |
|----------|------|
| 有 clipping 的帧比例 | ~90% 的帧受影响 |
| 每帧平均被裁剪像素数 | ~22000 / 70560 总像素 ≈ 31% |
| conf 值域 | min=1.0, max=6.5535（上限被裁剪） |

约 1/3 像素的高置信度被截断。但 clipping 影响的是高置信端，不影响低置信阈值计算，对下游百分位过滤逻辑影响有限。

---

## 二、置信度机制分析

### 2.1 clipping 的具体逻辑

保存阶段（`generate_da3_depth_multiview.py: save_depth_conf`）：

```python
conf_uint16 = np.clip(conf * conf_scale, 0, 65535).astype(np.uint16)
```

DA3 多视图推理输出的 `conf` 是每像素的绝对置信度值（不是 [0,1] 的概率），反映多视角几何一致性——值越大，多个视角对该像素深度估计越一致。实际范围大致 `[1.0, ~10+]`。乘以 `conf_scale=10000` 后超过 65535 的部分被截断。

### 2.2 多视角 vs 单图 fallback 都有 confidence

| 生成路径 | 触发条件 | 置信度来源 |
|---------|---------|-----------|
| 多视图推理 | 帧有 GT 位姿 | DA3 输出的几何一致性 conf |
| 单图 fallback | 帧无位姿（unposed） | DA3 模型输出的 conf（或梯度 proxy） |

两种方式都在 `data_rect/` 存深度，在 `data_rect_conf/` 存置信度，格式和 scale 完全相同。使用时统一读取、统一过滤即可，不需要区分生成方式。

### 2.3 Confidence 过滤是逐图片独立的

在线模式下阈值计算（`da3_depth_generator.py: _compute_low_confidence_mask`）：

```python
for i in range(N):
    conf_flat = conf[i].flatten()
    threshold = torch.quantile(conf_flat, self.conf_percentile / 100.0)
    low_conf_mask[i] = conf[i] < threshold
```

阈值按**每张图片单独计算**。一个训练样本包含多个视角（关键帧 + 相邻帧 + 立体对 + 鱼眼），每个视角独立计算自己的低置信 mask。

### 2.4 离线模式下 confidence 未被使用

当前 `trainer.py` 离线模式的处理：

```python
elif "depths" in data and len(data["depths"]) > 0:
    da3_depths = torch.stack(data["depths"], dim=1)
    # For offline mode, no confidence filtering (all high confidence)
    low_conf_mask = None
```

离线模式下 `low_conf_mask = None`，置信度过滤被完全跳过。confidence PNG 已生成保存但未被加载和使用。

---

## 三、当前实现的局限性

### 3.1 Confidence 是二值 fallback，不是连续调制

当前逻辑：
- `conf < 20th percentile` → **完全放弃**自适应，退回均匀采样
- `conf >= 20th percentile` → **完全信任** DA3 depth，用固定 σ 采样

第 21 百分位和第 99 百分位的像素获得完全相同的采样策略，而第 19 百分位的像素哪怕深度大体正确也被完全抛弃。

### 3.2 Confidence 只影响采样，不影响 Loss

当前 `loss.py` 是纯 RGB 重建 loss（L1/SSIM + 正则项），没有使用 DA3 深度做任何监督。DA3 depth 只用来决定"在哪采样"，不用来"教 NeRF 几何应该长什么样"。

### 3.3 σ 公式中 absrel_prior 是全局常数

当前：`σ(d) = max(min_thickness, absrel_prior × d)`，`absrel_prior=0.10` 是固定值。但实际不确定性逐像素变化——confidence 恰好编码了这个信息，却没被利用。

---

## 四、改进方案设计

### 方案 1：Confidence 调制采样宽度 σ（优先级 ★★★）

**核心思想**：不再二值 fallback，而是让 confidence 连续地控制采样分布的宽窄。

```
高置信 → σ 小 → 采样点紧密集中在表面附近 → 精细重建
低置信 → σ 大 → 采样点更分散 → 保守探索
极低置信/无深度 → fallback 到全局均匀采样
```

**公式**：

$$
\sigma(d, c) = \max\left(\text{min\_thickness},\ \frac{\text{absrel\_base}}{\text{normalize}(c)} \cdot d\right)
$$

其中 `normalize(c)` 将每张图片的 confidence 归一化到 `[ε, 1]`：

$$
\text{normalize}(c) = \text{clamp}\left(\frac{c - c_{\min}}{c_{\max} - c_{\min}},\ \epsilon,\ 1\right)
$$

效果：
- conf 最高时，`σ ≈ absrel_base × d`（如 `0.08 × d`），采样窗口最窄
- conf 最低时，`σ ≈ absrel_base / ε × d`（如 `0.8 × d`），采样窗口显著展宽
- 只有 `depth ≤ 0` 才 fallback 到全局均匀采样

**改动文件**：`nerf.py: sample_adaptive` — 接收连续 confidence 值替代二值 `low_conf_mask`；用 conf 调制 σ。

---

### 方案 2：Confidence 加权的深度监督 Loss（优先级 ★★★）

**核心思想**：利用 DA3 深度作为 NeRF rendered depth 的软监督，用 confidence 加权。

$$
\mathcal{L}_{\text{depth}} = \frac{\sum_i w_i \cdot |d_{\text{rendered},i} - d_{\text{DA3},i}|}{\sum_i w_i}
$$

权重 $w_i = \text{normalize}(c_i)^{\alpha}$，$\alpha \geq 1$ 控制对低置信的惩罚力度。

**为什么有效**：
- 当前系统只靠多视图 RGB 一致性学几何，收敛慢且容易在纹理稀少区域陷入局部最优
- DA3 提供了强深度先验，但并非处处准确
- confidence 告诉你哪里准、哪里不准
- 高置信区域深度被有效约束，低置信区域 NeRF 仍靠 RGB 自由学习

**改动文件**：`loss.py: ReconstructionLoss.__call__` — 新增 `lambda_depth_da3` 项。需要把 rendered depth、DA3 depth 和 conf 传入 loss。

---

### 方案 3：Confidence 加权的 RGB Loss（优先级 ★★☆）

某条射线的深度先验不准时，自适应采样把点放在错误位置，渲染出的 RGB 不可避免有误差，"噪声梯度"会干扰训练。

**做法**：

$$
\mathcal{L}_{\text{rgb}} = \frac{\sum_i w_i \cdot \text{L1+SSIM}(I_{\text{rendered},i}, I_{\text{gt},i})}{\sum_i w_i}
$$

- 高置信射线：采样准确 → RGB 重建可靠 → 高权重
- 低置信射线：采样可能偏 → RGB 重建有噪声 → 低权重

权重设置下限避免完全无监督：`w = clamp(normalize(c), 0.3, 1.0)`。

**改动文件**：`loss.py` — 在 `rgb_loss` 乘以 conf 权重。

---

### 方案 4：Confidence 自适应分配采样预算（优先级 ★☆☆）

根据 confidence 动态调整 `n_surface` vs `n_global` 分配比：
- 高置信：n_surface=48, n_global=16（大部分预算放表面附近）
- 低置信：n_surface=16, n_global=48（更多预算放全局探索）

实现复杂（需按射线不同数量做 scatter/gather），且边际收益不如方案 1（σ 展宽已达到类似效果）。**建议暂不实现**。

---

## 五、方案关系与优先级

```
                    ┌───────────────────────────────┐
                    │   DA3 Depth + Confidence Map   │
                    └───────┬──────────┬─────────────┘
                            │          │
              ┌─────────────▼──┐  ┌────▼──────────────────┐
              │  方案1: σ 调制  │  │  方案2: 深度监督 Loss  │
              │  (采样阶段)     │  │  (训练阶段)            │
              │  修改 nerf.py  │  │  修改 loss.py          │
              │  优先级: ★★★   │  │  优先级: ★★★           │
              └────────────────┘  └────────────────────────┘
                            │          │
                    ┌───────▼──────────▼─────────┐
                    │  方案3: RGB Loss 加权       │
                    │  (训练阶段)                  │
                    │  修改 loss.py               │
                    │  优先级: ★★☆                │
                    └────────────────────────────┘
```

- **方案 1 + 方案 2 是核心**，建议一起实现
- **方案 3 是锦上添花**，在方案 1+2 稳定后再加
- **方案 4 暂缓**，被方案 1 部分替代

---

## 六、代码改动清单

| 改动 | 文件 | 内容 |
|------|------|------|
| 离线加载 conf | `kitti_360_dataset.py` | `load_depth` 同时加载 `data_rect_conf/*.png`，返回 depth + conf |
| trainer 传递 conf | `trainer.py` | 离线模式下构建连续 `conf` tensor（非二值 mask），传入 sampler 和 renderer |
| σ 调制 | `nerf.py: sample_adaptive` | 接收 `conf` 张量，替代 `low_conf_mask`；用 conf 调制 σ |
| 深度监督 Loss | `loss.py` | 新增 `lambda_depth_da3` 项，conf 加权 L1 |
| RGB Loss 加权 | `loss.py` | 在 `rgb_loss` 乘以 conf 权重（可选，方案 3） |
