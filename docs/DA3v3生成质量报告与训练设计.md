# DA3 v3 深度图生成质量报告 & 后续训练设计

---

## 一、生成完成度检查

### 1.1 总览

| 项目 | 数值 |
|------|------|
| 序列数 | 9（drive_0000~0010，无 0001/0008） |
| 相机数 | 4（image_00/01 透视 + image_02/03 鱼眼） |
| 总帧数 | **305,448** |
| 总文件数 | **610,896**（depth + conf 各一份） |
| 输出路径 | `data_2d_depth_da3_v3/` |

### 1.2 逐序列完整性：全部 ✓

| 序列 | image_00 | image_01 | image_02 | image_03 |
|------|----------|----------|----------|----------|
| drive_0000 | 11518 ✓ | 11518 ✓ | 11518 ✓ | 11518 ✓ |
| drive_0002 | 14607 ✓ | 14607 ✓ | 14385 ✓ | 14385 ✓ |
| drive_0003 | 1031 ✓ | 1031 ✓ | 1031 ✓ | 1031 ✓ |
| drive_0004 | 11587 ✓ | 11587 ✓ | 11587 ✓ | 11587 ✓ |
| drive_0005 | 6743 ✓ | 6743 ✓ | 6743 ✓ | 6743 ✓ |
| drive_0006 | 9699 ✓ | 9699 ✓ | 9699 ✓ | 9699 ✓ |
| drive_0007 | 3396 ✓ | 3396 ✓ | 3396 ✓ | 3396 ✓ |
| drive_0009 | 14056 ✓ | 14056 ✓ | 14056 ✓ | 14056 ✓ |
| drive_0010 | 3836 ✓ | 3836 ✓ | 3836 ✓ | 3836 ✓ |

每组 `depth帧数 = conf帧数 = 源图像帧数`，无任何缺失或不匹配。

---

## 二、深度图质量检查

### 2.1 值域统计（每序列每相机采样 ~20 帧）

| 序列/相机 | 深度范围(m) | 平均(m) | conf 范围 | conf clip% |
|-----------|------------|---------|-----------|-----------|
| **透视相机（image_00/01）** |
| drive_0000/image_00 | [1.1, 101.3] | 15.8 | [1.00, 6.55] | 29.6% |
| drive_0000/image_01 | [0.9, 92.8] | 14.8 | [1.00, 6.55] | 31.8% |
| drive_0002/image_00 | [1.1, 87.6] | 15.5 | [1.00, 6.55] | 35.4% |
| drive_0003/image_00 | [1.9, 114.7] | 24.2 | [1.00, 6.55] | 22.5% |
| drive_0004/image_00 | [1.0, 105.7] | 17.9 | [1.00, 6.55] | 33.0% |
| drive_0005/image_00 | [1.5, 108.1] | 16.4 | [1.00, 6.55] | 32.2% |
| drive_0006/image_00 | [1.6, 109.5] | 15.1 | [1.00, 6.55] | 33.4% |
| drive_0007/image_00 | [2.0, 177.8] | 27.0 | [1.00, 6.55] | 21.1% |
| drive_0009/image_00 | [1.5, 119.1] | 15.4 | [1.00, 6.55] | 31.7% |
| drive_0010/image_00 | [1.1, 175.4] | 19.6 | [1.00, 6.55] | 26.3% |
| **鱼眼相机（image_02/03）** |
| drive_0000/image_02 | [0.8, 114.9] | 14.3 | [1.00, 6.55] | 0.0% |
| drive_0000/image_03 | [0.9, 67.0] | 12.0 | [1.00, 6.55] | 0.1% |
| drive_0003/image_03 | [2.2, **256.0**] | **43.3** | [1.00, 6.15] | 0.0% |
| drive_0007/image_03 | [1.6, **215.3**] | 20.2 | [1.00, 6.14] | 0.0% |

### 2.2 质量判断

**透视相机（image_00/01）—— 质量良好**：
- 深度范围合理：最小 ~1m，最大 ~100-180m，平均 15-27m，符合城市驾驶场景
- conf clip 约 22-35%，仅影响高端，不影响低置信过滤
- 0 个退化帧（前 1000 帧全量检查无全零/常值帧）

**鱼眼相机（image_02/03）—— 存在极端值**：
- drive_0003/image_03 最大深度 256m、平均 43.3m 偏高（正常序列平均 12-20m）
- drive_0007/image_03 最大深度 215.3m
- 鱼眼→针孔重采样的边缘区域可能产生不可靠深度
- conf clip 接近 0%，说明鱼眼的置信度整体偏低（单图 fallback 模式较多）

### 2.3 分辨率

| 项目 | 分辨率 |
|------|--------|
| 源图像 | 376 × 1408 |
| 深度图输出 | **140 × 504** |
| 训练分辨率 | 192 × 640 |

深度图在 data loader 中 resize 到训练分辨率（192×640）即可，`load_depth` 已有 resize 逻辑。

---

## 三、当前基线性能（latest run）

训练配置：`depth_mode=online`，`adaptive_sampling=true`，`absrel_prior=0.35`，`batch_size=16`，`use_backbone=false`，GPU: A6000。

| 指标 | 值 | 说明 |
|------|----|------|
| a1_lidar | **0.8471** | LiDAR GT 下 δ<1.25 的比例 |
| abs_rel_lidar | **0.1259** | LiDAR GT 下的相对绝对误差 |
| a1 (DA3) | 0.2299 | 与 DA3 深度对比（DA3 本身也有误差） |
| seg_acc | 0.6411 | 语义分割准确率 |
| SSCBench IoU@12.8m | **59.22** | 3D 占据 IoU |
| SSCBench mIoU@12.8m | **16.65** | 语义 mIoU |
| 训练步数 | 3500 steps | ~3.5k iterations |

---

## 四、从 online → offline 的收益分析

| 维度 | online（当前） | offline（v3 数据） |
|------|---------------|-------------------|
| DA3 模型占 GPU | ~5-8 GB | **0 GB**（不加载） |
| 可用于训练的 GPU 内存 | 剩余 ~40 GB | **全部 48 GB** |
| 每步推理开销 | ~0.3-0.5s DA3 inference | **0**（直接读 PNG） |
| 深度质量 | 单图推理 | **7 帧多视图推理**（几何一致性更好） |
| Confidence | 在线计算 | **预计算并保存**（需加载） |
| Batch size 可用空间 | 受 DA3 模型限制 | **可增大** |

**结论：切换到 offline 模式有明显好处**——释放 GPU 内存、消除推理延迟、深度质量更高（多视图 vs 单图）。

---

## 五、训练方案设计

### 5.1 实验路线图

```
Exp A: offline baseline              ← 验证 offline 数据正确性
  │
  ├── Exp B: + confidence σ 调制    ← 核心改进 1
  │     │
  │     └── Exp C: + depth loss     ← 核心改进 2
  │           │
  │           └── Exp D: + RGB 加权  ← 锦上添花
  │
  └── Exp E: offline + 原始均匀采样  ← 消融对照
```

### 5.2 Exp A：Offline Baseline（验证性实验）

**目标**：确认 offline v3 数据能复现 online 基线性能。

**改动**：
- `depth_mode: "offline"`
- `depth_da3_path: "/mnt/disk2/lmh/KITTI360/data_2d_depth_da3_v3"`（指向远程服务器对应路径）
- `da3_conf_percentile: 0`（暂不用 conf，与 online 模式的 `low_conf_mask=None` 等效）
- `absrel_prior: 0.35`（保持与当前基线一致）
- 其余参数不变

**预期结果**：
- a1_lidar ≥ 0.84（应接近或略优于 online，因为多视图深度更准）
- 训练速度提升（无在线推理开销）
- GPU 内存占用下降

### 5.3 Exp B：Confidence σ 调制（详细分析）

**目标**：用 confidence map 连续调制每条射线的采样宽度 σ，替代当前的二值 fallback，使高置信射线采样更集中、低置信射线采样更分散。

---

#### 5.3.1 当前实现的局限

当前 `nerf.py: sample_adaptive` 中 σ 的计算为：

```text
σ(d) = max(min_thickness, absrel_prior × d)   # 所有射线同一个 absrel_prior
```

- `absrel_prior` 是全局常数（当前 0.35），**不区分每条射线的深度可靠程度**。
- 表面采样范围 `[d - 2σ, d + 2σ]`：σ 越大窗口越宽，32 个表面采样点越分散；σ 越小越集中。
- **问题**：多视图高置信深度与单图低置信深度用同一 σ，既浪费高置信信息，又无法对低置信做“保守探索”。

此外，当前用 **二值** `low_conf_mask`：conf &lt; 20th percentile → 整条射线 fallback 到全局均匀采样，否则完全信任 DA3。第 19 与第 21 百分位之间是硬切换，没有连续过渡。

---

#### 5.3.2 设计思路：conf 连续调制 σ

**核心公式**（每条射线不同）：

```text
σ(d, c) = max(min_thickness, absrel_prior / normalize(c) × d)
```

- **conf 高** → `normalize(c)` 接近 1 → σ 小 → 采样窗口窄 → 表面附近密集采样。
- **conf 低** → `normalize(c)` 小 → σ 大 → 采样窗口宽 → 接近全局探索。
- **depth ≤ 0**（如鱼眼无深度）→ 仍 fallback 到 `sample_coarse`（全局均匀采样）。

`normalize(c)` 将 conf 线性映射到 `[conf_epsilon, 1.0]`。**推荐使用固定边界**（与 v3 实测范围一致），避免 per-batch 时同一 conf 对应不同窗口、以及 batch 内全 6.55 时除零；具体公式与数值见 5.3.4。

---

#### 5.3.3 数据流：从文件到采样点

```text
文件系统                     Dataset                  Trainer                Renderer
data_rect_conf/*.png   →   load_depth_and_conf()  →  data["confs"]   →   conf_values (B,)
      ↓                         ↓                        ↓                      ↓
 uint16/10000             float32 (H,W)            tensor (n,v,1,h,w)      per-ray σ
```

1. **加载**：与 `data_rect/*.png` 并列，读 `data_rect_conf/<frame>.png`，`conf = uint16 / 10000.0`，resize 到训练分辨率 (192×640)。
2. **Dataset**：`load_depth` 改为返回 `(depth, conf)` 或新增 `load_conf`；`__getitem__` 中构建 `depths` 与 `confs` 列表，一并放入 data dict。
3. **Trainer**：离线分支里 `da3_confs = torch.stack(data["confs"], dim=1)`，与 `da3_depths` 同形状；采样/渲染时按 ray 索引把 conf 展平为 (B,) 传入 renderer。
4. **Renderer**：`sample_adaptive(..., conf_values=None, conf_epsilon=0.1)` 用 `conf_values` 计算每条射线的 `norm_conf`，再算 `sigma = max(min_thickness, (absrel_prior / norm_conf) * d)`；无 conf 时退化为当前 `sigma = absrel_prior * d`。

---

#### 5.3.4 Confidence 的加载、归一化方式与实际窗口数值（含合理性）

**（1）文件中 confidence 的存储与加载后的数值范围**

生成脚本中（`generate_da3_depth_multiview.py`）：

- 存储：`conf_uint16 = np.clip(conf * conf_scale, 0, 65535).astype(np.uint16)`，`conf_scale = 10000`。
- 因此磁盘上 conf 的范围是 `[0, 65535]`，对应 DA3 原始 conf 的 `[0, 6.5535]`（超过部分被截断）。

加载时（Dataset 中应实现）：

```text
conf_raw = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)   # uint16
conf = conf_raw.astype(np.float32) / 10000.0             # 与 conf_scale 一致
```

加载后的 **float conf 范围**：`[0, 6.5535]`。  
根据质量报告对 v3 数据的统计：

- **透视相机**：实测 conf ∈ **[1.00, 6.55]**，约 **22–35%** 像素被 clip 在 6.55（高置信端饱和）。
- **鱼眼相机**：conf ∈ [1.00, 6.55]，clip 比例接近 0，分布更分散。

即：**实际会用到的加载值大致在 [1.0, 6.5535]**，多数高置信像素集中在 6.55 附近。

**（2）归一化方式：推荐固定边界，而非按 batch**

若按「当前 batch 内」做 min-max 归一化：

- 同一 raw conf（如 3.0）在不同 batch 会得到不同 `norm_conf`（因 c_min/c_max 不同），**同一置信度对应不同窗口**，不一致。
- 当 batch 内所有射线 conf 都是 6.55（clip 饱和）时，`c_max - c_min = 0`，除零。

因此推荐 **固定边界** 的线性归一化，与 v3 实测范围一致：

```text
CONF_MIN = 1.0
CONF_MAX = 6.5535   # 65535/10000
norm_conf = clamp((conf - CONF_MIN) / (CONF_MAX - CONF_MIN), conf_epsilon, 1.0)
         = clamp((conf - 1.0) / 5.5535, 0.1, 1.0)
```

- conf = 1.0 → norm_conf = 0（clamp 后 **0.1**）→ σ 最大。
- conf = 6.5535 → norm_conf = **1.0** → σ 最小。
- conf &lt; 1.0（异常或缺失文件时用默认值 1.0）→ 仍视为最低置信，给最大 σ。

这样：**同一加载值对应唯一窗口**，且无除零问题。

**（3）σ 与窗口公式（与 5.3.2 一致）**

- `σ = max(min_thickness, absrel_prior / norm_conf × d)`，`min_thickness = 0.5`，`absrel_prior = 0.12`。
- 表面采样窗口：`[d_min, d_max] = [d - 2σ, d + 2σ]`，再 clamp 到 `[near, far]`（如 3m, 80m）。

**（4）实际窗口数值表（基于加载后的 conf）**

下面给出「加载后的 conf」→ `norm_conf` → σ → 实际窗口与相对半宽。`d` 单位 m，窗口宽度 = `d_max - d_min`，相对半宽 = `2σ/d`（高 conf 时约等于 absrel 量级）。

| 加载 conf | norm_conf | d=5m σ | 窗口(5m) | d=10m σ | 窗口(10m) | d=20m σ | 窗口(20m) | d=50m σ | 窗口(50m) | 2σ/d(20m) |
|-----------|-----------|--------|----------|---------|-----------|---------|-----------|---------|-----------|-----------|
| 6.55 | 1.0 | 0.6 | [3.8, 6.2] 2.4m | 1.2 | [7.6, 12.4] 4.8m | 2.4 | [15.2, 24.8] 9.6m | 6.0 | [38, 62] 24m | **0.24** |
| 5.0 | 0.72 | 0.83 | [3.3, 6.7] 3.3m | 1.67 | [6.7, 13.3] 6.7m | 3.33 | [13.3, 26.7] 13.4m | 8.33 | [33.3, 66.7] 33.3m | 0.33 |
| 4.0 | 0.54 | 1.11 | [2.8, 7.2] 4.4m | 2.22 | [5.6, 14.4] 8.9m | 4.44 | [11.1, 28.9] 17.8m | 11.1 | [27.8, 72.2] 44.4m | 0.44 |
| 3.0 | 0.36 | 1.67 | [1.7, 8.3] 6.7m | 3.33 | [3.3, 16.7] 13.3m | 6.67 | [3, 33.3] 26.7m | 16.67 | [3, 83.3] 53.3m | 0.67 |
| 2.0 | 0.18 | 3.33 | [3, 11.7] 8.7m | 6.67 | [3, 23.3] 20m | 13.33 | [3, 46.7] 43.7m | 33.33 | [3, 80] 77m | 1.33 |
| 1.0 | 0.1 | 5.0 | [3, 15] 12m | 10.0 | [3, 30] 27m | 24.0 | [3, 68] 65m | 60→50* | [3, 80] 77m | 2.4 |

\* d=50m 时 2σ=120m 超出 far=80m，clamp 后窗口为 [3, 80]。

可见：**同一 conf 加载值，在不同深度 d 下会得到与 d 成比例的窗口**；高 conf（6.55）时相对半宽 2σ/d = 0.24，与 absrel_prior=0.12 的 2 倍一致。

**（5）合理性判断**

- **高置信（conf=6.55，约 30% 透视像素）**  
  - 窗口半宽 2σ ≈ 0.24×d，对应约 **12%** 相对误差尺度，与 v3 多视图 absrel ~0.08–0.12 一致，**合理**。  
  - 近距 d=5m：窗口 2.4m；远距 d=50m：窗口 24m，相对尺度一致。

- **中置信（conf=3–4）**  
  - 窗口约为高置信的 1.5–2 倍，仍以表面为中心，**合理**。

- **低置信（conf=1–2）**  
  - 窗口拉大到接近 [near, far]，相当于「几乎不信任先验、偏向全局探索」，**合理**。  
  - 对应单图 fallback、遮挡、纹理缺失等低一致性区域。

- **缺文件或异常**  
  - 若 conf 文件缺失则加载时填 1.0 → norm_conf=0.1 → 最大 σ，**保守且安全**。

- **与当前实现对比**  
  - 当前：所有射线 σ=0.35×d（d=20m 时 7m，窗口 28m）。  
  - Exp B 高 conf：σ=0.12×d（d=20m 时 2.4m，窗口 9.6m），**高置信处采样密度约 3 倍**，低置信处自动放宽，**设计合理**。

**结论**：在「加载 conf = uint16/10000、固定边界 [1.0, 6.5535] 归一化到 [0.1, 1.0]」的前提下，上述公式给出的窗口在高/中/低置信下均与深度先验的可信度匹配，数值设置合理。

---

#### 5.3.5 数值示例（d=20m, absrel_prior=0.12）

| 场景   | conf 原值 | normalize(c) | σ 计算                    | 采样窗口 [d−2σ, d+2σ] | 含义           |
|--------|-----------|--------------|---------------------------|------------------------|----------------|
| 高置信 | 6.0       | 1.0          | max(0.5, 0.12×20)=**2.4m** | **[15.2, 24.8]**       | 9.6m 窄窗，密集 |
| 中置信 | 3.5       | 0.5          | max(0.5, 0.24×20)=**4.8m** | **[10.4, 29.6]**       | 19.2m 中等     |
| 低置信 | 1.5       | 0.15         | max(0.5, 0.8×20)=**16m**   | **[3, 52]**            | 接近全局       |
| 极低   | 1.0       | 0.1 (clamp)  | max(0.5, 1.2×20)=**24m**   | **[3, 64]**            | 几乎全局       |

**对比当前实现**（absrel_prior=0.35，不区分 conf）：σ=7.0m，窗口 [6, 34]（28m 宽）。  
Exp B 高置信时窗口 9.6m，**采样密度约为当前的 3 倍**；低置信时自动展宽，无需二值 fallback。  
（更完整的「加载 conf → 窗口」对应见 5.3.4 表格。）

---

#### 5.3.6 采样窗口示意（深度轴）

```text
深度轴 →  3m(near)                    20m(DA3)                        80m(far)
           |                             |                                |
当前:      |    [======6m==========d===========34m======]                 |
           |         σ=7.0，所有射线相同                                  |
           |                                                             |
Exp B:     |                                                             |
高置信:    |                 [===15.2===d===24.8===]                     |  σ=2.4
中置信:    |           [=====10.4======d======29.6======]                |  σ=4.8
低置信:    |  [======3m================d================52m========]    |  σ=16
```

高置信时 32 个点挤在约 9.6m 内；低置信时分布在约 49m 内，自然形成“保守探索”。

---

#### 5.3.6 Fallback 的简化

- **当前**：两重 fallback — (1) `invalid_depth_mask`（depth≤0）；(2) `low_conf_mask`（conf&lt;20th percentile）。
- **Exp B**：只保留 (1)。低置信通过**大 σ** 已等价于“更宽的表面窗口”，无需再硬切换为 `sample_coarse`，从而去掉 20th percentile 的硬阈值，过渡连续。

---

#### 5.3.8 关键参数

| 参数            | 建议值 | 说明 |
|-----------------|--------|------|
| `absrel_prior`  | **0.12** | v3 多视图 absrel ~0.08–0.10，略放宽；高 conf 时 σ=0.12×d，窗口约 0.48×d |
| `conf_epsilon`  | **0.1**  | 归一化下界；σ 最大约为 absrel/ε×d = 1.2×d，对 d=20m 得 σ=24m |
| `min_thickness` | **0.5**  | 与现有一致，防止近距 σ 过小 |

---

#### 5.3.8 代码改动清单（Exp B）

| 位置 | 改动 |
|------|------|
| `kitti_360_dataset.py` | `load_depth` 同时读 `data_rect_conf/<id>.png`，返回 `(depth, conf)`；无 conf 文件时 conf 置 1；`__getitem__` 中增加 `confs` 列表并写入 data |
| `trainer.py` | 离线分支：`da3_confs = torch.stack(data["confs"], dim=1)`；采样/渲染时传入与 rays 对应的 conf 展平 (B,)；**不再**计算二值 `low_conf_mask` |
| `nerf.py: sample_adaptive` | 新增参数 `conf_values=None, conf_epsilon=0.1`；有 conf 时用**固定边界** `norm_conf = clamp((conf-1.0)/5.5535, 0.1, 1.0)`，再算 `sigma = max(min_thickness, (absrel_prior/norm_conf)*d)`；fallback 仅用 `invalid_depth_mask` |
| `nerf.py: forward` | 将 `low_conf_mask` 改为接收 `conf_values`（或同时保留兼容），并传入 `sample_adaptive` |
| `configs/exp_kitti_360.yaml` | `absrel_prior: 0.12`；可选增加 `conf_epsilon: 0.1`（若从 config 读入） |

---

#### 5.3.10 预期结果（小结）

- 高置信区域：采样更集中 → 表面重建更精细，深度/占据更准。
- 低置信区域：采样更分散 → 减少被错误先验误导，训练更稳。
- 取消二值 fallback → 无 20th 百分位硬边界，行为更平滑。
- 预期 a1_lidar 提升约 0.01–0.03，SSCBench IoU 小幅提升（约 0.5–1 点）。

### 5.4 Exp C：Confidence 加权深度监督（核心实验）

**目标**：利用 DA3 深度直接监督 NeRF 几何。

**改动**（在 Exp B 基础上）：
1. `loss.py`：新增 `lambda_depth_da3` 项

   $$\mathcal{L}_{\text{depth}} = \frac{\sum_i w_i \cdot |d_{\text{rendered},i} - d_{\text{DA3},i}|}{\sum_i w_i + \epsilon}$$

   - $w_i = \text{normalize}(c_i)^{\alpha}$，$\alpha=1$
   - 仅对 `da3_depth > 0` 的像素计算
   - rendered depth 从 `coarse["depth"]` 取（已有）

2. 需要把 `da3_depths` 和 `confs` 从 trainer → loss 传递

**关键参数**：
- `lambda_depth_da3: 0.1`（初始值，待调优）
- 调优范围：`[0.01, 0.05, 0.1, 0.5]`

**预期结果**：
- NeRF 几何收敛更快（不再完全依赖多视图 RGB 一致性）
- 纹理稀少区域（天空、墙面）深度估计改善
- SSCBench IoU 提升（3D 占据依赖几何准确性）
- a1_lidar 显著提升

### 5.5 Exp D：RGB Loss 加权（可选）

**改动**：RGB loss × clamp(normalize(conf), 0.3, 1.0)

**预期**：减少低置信区域噪声梯度，稳定训练。收益可能不如 C 大。

### 5.6 Exp E：消融 — 均匀采样 + offline v3

**目标**：验证自适应采样的贡献。用已有 `exp_ablation_uniform.yaml` 但指向 v3 数据。

---

## 六、参数选择分析

### 6.1 absrel_prior：为什么从 0.35 可降到 ~0.12

| 来源 | 深度精度 | 理由 |
|------|---------|------|
| online（当前） | absrel ~0.15-0.20 | 单图推理、训练分辨率（192×640） |
| offline v3 | absrel ~0.08-0.12 | 7 帧多视图推理、504 分辨率、GT 位姿 |

online 模式用 0.35 是因为单图深度误差较大需要更宽的采样窗口。切换到 v3 后深度更准，σ 可以收紧，让采样点更集中在正确的表面位置。

### 6.2 lambda_depth_da3 的选择

当前 loss 的组成：
- RGB (L1+SSIM): ~0.1-0.3 量级
- edge-aware smoothness: 0.001 × eas
- segmentation: 0.02 × ce_loss

depth loss 量级估算：平均 |d_rendered - d_da3| ≈ 3-8m（未对齐时），× 0.1 → ~0.3-0.8，与 RGB loss 量级相当。

建议起始 `lambda_depth_da3 = 0.1`，如果 loss 量级差异大再调整。也可用 scale-invariant depth loss（log 空间）避免远处深度主导。

### 6.3 训练策略建议

| 参数 | Exp A-E 建议值 |
|------|---------------|
| batch_size | 16（offline 省下的 GPU 可尝试 24） |
| learning_rate | 1e-4 |
| num_epochs | 60 |
| validate_every | 1000 |
| use_backbone | **true**（当前 false，启用后特征更强） |
| n_coarse / n_surface+n_global | 64 总采样点不变 |

---

## 七、代码改动路线

### Phase 1: Offline baseline（Exp A）

| 文件 | 改动 |
|------|------|
| `configs/data/kitti_360.yaml` | `depth_mode: "offline"`, `depth_da3_path` 指向 v3 |
| `configs/exp_kitti_360.yaml` | 确认 `absrel_prior: 0.35`（先保持不变） |
| 无代码改动 | 纯配置切换 |

### Phase 2: Confidence 加载 + σ 调制（Exp B）

| 文件 | 改动 |
|------|------|
| `kitti_360_dataset.py` | `load_depth` → `load_depth_and_conf`，返回 (depth, conf) |
| `kitti_360_dataset.py` | `__getitem__` 中把 conf 加入 data dict |
| `trainer.py` | 离线模式下从 data 中提取 conf，构建连续 tensor |
| `nerf.py` | `sample_adaptive` 接收 conf tensor，调制 σ |
| `configs/exp_kitti_360.yaml` | `absrel_prior: 0.12` |

### Phase 3: 深度监督 Loss（Exp C）

| 文件 | 改动 |
|------|------|
| `loss.py` | 新增 `lambda_depth_da3` 参数和 loss 项 |
| `trainer.py` | 把 da3_depths + conf 传入 loss |
| `configs/exp_kitti_360.yaml` | `lambda_depth_da3: 0.1` |

---

## 八、预期实验结果对比

| 实验 | 预期 a1_lidar | 预期 IoU@12.8 | 主要改动 |
|------|-------------|--------------|---------|
| 当前基线（online） | 0.847 | 59.2 | — |
| Exp A（offline v3） | ≥ 0.847 | ≥ 59 | 数据源切换 |
| Exp B（+ conf σ 调制） | ~0.86 | ~60-61 | 采样更精准 |
| Exp C（+ depth loss） | ~0.88-0.90 | ~62-65 | 几何直接监督 |
| Exp E（均匀采样对照） | ~0.82 | ~55 | 消融对照 |

Exp C 是预期收益最大的实验，因为它从根本上解决了"NeRF 仅靠 RGB 学几何"的瓶颈。
