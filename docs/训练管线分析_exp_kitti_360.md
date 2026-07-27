# S4C-Adaptive-Sampling1 训练管线完整分析报告

> 命令: `python train.py -cn exp_kitti_360`

---

## 一、训练入口与配置

**入口**：`train.py` 使用 Hydra 加载 `configs/exp_kitti_360.yaml` 配置，调用 `models.bts.trainer.training()`，最终由 `base_trainer.base_training()` 驱动 PyTorch-Ignite 引擎。

**核心配置参数**：

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| num_epochs | 60 |
| learning_rate | 1e-4 |
| image_size | [192, 640] |
| z_near / z_far | 3 / 80 |
| ray_batch_size | 4096 |
| patch_size | 8 |
| sample_mode | patch |
| n_coarse (renderer) | 64 |
| n_fine | 0 |
| lindisp | true (逆深度空间采样) |

---

## 二、数据加载

**数据集类**：`Kitti360Dataset`（`datasets/kitti_360/kitti_360_dataset.py`）

**4个相机**：
- **image_00** / **image_01**：前向左/右透视相机
- **image_02** / **image_03**：左/右鱼眼相机（通过 `FisheyeToPinholeSampler` 重采样为针孔模型）

**每个样本加载的帧数**：`frame_count=2`，`data_fc=2`，加上立体和鱼眼，每个样本有 **v = 4 cameras × 2 frames = 8 个视图**。

**深度来源（实际分支追踪）**：

配置文件写了 `depth_mode: "online"`，但由于 `data_util.py` 构造 `Kitti360Dataset` 时**没有传递** `depth_mode` 参数，Dataset 实际使用默认值 `depth_mode="offline"`。
因此 `online_depth_generation=False`，DA V2 在线深度模型不会被初始化。

实际走的是 `depth_source="da3"` 的离线分支，从预计算的 DA3 深度 PNG 文件加载：
- 路径：`/mnt/remote_data/lmh_data/KITTI360/data_2d_depth_da3/{seq}/image_0X/data_rect/{id}.png`
- 格式：uint16 PNG，除以 256.0 得到米制深度
- 原始分辨率：140×504，代码会 `cv2.resize` 到 192×640
- 覆盖：100% 密集（所有像素都有深度值）
- 鱼眼相机(image_02/03)：填零占位（DA3 没有处理鱼眼）

**`__getitem__` 返回的数据字典**：

```python
{
    "imgs":         [img_0, img_1, ..., img_7],       # 8个视图图像, 每个 (3, 192, 640)
    "poses":        [pose_0, ..., pose_7],             # camera-to-world 4x4
    "projs":        [K_0, ..., K_7],                   # 归一化内参 3x3 ([-1,1]空间)
    "depths":       [depth_0, ..., depth_7],           # DA3离线深度图 (1, 192, 640) — 鱼眼为零
    "lidar_depths": [lidar_depth_0],                   # LiDAR稀疏深度 (仅第0帧，用于评估)
    "segs_gt":      [seg_0, ..., seg_7],               # mask2former分割标签
    "segs_kitti_gt": [],                               # KITTI-360原始GT (未启用)
}
```

---

## 三、数据流向（完整前向传播）

**BTSWrapper.forward()** 的完整流程：

```
1. 数据准备
   imgs (n,v,3,192,640) → poses (n,v,4,4) → projs (n,v,4,4)
   以第0帧为关键帧，所有pose变换到第0帧坐标系

2. 帧分配 (kitti360-mono模式)
   将v=8视图随机分为:
   - ids_encoder = [0]         (仅第0帧编码特征，硬编码)
   - ids_render  = [约4个视图]  (提供颜色的"源"视图)
   - ids_loss    = [约4个视图]  (计算loss的"目标"视图)

3. 编码器提取特征
   imgs[:, ids_encoder] → Monodepth2(ResNet50 + Decoder)
   → grid_f_features: 多尺度特征图 (n, 1, 64, h', w')

4. DA3深度获取 (实际走 offline 分支)
   ⚠️ 配置写了 depth_mode="online"，但由于参数传递断裂
   （详见第九节），实际走的是 offline 分支：
   da3_depths = torch.stack(data["depths"], dim=1)  # 从 Dataset 预加载的 DA3 PNG
   low_conf_mask = None   # 离线模式没有置信度信息

5. 射线采样 (PatchRaySampler)
   从 ids_loss 视图中随机采 64 个 8×8 patch：
   → all_rays        (n, 4096, 8)
   → all_rgb_gt      (n, 4096, 3)
   → all_da3_depths  (n, 4096)      # 对应patch位置的DA3深度
   → all_low_conf_mask = None       # 无置信度过滤

6. 分割射线采样 (同样的PatchRaySampler)
   → segmentation_rays, seg_rgb_gt, segmentation_gt

7. 构建自适应采样配置
   adaptive_config = {
       "n_surface": 32, "n_global": 32,
       "absrel_prior": 0.10, "min_thickness": 0.5
   }

8. NeRF渲染 (prediction_mode="default"，走 else 分支，两次渲染)
   (a) 分割渲染: renderer(segmentation_rays, predict_segmentation=True,
                          da3_depths=..., adaptive_sampling_config=...)
   (b) RGB渲染:   renderer(all_rays, da3_depths=..., adaptive_sampling_config=...)

9. 重建输出
   → 将平坦射线结果 reshape 回 patch 格式
```

---

## 四、模型结构详解

### 4.1 Encoder: Monodepth2

```
输入: (n*nv, 3, 192, 640)
├── ResNet50 Encoder (pretrained ImageNet, 硬编码为True)
│   ├── conv1 → bn1 → relu     → features[0]: 64ch
│   ├── layer1 (maxpool后)      → features[1]: 256ch
│   ├── layer2                   → features[2]: 512ch
│   ├── layer3                   → features[3]: 1024ch
│   └── layer4                   → features[4]: 2048ch
└── Decoder (UNet-style with skip connections)
    ├── upconv blocks: num_ch_dec = [32, 32, 64, 128, 256]
    └── dispconv → d_out = 64
输出: 多尺度特征图, 最高分辨率 (n*nv, 64, 48, 160)
```

### 4.2 BTSNet (主模型)

```
输入: 3D点 xyz (n, n_pts, 3)

├── sample_features(xyz):
│   ├── 3D→2D投影: xyz → 通过pose和K投影到编码器图像坐标
│   ├── grid_sample: 双线性插值取特征 → (n, n_pts, 64)
│   ├── positional_encoding(z坐标): code_xyz (无条件启用)
│   │   └── num_freqs=6, freq_factor=1.5, include_input=True
│   │       d_out = 3 + 3*6*2 = 39
│   └── concat → (n, n_pts, 64+39 = 103)
│
├── mlp_coarse (ResnetFC):
│   ├── d_in = 103, d_out = 1 (仅预测sigma)
│   ├── n_blocks = 0 (无残差块，仅线性层)
│   └── d_hidden = 64
│   输出: sigma → softplus激活
│
├── sample_colors(xyz):
│   └── 投影到渲染源视图 → grid_sample → RGB (n, nv, n_pts, 3)
│
├── mlp_segmentation (SegNet, 硬编码结构):
│   ├── d_in = 103, d_out = 19 (Cityscapes 19类)
│   ├── 结构: Linear(103→64) → ReLU → Linear(64→19)
│   │   （⚠️ 配置文件 mlp_segmentation 参数被忽略，详见第九节）
│   └── softmax输出
│
└── 输出: (rgb, invalid, sigma) 或 (rgb, invalid, sigma, segs)
```

**关键设计**：`sample_color=True` → MLP只预测密度(sigma)，颜色通过直接从源图像采样获得。

### 4.3 NeRF渲染器

```
输入: rays (n, B, 8) = [origin(3), direction(3), near(1), far(1)]

├── 采样策略选择:
│   ├── 有DA3深度+adaptive_config → sample_adaptive() (自适应采样)
│   └── 无DA3深度 → sample_coarse()   (均匀采样)
│
├── composite(model, rays, z_samp):
│   ├── 沿射线生成3D点: points = origin + t * direction
│   ├── 调用BTSNet.forward(points) → rgb, sigma
│   ├── alpha = 1 - exp(-delta * relu(sigma))
│   ├── hard_alpha_cap=True → alphas[-1] = 1 (最远处强制不透明)
│   ├── T = cumprod(1 - alpha)  # 累积透射率
│   ├── weights = alpha * T
│   ├── rgb_final = sum(weights * rgb)
│   ├── depth_final = sum(weights * z_samp)
│   └── segs_final = sum(weights * segs) (如有)
│
└── 输出: {rgb, depth, weights, alphas, invalid, z_samps}
```

---

## 五、射线采样详解

### 5.1 Patch采样模式

```python
ray_batch_size = 4096
patch_size = 8×8 = 64 pixels/patch
patch_count = 4096 / 64 = 64 patches/sample
```

每个训练step，随机从 `ids_loss` 视图中采 64 个 8×8 的patch，共 4096 条射线。

分割射线也使用同样的PatchRaySampler独立采样一组。

### 5.2 DA3自适应采样 (核心创新)

**配置参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| adaptive_sampling | true | 启用自适应采样 |
| n_surface_samples | 32 | 表面附近采样点数 |
| n_global_samples | 32 | 全局均匀采样点数 |
| absrel_prior | 0.10 | DA3相对误差先验 |
| min_thickness | 0.5m | 最小采样厚度 |
| da3_conf_percentile | 20% | 低置信度阈值（⚠️ 实际未生效） |

**总采样点数**：每条射线 32 + 32 = **64个采样点**（与 `n_coarse=64` 一致）。

**自适应采样算法** (`NeRFRenderer.sample_adaptive()`):

```
对每条射线:
1. 获取DA3深度估计 d（从离线 DA3 PNG 文件加载）
2. 计算动态标准差: σ = max(0.5m, 0.10 × d)
3. Surface采样: 32个点均匀分布在 [d-2σ, d+2σ]，加上stratified jitter
4. Global采样:  32个点均匀分布在 [z_near=3, z_far=80]（逆深度空间）
5. 合并排序: cat([surface, global]) → sort → 64个采样点

低置信度处理 (设计意图: bottom 20%退化为均匀采样):
⚠️ 实际未生效！因为离线模式 low_conf_mask=None，
   所有射线都无条件走自适应采样，没有任何射线退化为均匀采样。
```

### 5.3 鱼眼视图的特殊情况

鱼眼相机(image_02/03)的DA3深度为全零。当射线来自鱼眼视图时：
- `da3_depths = 0` → 被 clamp 到 `z_near=3`
- `σ = max(0.5, 0.10 × 3) = 0.5`
- Surface 采样范围 = `[3-1, 3+1] = [2, 4]`（被 near clamp 到 [3, 4]）
- 所有 32 个 surface 采样点集中在 3~4m 极窄范围内

这意味着鱼眼视图的自适应采样严重退化，大部分采样点被浪费在近处。

---

## 六、监督信号与Loss

**ReconstructionLoss** 包含以下组件：

### 6.1 RGB重建损失 (主损失)

```
criterion = "l1+ssim"
loss_rgb = 0.85 × SSIM + 0.15 × L1
```

- 对比：渲染的RGB vs 目标视图真实RGB
- invalid_policy = `"weight_guided"`：如果射线超过90%权重落在无效区域，则忽略该射线
- 取所有渲染源视图的最小误差(`amin`)

### 6.2 语义分割损失

```
lambda_segmentation = 0.02
loss_seg = NLLLoss(log(softmax(pred)), gt)
```

- 使用 **mask2former** 预处理的19类Cityscapes分割标签作为GT
- 类别加权：pole(10×), traffic_sign(10×), sidewalk(10×), person(5×), traffic_light(5×) 等

### 6.3 正则化损失

```
lambda_edge_aware_smoothness = 0.001   # 边缘感知深度平滑
lambda_density_entropy = 0             # 密度熵正则 (未启用)
```

### 6.4 评估指标

**DA3深度对比** (使用离线DA3深度图作为参考):
- abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

**LiDAR深度对比** (使用真实LiDAR投影深度):
- a1_lidar, abs_rel_lidar

**分割准确率**: seg_acc, seg_acc_front, seg_acc_side

---

## 七、训练循环总结

```
每个训练Step:
┌─────────────────────────────────────────────────────┐
│ 1. DataLoader 加载 batch (16个样本, 每个8视图)       │
│    → imgs, poses, projs, depths(DA3离线), segs_gt   │
│                                                      │
│ 2. to(device) 移到GPU                               │
│                                                      │
│ 3. model.forward(data):                              │
│    a. 变换到第0帧坐标系                               │
│    b. 分配编码/渲染/loss帧 (kitti360-mono)            │
│    c. Monodepth2编码第0帧 → 64维特征图               │
│    d. 从data["depths"]获取DA3离线深度 (非在线生成)    │
│       low_conf_mask = None (无置信度过滤)            │
│    e. PatchRaySampler: 64个8×8 patch → 4096条射线    │
│       同时采样对应patch位置的DA3深度                   │
│    f. NeRF渲染 (分割+RGB各一次):                     │
│       - 自适应采样64点/射线 (32 surface + 32 global) │
│       - 投影到编码帧取特征+位置编码                    │
│       - MLP预测sigma, 从源视图采样颜色                │
│       - alpha compositing → RGB, depth, segs         │
│    g. Reshape回patch格式                             │
│                                                      │
│ 4. Loss计算:                                         │
│    loss = L1+SSIM(rgb) + 0.02×NLL(seg) + 0.001×EAS  │
│                                                      │
│ 5. Adam优化器更新 (lr=1e-4, 无warmup)               │
│    StepLR: 120000步后 ×0.1                           │
│                                                      │
│ 6. 每500iter可视化, 每1000iter评估                    │
│    保存最佳模型 (a1_lidar↑, abs_rel_lidar↓, a1↑,     │
│    seg_acc↑)                                         │
└─────────────────────────────────────────────────────┘
```

---

## 八、关键设计特点总结

1. **DA3引导的自适应采样**是本项目的核心创新，用DA3深度先验将采样点集中在预测表面附近，同时保留全局覆盖
2. **颜色不由MLP预测**，而是直接从源图像双线性插值获取，MLP仅输出1维密度sigma
3. **双分支渲染**：RGB渲染和分割渲染使用不同的射线集合独立进行
4. **多相机融合**：4个相机（2透视+2鱼眼→重采样为针孔）提供环视覆盖

---

## 九、配置参数断裂审计

以下逐一列出配置文件中设置了但**未正确传递到代码**的参数，以及**写了但无实际效果**的参数。

### 9.1 ❌ `data.depth_mode` — 未传入 Dataset 构造，online 分支不生效

**配置** (`data/kitti_360.yaml` 第33行):
```yaml
depth_mode: "online"
```

**断裂点**：`data_util.py` 第71-96行构造 `Kitti360Dataset` 时，没有传递 `depth_mode` 参数：
```python
# data_util.py 第71-96行
train_dataset = Kitti360Dataset(
    ...
    online_depth_generation=config.get("online_depth_generation", False),
    # ❌ 缺少: depth_mode=config.get("depth_mode", "offline"),
)
```

**实际效果**：`Kitti360Dataset` 使用默认值 `depth_mode="offline"` → `online_depth_generation=False` → DA V2 在线模型不会初始化 → `load_depth()` 走 `depth_source="da3"` 的离线 PNG 加载分支。

### 9.2 ❌ `data.da3_checkpoint_path` / `data.da3_conf_percentile` — 未传入 Dataset

**配置** (`data/kitti_360.yaml` 第43-44行):
```yaml
da3_checkpoint_path: "/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE"
da3_conf_percentile: 20.0
```

**断裂点**：`data_util.py` 构造 `Kitti360Dataset` 时完全没有传递这两个参数。

**实际效果**：使用默认值 `da3_checkpoint_path=None`，`da3_conf_percentile=20.0`（后者碰巧与配置一致但无意义，因为 online 模式未启用）。

### 9.3 ❌ `model_conf.online_depth_generation` — Trainer 层 DA3 在线生成不生效

**配置**：`model_conf` 中**没有设置** `online_depth_generation` 字段。

**代码** (`trainer.py` 第90行):
```python
self.online_depth_generation = config.get("online_depth_generation", False)  # → False
```

`BTSWrapper` 接收的是 `config["model_conf"]`，其中没有此字段 → 默认 `False` → Trainer 层的 `DA3DepthGenerator`（使用 DA3NESTED-GIANT-LARGE 模型的在线推理）永远不会被触发。

**实际效果**：Trainer 层走 else 分支 `elif "depths" in data`，从 Dataset 预加载的离线深度中获取。同时 `low_conf_mask = None`，置信度过滤完全不生效。

### 9.4 ⚠️ `data.return_depth` / `data.return_lidar_depth` — 配置无意义，靠硬编码覆盖

**配置** (`data/kitti_360.yaml` 第47-49行):
```yaml
return_depth: true
return_lidar_depth: true
```

**断裂点**：`data_util.py` 构造 `Kitti360Dataset` 时没有传递这两个参数（默认均为 `False`）。

**但已打补丁**：`trainer.py` 的 `get_dataflow()` 里手动覆盖：
```python
# trainer.py 第633-634行
train_dataset.return_depth = True
train_dataset.return_lidar_depth = True
```

**实际效果**：功能正常，但 yaml 里的设置是摆设。

### 9.5 ⚠️ `encoder.pretrained` — 配置写了但代码硬编码

**配置** (`exp_kitti_360.yaml` 第51行):
```yaml
encoder:
    pretrained: true
```

**代码** (`monodepth2.py` 第261行):
```python
self.encoder = ResnetEncoder(resnet_layers, True, 1)  # 第二参数 pretrained 硬编码为 True
```

`Monodepth2.from_conf()` 不读取 `pretrained` 字段。

**实际效果**：碰巧与配置一致（都是 True），但无法通过配置关闭预训练。

### 9.6 ❌ `model_conf.n_frames_encoder` — 完全无效

**配置** (`exp_kitti_360.yaml` 第82行):
```yaml
n_frames_encoder: 1
```

**代码**：整个项目中没有任何代码读取 `n_frames_encoder`。编码器帧在 `BTSWrapper.forward()` 中硬编码为 `ids_encoder = [0]`（第144行）。

**实际效果**：参数完全无效，编码器始终只用第0帧。

### 9.7 ❌ `model_conf.use_code` — 完全无效

**配置** (`exp_kitti_360.yaml` 第40行):
```yaml
use_code: true
```

**代码**：`BTSNet.__init__` 不读取 `use_code`，位置编码 `code_xyz` 无条件创建（第46行）。

**实际效果**：位置编码始终启用，无法通过配置禁用。

### 9.8 ❌ `model_conf.mlp_segmentation` 子配置 — 被忽略

**配置** (`exp_kitti_360.yaml` 第73-76行):
```yaml
mlp_segmentation:
    type: "resnet"
    n_blocks: 0
    d_hidden: 64
```

**代码** (`models_bts.py` 第65-66行):
```python
elif self.segmentation_mode in ('panoptic_deeplab', 'mask2former', 'm2f'):
    self.mlp_segmentation = make_segnet(d_in=d_in, d_out=19, d_hidden_list=[64])
```

当 `segmentation_mode='panoptic_deeplab'` 时，直接调用 `make_segnet` 并硬编码 `d_hidden_list=[64]`，完全不读取 `conf["mlp_segmentation"]`。

**实际效果**：分割头固定为 `SegNet(Linear(103→64) → ReLU → Linear(64→19))`，配置参数中的 `type`、`n_blocks` 等全部被忽略。

### 9.9 ❌ `default.yaml` 中的 `warmup_steps` 等 — 完全无效

**配置** (`default.yaml`):
```yaml
warmup_steps: 10000
decay_rate: 0.5
decay_steps: 100000
num_steps: 100000
```

**代码**：`scheduler.py` 的 `make_scheduler` 只支持 `fix` 和 `step` 两种类型，不读取这四个参数。`base_trainer.py` 也没有实现任何 warmup 逻辑。

**实际学习率调度**：从第1步开始就是 `lr=1e-4`，无 warmup，到第120000步降为 `1e-5`。

### 9.10 总结表

| 参数 | 配置值 | 状态 | 实际行为 |
|------|--------|------|---------|
| `data.depth_mode` | `"online"` | ❌ 未传入构造 | 默认 `"offline"`，读 DA3 PNG 文件 |
| `data.da3_checkpoint_path` | 设了路径 | ❌ 未传入构造 | 默认 `None`，无影响 |
| `data.da3_conf_percentile` | `20.0` | ❌ 未传入构造 | 默认 `20.0`（碰巧一致但无意义） |
| `data.return_depth` | `true` | ⚠️ 未传入，trainer硬覆盖 | 功能正常，配置是摆设 |
| `data.return_lidar_depth` | `true` | ⚠️ 未传入，trainer硬覆盖 | 功能正常，配置是摆设 |
| `model_conf` 无 `online_depth_generation` | 未设 | ❌ 默认 `False` | Trainer 层 DA3 在线生成不触发 |
| `encoder.pretrained` | `true` | ⚠️ 代码硬编码 `True` | 碰巧一致，但配置无法控制 |
| `model_conf.n_frames_encoder` | `1` | ❌ 无代码读取 | 完全无效 |
| `model_conf.use_code` | `true` | ❌ 无代码读取 | 完全无效 |
| `mlp_segmentation.*` | `type/n_blocks/d_hidden` | ❌ panoptic_deeplab模式不读取 | 分割头结构固定 |
| `warmup_steps` | `10000` | ❌ 无代码读取 | 完全无效，无warmup |
| `decay_rate/decay_steps/num_steps` | 各有值 | ❌ 无代码读取 | 完全无效 |

### 9.11 关键影响分析

**最严重的问题**是 `depth_mode="online"` 未生效导致的连锁反应：

1. **深度来源**：不是 DA3 在线推理，而是从离线 PNG 文件加载（分辨率 140×504，需上采样到 192×640）
2. **置信度过滤不生效**：`low_conf_mask=None`，所有射线无条件走自适应采样，包括深度估计可能不准的区域
3. **鱼眼视图退化**：鱼眼相机的 DA3 深度为全零，被 clamp 到 `z_near=3` 后，32 个 surface 采样点集中在 3~4m 极窄范围，采样效率极低
