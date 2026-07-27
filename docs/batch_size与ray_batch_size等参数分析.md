# batch_size、ray_batch_size 等训练参数分析

基于 exp_kitti_360 配置与代码的数据流、计算量与收敛关系分析。

---

## 一、参数层级与角色

训练中射线与采样呈四层结构：

- 层级 1：batch_size = 16，每 step 的样本（场景）数
- 层级 2：ray_batch_size = 4096，每个场景参与 loss 的射线条数
- 层级 3：patch_size = 8×8，射线按 patch 组织，每 patch 64 条射线
- 层级 4：n_coarse = 64（renderer），每条射线上的深度采样点数（32 surface + 32 global）

因此：每 step 总射线数 = batch_size × ray_batch_size = 65536；每场景 patch 数 = 4096/64 = 64 个 8×8 patch；每 step 总 3D 采样点数 = 65536×64。

---

## 二、各参数的具体作用

**batch_size**：决定每步场景数与 GPU 显存/吞吐。越大梯度基于更多场景平均、方差越小，但显存与每 step 时间上升；越小可把省下的显存用于增大 ray_batch_size，提高单场景覆盖。

**ray_batch_size**：每个场景参与 RGB/分割 loss 的射线条数。越大单场景空间覆盖越高，有利于收敛与稳定性；越小则每场景只看到很少像素，梯度信号弱。在 patch 模式下必须能被 patch_size_x * patch_size_y 整除。

**patch_size**：把射线按空间局部块组织，便于 SSIM。不直接改变总射线数或总 3D 采样数。当前 8×8 与 SSIM 窗口一致。

**n_coarse**：每条射线上的深度采样点数。MLP 前向次数正比于射线数×n_coarse×2（RGB+分割）。当前 64 与自适应采样设计一致。

---

## 三、与计算量的关系

每 step MLP 调用次数量级 = batch_size × ray_batch_size × n_coarse × 2，约 838 万次“点”的评估。单场景像素覆盖率：单场景 4096 条射线，图像 192×640=122880 像素，约 3.3% 像素被采样。可考虑减小 batch_size、增大 ray_batch_size 提高单场景覆盖率。

---

## 四、参数间关系与平衡建议

batch_size 与 ray_batch_size 在显存允许下此消彼长。若 loss 方差大、单场景学得慢，可减小 batch_size（如 8）、增大 ray_batch_size（如 8192）。增大 ray_batch_size 时可保持 patch_size=8 或略增（如 16）。n_coarse 当前 64 一般无需单独减小。参考组合：batch_size=8，ray_batch_size=8192，patch_size=8 或 16。

---

## 五、Epoch 与 step 关系

训练集约 96901 样本，batch_size=16 时每 epoch 约 6056 iters。5000 step 约 82.5% 一个 epoch。StepLR 在 120000 step 衰减，会在训练中段生效。

---

## 六、小结

优先保证足够的 ray_batch_size（单场景覆盖），再通过 batch_size 调节方差与吞吐；patch_size 与 n_coarse 可按 SSIM 与现有设计微调。
