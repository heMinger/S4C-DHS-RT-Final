#!/usr/bin/env bash
# 启动 A5(遮挡监督) 训练。
#
# A5 的完整代码/配置只在 occlusion-supervision 分支(当前 s4c-dhs-rt-e2 分支已删掉
# occlusion_supervision_enabled / lambda_occlusion 等字段)。本脚本用独立 worktree 跑,
# 不扰动当前分支未提交的改动, 也不污染 s4c_dhs_rt_e1 的训练输出。
#
# 用法:
#   bash scripts/launch_a5.sh compressed              # bs2+ray1024+accum16, 有效 batch 减半(32768), smoke test
#   bash scripts/launch_a5.sh strict                  # bs2+ray1024+accum32, 有效 batch 与原一致(65536), 可对比
#   bash scripts/launch_a5.sh full                    # 原配置(约 29.6 GiB), 需先停 E1
#   bash scripts/launch_a5.sh compressed resume_from=/home/lmh/s4c/out/training_checkpoint_176000.pt reset_training_state=true
#                                                     # warm-start(剩余参数透传给 train.py, 等价于在 config 里设)
#
# 说明:
#   - 有效 batch(射线数/优化步) = batch_size x ray_batch_size x accumulation_steps。
#     原 A5/E1 = 8 x 4096 x 2 = 65536。
#   - compressed 档(bs2+ray1024+accum16)= 32768, 有效 batch 减半, 别和 E1 直接比指标。
#   - strict 档(bs2+ray1024+accum32)= 65536, 与原配置一致, 但 32 步梯度累积更慢。
#   - 降 bs 省 encoder(ResnetEncoder 带 BatchNorm2d)+渲染内存; ray_batch_size 只省渲染内存
#     (渲染器用 LayerNorm, 不碰 BN)。with_amp 已确认在 utils/base_trainer.py 生效。
set -euo pipefail

MAIN_REPO="/home/lmh/s4c-DHS-RayTransformer"
WT_DIR="/home/lmh/s4c-A5-occlusion"
BRANCH="occlusion-supervision"
A5_OUT="/mnt/disk1/lmh/A6000-bp/s4c-DHS-out"   # A5 历史输出盘(GPU 主机上可访问)

MODE="${1:-compressed}"; shift || true
EXTRA_ARGS=("$@")

# 1) 建立/复用独立 worktree
if [ ! -d "$WT_DIR/.git" ]; then
  echo "[launch_a5] 创建 worktree: $WT_DIR ($BRANCH)"
  git -C "$MAIN_REPO" worktree add "$WT_DIR" "$BRANCH"
else
  echo "[launch_a5] 复用已有 worktree: $WT_DIR"
fi

# 2) worktree 内 out/kitti_360 软链到 A5 输出盘(与历史训练一致)
if [ ! -e "$WT_DIR/out/kitti_360" ]; then
  mkdir -p "$WT_DIR/out"
  ln -s "$A5_OUT" "$WT_DIR/out/kitti_360"
  echo "[launch_a5] 软链 out/kitti_360 -> $A5_OUT"
fi

# 2b) git worktree 只检出 tracked 文件; datasets/ 与 configs/data/ 是 gitignored,
#     必须从主仓库软链过来, 否则启动时 `from datasets.data_util import make_datasets`
#     会报 ModuleNotFoundError, 数据配置也无法解析。
for LINK in datasets configs/data; do
  if [ ! -e "$WT_DIR/$LINK" ]; then
    ln -s "$MAIN_REPO/$LINK" "$WT_DIR/$LINK"
    echo "[launch_a5] 软链 $LINK -> $MAIN_REPO/$LINK"
  fi
done

# 3) 选择显存档位
case "$MODE" in
  compressed)
    OVERRIDES=(
      batch_size=2
      accumulation_steps=16
      model_conf.ray_batch_size=1024
      with_amp=true
      renderer.eval_batch_size=50000
    )
    ;;
  strict)
    OVERRIDES=(
      batch_size=2
      accumulation_steps=32
      model_conf.ray_batch_size=1024
      with_amp=true
      renderer.eval_batch_size=50000
    )
    ;;
  full)
    OVERRIDES=()
    ;;
  *)
    echo "[launch_a5] 未知档位: $MODE (可选 compressed|strict|full)" >&2
    exit 1
    ;;
esac

echo "[launch_a5] 档位=$MODE overrides=${OVERRIDES[*]} extra=${EXTRA_ARGS[*]}"
cd "$WT_DIR"
# 减少 PyTorch 显存碎片 (OOM 报错提示: reserved-but-unallocated 较大时开启)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py -cn exp_kitti_360 "${OVERRIDES[@]}" "${EXTRA_ARGS[@]}"
