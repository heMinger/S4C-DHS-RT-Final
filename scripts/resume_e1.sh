#!/usr/bin/env bash
# 一键续跑 E1 实验(S4C + DHS + RT 103 维加图像特征)。
#
# 背景: E1 的续跑命令每次都写在 run 目录的 train_command.txt 里, 但一旦换终端/清历史
# 就找不到。本脚本自动定位最新 run 目录里最新的 training_checkpoint_*.pt, 拼出等价命令。
#
# 用法:
#   bash scripts/resume_e1.sh                     # 自动取最新 run 的最新 checkpoint 续跑
#   bash scripts/resume_e1.sh --dry-run           # 只打印将要执行的命令, 不启动训练
#   bash scripts/resume_e1.sh --ckpt <path.pt>    # 指定 checkpoint
#   bash scripts/resume_e1.sh --fresh             # reset_training_state=true(只载权重, 不恢复优化器/步数)
#   bash scripts/resume_e1.sh -- batch_size=4     # "--" 之后的参数原样透传给 train.py
#
# 例(E1 在 2026-10-31 中断于 78500 步的等价命令):
#   bash scripts/resume_e1.sh --ckpt \
#     out/s4c_dhs_rt_e1/s4c_dhs_rt_e1_backend-None-1_20261029-225544/training_checkpoint_78500.pt
#
# 说明:
#   - 续跑(reset_training_state=false)会恢复 model + optimizer + lr_scheduler + iteration,
#     无缝接续; --fresh 则只载模型权重, 步数从 0 开始(用于换架构/换 batch 的 warm-start)。
#   - 每次启动都会新建 out/s4c_dhs_rt_e1/<name>_backend-None-1_<时间戳>/ 目录, 旧 run 保持不动。
#   - E1 配置里的 resume_from 默认指向 S4C 176000 warm-start, 这里用命令行覆盖。
set -euo pipefail

REPO="/home/lmh/s4c-DHS-RayTransformer"
WORKTREE="/home/lmh/s4c-dhs-rt-e1"   # E1 专用 worktree(推荐在这里跑, 不动主仓库分支)
BRANCH="s4c-dhs-rt-e1"
CONFIG="exp_s4c_dhs_rt_e1"
OUT_DIR="out/s4c_dhs_rt_e1"

CKPT=""
DRY_RUN=0
RESET_TRAINING_STATE="false"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --fresh)   RESET_TRAINING_STATE="true"; shift ;;
    --ckpt)    CKPT="${2:?--ckpt 需要参数}"; shift 2 ;;
    --)        shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

cd "$REPO"

# 0) 选择运行目录: 优先用 E1 专用 worktree(它已检出 s4c-dhs-rt-e1, 不必切主仓库分支)
RUN_DIR="$REPO"
if [ -d "$WORKTREE/.git" ] || [ -f "$WORKTREE/.git" ]; then
  WT_BRANCH="$(git -C "$WORKTREE" branch --show-current)"
  if [ "$WT_BRANCH" = "$BRANCH" ]; then
    RUN_DIR="$WORKTREE"
  else
    echo "[resume_e1] 警告: $WORKTREE 当前分支是 $WT_BRANCH (期望 $BRANCH), 改用主仓库" >&2
  fi
fi

# 1) 定位 checkpoint: 最新 run 目录里编号最大的 training_checkpoint_*.pt
#    (--dry-run 也要走这一步, 但它只读文件系统, 不改任何状态)
if [ -z "$CKPT" ]; then
  LATEST_RUN="$(ls -dt "$RUN_DIR/$OUT_DIR"/*/ 2>/dev/null | head -1 || true)"
  [ -n "$LATEST_RUN" ] || { echo "[resume_e1] 找不到 run 目录: $RUN_DIR/$OUT_DIR" >&2; exit 1; }
  CKPT="$(ls -1 "${LATEST_RUN}"training_checkpoint_*.pt 2>/dev/null \
          | sed -E 's/.*training_checkpoint_([0-9]+)\.pt/\1 &/' \
          | sort -n | tail -1 | cut -d' ' -f2- || true)"
  [ -n "$CKPT" ] || { echo "[resume_e1] $LATEST_RUN 下没有 training_checkpoint_*.pt" >&2; exit 1; }
  echo "[resume_e1] 最新 run: $LATEST_RUN"
fi

[ -f "$CKPT" ] || { echo "[resume_e1] checkpoint 不存在: $CKPT" >&2; exit 1; }
ITER="$(basename "$CKPT" | sed -E 's/.*_([0-9]+)\.pt/\1/')"
CKPT_ABS="$(realpath "$CKPT")"
echo "[resume_e1] checkpoint: $CKPT_ABS (iteration=$ITER, reset_training_state=$RESET_TRAINING_STATE)"

CMD=(python train.py -cn "$CONFIG"
     "resume_from=$CKPT_ABS"
     "reset_training_state=$RESET_TRAINING_STATE"
     "${EXTRA_ARGS[@]}")

if [ "$DRY_RUN" -eq 1 ]; then
  CUR_BRANCH="$(git -C "$RUN_DIR" branch --show-current)"
  echo "[resume_e1] --dry-run (不做任何改动):"
  echo "  运行目录: $RUN_DIR"
  echo "  当前分支: $CUR_BRANCH  ->  期望分支: $BRANCH"
  echo "  将执行:"
  echo "    (cd $RUN_DIR && ${CMD[*]})"
  exit 0
fi

cd "$RUN_DIR"

# 2) 若不是 worktree(而是主仓库), 需要先切到 E1 分支
CUR_BRANCH="$(git branch --show-current)"
if [ "$CUR_BRANCH" != "$BRANCH" ]; then
  echo "[resume_e1] 当前分支 $CUR_BRANCH -> 切换到 $BRANCH"
  git checkout "$BRANCH"
else
  echo "[resume_e1] 运行目录 $RUN_DIR (分支 $BRANCH)"
fi

# 3) 提示未提交改动(E2 的 use_base_density 默认 false, 对 E1 无效但需知情)
if ! git diff --quiet -- models/bts/model/models_bts.py; then
  echo "[resume_e1] 警告: models_bts.py 有未提交改动(E2 WIP)。"
  echo "[resume_e1]       E1 配置未启用 use_base_density, 该改动对 E1 行为无影响; 若要纯净复现请先 git stash。"
fi

# 4) worktree 缺 datasets / configs/data(gitignored, 不随 worktree 检出)会报
#    ModuleNotFoundError: No module named 'datasets', 这里缺就自动软链主仓库的
for LINK in datasets configs/data out; do
  if [ ! -e "$LINK" ]; then
    mkdir -p "$(dirname "$LINK")"
    ln -s "$REPO/$LINK" "$LINK"
    echo "[resume_e1] 自动软链 $LINK -> $REPO/$LINK"
  fi
done

echo "[resume_e1] 启动训练(日志: outputs/<日期>/<时间>/train.log)"
exec "${CMD[@]}"
