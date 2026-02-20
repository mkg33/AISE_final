set -euo pipefail

DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-task2_output_gpu_baseline}"

ADAM_STEPS="${ADAM_STEPS:-5000}"
LBFGS_STEPS="${LBFGS_STEPS:-500}"
HIDDEN_WIDTH="${HIDDEN_WIDTH:-128}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
K_VALUES="${K_VALUES:-1 4 16}"

python task2_train.py \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  --k-values ${K_VALUES} \
  --adam-steps "${ADAM_STEPS}" \
  --lbfgs-steps "${LBFGS_STEPS}" \
  --hidden-width "${HIDDEN_WIDTH}" \
  --hidden-layers "${HIDDEN_LAYERS}" \
  --sample-idx "${SAMPLE_IDX}"
