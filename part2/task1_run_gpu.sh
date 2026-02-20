set -euo pipefail

DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-part2/task1_output}"
DATA_DIR="${DATA_DIR:-part2/FNO_data}"

EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MODES="${MODES:-16}"
WIDTH="${WIDTH:-64}"
DEPTH="${DEPTH:-4}"

python3 part2/task1_train_fno.py \
  --device "${DEVICE}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --modes "${MODES}" \
  --width "${WIDTH}" \
  --depth "${DEPTH}"
