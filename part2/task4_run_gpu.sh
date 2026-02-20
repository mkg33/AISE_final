set -euo pipefail

DEVICE="${DEVICE:-cuda}"
PRETRAINED_DIR="${PRETRAINED_DIR:-part2/task3_output}"
DATA_DIR="${DATA_DIR:-part2/FNO_data}"
OUT_DIR="${OUT_DIR:-part2/task4_output}"

BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-500}"
TRAIN_SCRATCH="${TRAIN_SCRATCH:-}"
TIME_INPUT="${TIME_INPUT:-}"

python3 part2/task4_finetune.py \
  --device "${DEVICE}" \
  --data-dir "${DATA_DIR}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --finetune-epochs "${FINETUNE_EPOCHS}" \
  --scratch-epochs "${SCRATCH_EPOCHS}" \
  ${TRAIN_SCRATCH} \
  ${TIME_INPUT:+--time-input "${TIME_INPUT}"}
