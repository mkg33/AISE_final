set -euo pipefail

DEVICE="${DEVICE:-cuda}"
MODEL_DIR="${MODEL_DIR:-part2/task1_output}"
DATA_DIR="${DATA_DIR:-part2/FNO_data}"
OUT_DIR="${OUT_DIR:-part2/task2_output}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESOLUTIONS="${RESOLUTIONS:-32 64 96 128}"

python3 part2/task2_eval_resolutions.py \
  --device "${DEVICE}" \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --resolutions ${RESOLUTIONS}
