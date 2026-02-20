set -euo pipefail

DEVICE="${DEVICE:-cuda}"
CASE_ROOT="${CASE_ROOT:-task2_output_gpu_baseline_traj}"
OUT_DIR="${OUT_DIR:-task3_output_gpu}"

K_LIST="${K_LIST:-1 4 16}"
GRID_SIZE="${GRID_SIZE:-41}"
ALPHA_RANGE="${ALPHA_RANGE:-1.0}"
BETA_RANGE="${BETA_RANGE:-1.0}"
SUBSAMPLE_INTERIOR="${SUBSAMPLE_INTERIOR:--1}"
SUBSAMPLE_BOUNDARY="${SUBSAMPLE_BOUNDARY:--1}"
LOG_SCALE="${LOG_SCALE:---log-scale}"
DIRECTION_SOURCE="${DIRECTION_SOURCE:-pca}"
NORMALIZE_PCA="${NORMALIZE_PCA:---normalize-pca}"

missing=0
for k in ${K_LIST}; do
  for method in data pinn; do
    case_dir="${CASE_ROOT}/K${k}/${method}"
    if [[ ! -f "${case_dir}/model.pt" ]]; then
      missing=1
    fi
    if [[ "${DIRECTION_SOURCE}" == "pca" && ! -f "${case_dir}/trajectory.npz" ]]; then
      missing=1
    fi
  done
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi


python3 task3_landscape.py \
  --device "${DEVICE}" \
  --case-root "${CASE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --k-values ${K_LIST} \
  --methods data pinn \
  --grid-size "${GRID_SIZE}" \
  --alpha-range "${ALPHA_RANGE}" \
  --beta-range "${BETA_RANGE}" \
  --direction-source "${DIRECTION_SOURCE}" \
  --subsample-interior "${SUBSAMPLE_INTERIOR}" \
  --subsample-boundary "${SUBSAMPLE_BOUNDARY}" \
  ${LOG_SCALE} \
  ${NORMALIZE_PCA}
