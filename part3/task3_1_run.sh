set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/part3/task3_1_elasticity.json"
GAOT_DIR="${GAOT_DIR:-${ROOT_DIR}/part3/GAOT-upstream}"

if [[ ! -d "${GAOT_DIR}" ]]; then
  exit 1
fi

cd "${GAOT_DIR}"
python3 main.py --config "${CONFIG_PATH}"

python3 "${ROOT_DIR}/part3/task3_1_summary.py" --config "${CONFIG_PATH}"
