set -euo pipefail

DEVICE="${DEVICE:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIG_CONFIG="${ROOT_DIR}/part3/task3_3_elasticity.json"
CONFIG_PATH="${ORIG_CONFIG}"

cd "${ROOT_DIR}/part3/GAOT-main"
if [[ -n "${DEVICE}" ]]; then
  TMP_CONFIG="$(mktemp)"
  trap 'rm -f "${TMP_CONFIG}"' EXIT
  python3 - <<PY
import json
import os

config_path = os.path.abspath("${ORIG_CONFIG}")
device = "${DEVICE}"
tmp_path = "${TMP_CONFIG}"

with open(config_path) as f:
    cfg = json.load(f)

cfg.setdefault("setup", {})["device"] = device

with open(tmp_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
  CONFIG_PATH="${TMP_CONFIG}"
fi

python3 main.py --config "${CONFIG_PATH}"

python3 "${ROOT_DIR}/part3/task3_3_summary.py" --config "${ORIG_CONFIG}"
