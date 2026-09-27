#!/usr/bin/env bash
# MoE TP / EP / CUDA Graph latency matrix (Qwen1.5-MoE-A2.7B).
# Each run remaps physical GPUs via CUDA_VISIBLE_DEVICES so --devices are 0-based.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N:-64}"
WARMUP="${WARMUP:-2}"
REPS="${REPS:-3}"
OUT="${OUT:-/tmp/moe_matrix_$(date +%Y%m%d_%H%M%S).log}"
PORT_BASE="${PORT_BASE:-29610}"
: >"$OUT"
PORT=$PORT_BASE

run() {
  local label="$1"
  local phys="$2"   # physical PCI ids, e.g. 3 or 1,2
  local arch="$3"   # FlashInfer arch list
  shift 3
  echo "===== $label (phys=$phys arch=$arch port=$PORT) =====" | tee -a "$OUT"
  set +e
  env -u CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$phys" \
    FLASHINFER_CUDA_ARCH_LIST="$arch" \
    PYTHONPATH=. \
    "$PY" bench/moe_e2e.py --model moe --n "$N" --warmup "$WARMUP" --reps "$REPS" \
      --nccl-port "$PORT" "$@" \
    2>&1 | tee -a "$OUT"
  local rc=${PIPESTATUS[0]}
  set -e
  PORT=$((PORT + 16))
  sleep 3
  if [[ $rc -ne 0 ]]; then
    echo "WARN: $label failed rc=$rc (continuing)" | tee -a "$OUT"
  fi
  echo | tee -a "$OUT"
}

echo "writing $OUT"
run "tp1 eager A100"      "3"   "8.0" --devices 0 --tp 1 --eager --moe-backend triton
run "tp1 cg A100"         "3"   "8.0" --devices 0 --tp 1 --cg --moe-backend triton
run "tp2 eager 3090x2"    "1,2" "8.6" --devices 0,1 --tp 2 --eager --moe-backend triton
run "tp2 cg 3090x2"       "1,2" "8.6" --devices 0,1 --tp 2 --cg --moe-backend triton
run "tp2+ep eager 3090x2" "4,5" "8.6" --devices 0,1 --tp 2 --ep --eager --moe-backend triton
run "tp2+ep cg 3090x2"    "4,5" "8.6" --devices 0,1 --tp 2 --ep --cg --moe-backend triton
echo "done -> $OUT" | tee -a "$OUT"
