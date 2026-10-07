#!/usr/bin/env bash
# xla_tpu_v5e8.sh — launch ComfyUI-XLA on a Kaggle TPU v5e-8 slice (2x4).
#
# WHY THIS FILE EXISTS (measured, not assumed)
# -------------------------------------------
# comfy/model_management.py does `import torch_xla` at MODULE SCOPE, before
# any of ComfyUI's own code runs. In torch_xla 2.8 that import executes
# `_setup_default_env()` -> `_setup_libtpu_flags()` -> `tpu.version()` ->
# `tpu.get_tpu_env()`, which does an HTTP GET against
#   http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env
# when TPU_SKIP_MDS_QUERY is unset. On this host that URL 404s (the Cell 6
# GCE shim serves accelerator-type / instance-id / agent-worker-number, NOT
# instance/attributes/tpu-env), so a plain `python main.py --xla` dies with:
#
#   OSError: Failed to get TPU metadata
#   requests.exceptions.HTTPError: 404 ... /computeMetadata/v1/instance/attributes/tpu-env
#
# Setting TPU_SKIP_MDS_QUERY=1 makes torch_xla read the topology from env vars
# instead (`_using_env_vars()` in torch_xla/_internal/tpu.py). Same knob JAX
# needs — one variable, both stacks.
#
# TPU_ACCELERATOR_TYPE must match torch_xla's own regex in `tpu.version()`:
#   ^v(\d)([A-Za-z]?){7}-(\d+)$
# "v5litepod-8" matches (v, 5, "litepod", 8) -> version() == 5, which both
# selects the v5 LIBTPU_INIT_ARGS and makes num_logical_cores_per_chip()
# return 1 (not 2), i.e. 8 chips -> 8 XLA devices.
#
# STORAGE TIERS ON THIS HOST (AGENTS.md STORAGE REALITY)
#   /, /tmp and /kaggle/temp share ONE Docker overlay whose real budget is a
#   ~68 GiB COW store — `df` lies about this. /dev/shm is a real 164 GiB tmpfs.
#   XLA compile artefacts are large and read-heavy, so both the compile cache
#   and HF_HOME are placed on /dev/shm. Never point them at /kaggle/temp.
#
# USAGE
#   ./xla_tpu_v5e8.sh                          # SPMD over all 8 chips
#   ./xla_tpu_v5e8.sh --xla_spmd --xla_mesh 2,4 --xla_spmd_mem_divisor 1.0
#   ./xla_tpu_v5e8.sh --listen 0.0.0.0 --port 8188
#
#   NOTE --xla and --xla_spmd are mutually exclusive (ComfyUI's own argparse
#   group). This launcher picks the backend for you from $COMFY_XLA_MODE.

set -euo pipefail

# ---------------------------------------------------------------- backend mode
# SPMD/FSDPv2 (default): shards weights and activations across all chips and is
# the only mode that scales past one chip's HBM. This is the mode to use on v5e-8.
# Non-SPMD (--xla) replicates the model per device and does not grow the slice.
COMFY_XLA_MODE="${COMFY_XLA_MODE:-spmd}"
COMFY_XLA_MESH="${COMFY_XLA_MESH:-2,4}"

# SPMD usable-HBM divisor. The fork's original value of 3.0 reproduces its
# TPU v3-8 measurement ("only 3/8 of the memory is available in SPMD mode").
# It is an empirical constant from a different chip generation, NOT a law of
# SPMD. v5e must be calibrated: run with --xla_spmd_mem_divisor 1.0, watch the
# "SPMD memory:" log line, and lower it only if the runtime actually OOMs.
# See docs/XLA_PORT_MANIFEST.md for how to calibrate without guessing.
COMFY_XLA_MEM_DIVISOR="${COMFY_XLA_MEM_DIVISOR:-3.0}"

# ------------------------------------------------------------- TPU attach env
# Measured on this box; see AGENTS.md STEP 0 for the full rationale.
# TPU_SKIP_MDS_QUERY is the load-bearing one (no MDS = no 404 above).
# All three worker-hostname vars are exported explicitly even though the image
# ships some of them, so the launcher is self-contained.
export PJRT_DEVICE=TPU
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v5litepod-8
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost
export TPU_PROCESS_ADDRESSES=local
export TPU_CHIPS_PER_HOST_BOUNDS=2,4,1

# HBM fraction PJRT is allowed to preallocate.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

# ------------------------------------------------------------------ storage tier
# RAM-backed tmpfs. Compile artefacts and HF blobs are read-heavy and large;
# the root overlay's COW store is scarce and its reported free space is fiction.
export COMFY_XLA_CACHE_PATH="${COMFY_XLA_CACHE_PATH:-/dev/shm/xla_comfy_cache}"
export HF_HOME="${HF_HOME:-/dev/shm/hf}"
mkdir -p "$COMFY_XLA_CACHE_PATH" "$HF_HOME"

# ------------------------------------------------------------- custody guard
# TPU Flash Custody Contract: exactly one owner at a time. If another process
# holds /dev/accel*, refuse rather than fight for the chips.
#
# `lsof -t` (terse) is used deliberately. It emits ONLY PIDs, one per line, with
# no header. An earlier version of this guard parsed default `lsof` output with
# `awk 'NR>1'` to skip the header row — which silently reports "nobody owns the
# chips" whenever no header is present, i.e. the guard fails OPEN on exactly the
# input it exists to catch. Field-selected output removes the assumption.
if command -v lsof >/dev/null 2>&1; then
  holder_pids="$(lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | sort -u | tr '\n' ' ' || true)"
  # Strip whitespace-only results so an empty match is not a "holder".
  holder_pids="$(printf '%s' "$holder_pids" | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')"
  if [ -n "$holder_pids" ]; then
    echo "REFUSING TO START: another process owns TPU device nodes." >&2
    for p in $holder_pids; do
      echo "  pid=$p cmd=$(ps -o comm= -p "$p" 2>/dev/null || echo '<unknown>')" >&2
    done
    echo "Stop it first (or use tpu-audit to see ownership). Not touching it." >&2
    exit 1
  fi
fi

# ------------------------------------------------------------------- backend
case "$COMFY_XLA_MODE" in
  spmd)
    BACKEND=(--xla_spmd --xla_mesh "$COMFY_XLA_MESH" --xla_spmd_mem_divisor "$COMFY_XLA_MEM_DIVISOR")
    ;;
  replica)
    # Non-SPMD: model is replicated per device. Does NOT shard. Kept for
    # single-chip debugging only.
    BACKEND=(--xla)
    ;;
  *)
    echo "Unknown COMFY_XLA_MODE=$COMFY_XLA_MODE (expected: spmd | replica)" >&2
    exit 2
    ;;
esac

# Compile cache location. ComfyUI's model_management reads this env first and
# falls back to --xla_cache_path, then to /tmp.
export XLA_COMFY_CACHE_PATH="$COMFY_XLA_CACHE_PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "ComfyUI-XLA launch"
echo "  backend        : $COMFY_XLA_MODE ${BACKEND[*]}"
echo "  mesh           : $COMFY_XLA_MESH   (mem divisor $COMFY_XLA_MEM_DIVISOR)"
echo "  compile cache  : $COMFY_XLA_CACHE_PATH"
echo "  HF_HOME        : $HF_HOME"
echo "  extra args     : $*"

exec "$PYTHON_BIN" "$HERE/main.py" \
  "${BACKEND[@]}" \
  "$@"