#!/usr/bin/env bash
# install_xla_backend.sh — apply this fork's comfy-kitchen XLA backend.
#
# WHY THIS IS A SCRIPT AND NOT A PATCH FILE
# ------------------------------------------
# The backend has to live inside an *installed* Python package:
# `comfy_kitchen/backends/xla/`, plus one line in `registry.py`'s priority list
# and one import in `comfy_kitchen/__init__.py`. comfy-kitchen is installed
# normally (not editable), so a bare `pip install comfy-kitchen` — or a fresh
# machine — erases it silently. Anything not committed to this fork is work that
# evaporates.
#
# This script is idempotent. Run it after any comfy-kitchen install.
#
# WHAT IT DOES
#   1. copies backends/xla/{__init__,quantization}.py into the installed package
#   2. patches the priority list to insert "xla" before "eager"
#   3. adds the backend import
#   4. verifies dispatch: an xla tensor selects xla, a cpu tensor still selects
#      eager (the negative control — if xla started winning on cpu, the
#      device scoping is broken)
#
# IDEMPOTENCE NOTE
#   The two one-line edits are applied by exact-string match and skipped when the
#   desired line is already present, so re-running is safe and reports "already".
#   If comfy-kitchen is upgraded past 0.2.37 those exact strings may stop
#   matching — the script FAILS LOUDLY in that case rather than silently
#   half-applying, because a backend that is copied in but never registered
#   dispatches nowhere and looks like a working install.
#
# USAGE
#   ./patches/comfy_kitchen/install_xla_backend.sh            # apply + verify
#   ./patches/comfy_kitchen/install_xla_backend.sh --verify   # verify only
#
# TPU SAFETY: this script never imports with PJRT_DEVICE=TPU and never opens
# /dev/accel* or /dev/vfio/*. Verification runs on PJRT_DEVICE=CPU only.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/backends/xla"

# Verify with the CPU backend only. PJRT_DEVICE=CPU is deliberate: it must never
# claim chips, and device dispatch is testable without them.
export PJRT_DEVICE=CPU
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v5litepod-8

PY="${PYTHON_BIN:-python3}"

echo "== comfy-kitchen XLA backend installer =="
"$PY" -c "
import comfy_kitchen, os
print('  target package :', os.path.dirname(comfy_kitchen.__file__))
import importlib.metadata as m
print('  version        :', m.version('comfy-kitchen'))
" 2>/dev/null

if [ "${1:-}" = "--verify" ]; then
  exec "$PY" "$HERE/verify_xla_backend.py"
fi

DEST="$("$PY" -c 'import comfy_kitchen,os;print(os.path.dirname(comfy_kitchen.__file__))' 2>/dev/null)"
[ -d "$DEST" ] || { echo "FATAL: comfy_kitchen not importable" >&2; exit 1; }

# ---------------------------------------------------------------- 1. copy files
mkdir -p "$DEST/backends/xla"
for f in __init__.py quantization.py; do
  if [ ! -f "$SRC/$f" ]; then
    echo "FATAL: missing source $SRC/$f" >&2
    exit 1
  fi
  cp "$SRC/$f" "$DEST/backends/xla/$f"
  echo "  copied backends/xla/$f"
done

# ------------------------------------------------------- 2. priority list edit
PRIORITY_NEW='self._priority = ["ascend", "cuda", "triton", "xla", "eager"]'
if grep -qF '"xla", "eager"' "$DEST/registry.py"; then
  echo "  registry.py priority already contains xla"
elif grep -qF 'self._priority = ["ascend", "cuda", "triton", "eager"]' "$DEST/registry.py"; then
  "$PY" - "$DEST/registry.py" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = 'self._priority = ["ascend", "cuda", "triton", "eager"]'
new = 'self._priority = ["ascend", "cuda", "triton", "xla", "eager"]'
assert s.count(old) == 1, f"expected exactly one match, found {s.count(old)}"
p.write_text(s.replace(old, new))
print("  registry.py: inserted xla before eager")
PYEOF
else
  echo "FATAL: registry.py priority line not recognised." >&2
  echo "       comfy-kitchen was probably upgraded past 0.2.37." >&2
  echo "       The files are copied but NOT registered; dispatch will not use them." >&2
  echo "       Re-derive the edit against the new registry before trusting this." >&2
  exit 1
fi

# --------------------------------------------------------- 3. backend import
if grep -qF 'from .backends import xla as _xla_backend' "$DEST/__init__.py"; then
  echo "  __init__.py backend import already present"
else
  "$PY" - "$DEST/__init__.py" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); lines = p.read_text().splitlines(keepends=True)
idx = [i for i, l in enumerate(lines) if l.startswith("from .backends import ") and "eager" in l]
if not idx:
    raise SystemExit("FATAL: could not find the backend import block in comfy_kitchen/__init__.py")
i = idx[0]
indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines.insert(i + 1, f"{indent}from .backends import xla as _xla_backend  # noqa: F401\n")
p.write_text("".join(lines))
print("  __init__.py: added xla backend import")
PYEOF
fi

# ------------------------------------------------------------- 4. verify it took
echo
exec "$PY" "$HERE/verify_xla_backend.py"