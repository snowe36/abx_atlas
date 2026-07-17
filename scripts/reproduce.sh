#!/usr/bin/env bash
# End-to-end CPU reproduce: download → atlas → QSAR leakage diagnostics
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer an active venv; otherwise require Python 3.11+
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  else
    PYTHON=python3
  fi
else
  PYTHON=python
fi

VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
MAJOR="${VER%%.*}"
MINOR="${VER#*.}"
if (( MAJOR < 3 || (MAJOR == 3 && MINOR < 11) )); then
  echo "error: abx_atlas requires Python >= 3.11 (found $VER via $PYTHON)" >&2
  exit 1
fi

"$PYTHON" -m pip install -e ".[dev]"
abx-download --max-per-organism "${MAX_PER_ORG:-20000}" --all-organisms
abx-atlas
abx-qsar

echo ""
echo "Done. Figures → reports/figures/"
echo "Tables  → data/processed/"
