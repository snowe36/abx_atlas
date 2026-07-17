#!/usr/bin/env bash
# End-to-end CPU reproduce: download → atlas → QSAR leakage diagnostics
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -e ".[dev]"
abx-download --max-per-organism "${MAX_PER_ORG:-5000}"
abx-atlas
abx-qsar

echo ""
echo "Done. Figures → reports/figures/"
echo "Tables  → data/processed/"
