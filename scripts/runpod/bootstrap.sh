#!/usr/bin/env bash
# Pod-side entrypoint for the GPU deep-model run (GNN + pretrained transformer).
#
# Uploaded and executed on the RunPod pod by launch_gpu_job.py — not meant to
# be run directly on your laptop. Reads its knobs from env vars so the
# orchestrator can configure a run without editing this file.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abx_atlas}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
cd "$REPO_DIR"

# --- Safety net: self-terminate no matter what happens below --------------
# This is a hard backstop independent of the local orchestrator (robust to a
# hung job, a crashed laptop, or a dropped SSH session). The orchestrator
# itself stops the pod immediately on normal completion; this is insurance.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-14400}" # 4h default
if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
  (
    sleep "$MAX_RUNTIME_SECONDS"
    echo "[watchdog] Max runtime (${MAX_RUNTIME_SECONDS}s) exceeded — self-terminating pod ${RUNPOD_POD_ID}"
    curl -s --request DELETE \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      >> /workspace/watchdog.log 2>&1
  ) & disown
  echo "[bootstrap] Watchdog armed: self-terminate after ${MAX_RUNTIME_SECONDS}s (pod ${RUNPOD_POD_ID})"
else
  echo "[bootstrap] WARNING: RUNPOD_API_KEY/RUNPOD_POD_ID not set — no self-terminate watchdog!" >&2
fi

echo "[bootstrap] Creating / activating venv at ${VENV_DIR}..."
# Official RunPod PyTorch images ship a PEP-668-managed system Python; always
# install into a workspace venv so `pip install -e ".[gpu]"` works.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip -q

echo "[bootstrap] Installing abx-atlas with the gpu extra..."
# Prefer the image's preinstalled torch if present; still pull the rest of the
# gpu extra (torch_geometric, transformers, optuna, …).
pip install -e ".[gpu]" -q

WITH_GNN="${WITH_GNN:-1}"
WITH_PRETRAINED="${WITH_PRETRAINED:-1}"
GNN_EPOCHS="${GNN_EPOCHS:-60}"
GNN_HPO_TRIALS="${GNN_HPO_TRIALS:-20}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-seyonec/ChemBERTa-zinc-base-v1}"
PRETRAINED_EPOCHS="${PRETRAINED_EPOCHS:-3}"
PRETRAINED_HPO_TRIALS="${PRETRAINED_HPO_TRIALS:-6}"

ARGS=(-v)
[[ "$WITH_GNN" == "1" ]] && ARGS+=(--with-gnn --gnn-epochs "$GNN_EPOCHS" --gnn-hpo-trials "$GNN_HPO_TRIALS")
[[ "$WITH_PRETRAINED" == "1" ]] && ARGS+=(
  --with-pretrained
  --pretrained-model "$PRETRAINED_MODEL"
  --pretrained-epochs "$PRETRAINED_EPOCHS"
  --pretrained-hpo-trials "$PRETRAINED_HPO_TRIALS"
)

echo "[bootstrap] Running: abx-qsar ${ARGS[*]}"
set +e
abx-qsar "${ARGS[@]}" 2>&1 | tee /workspace/abx_qsar_run.log
STATUS="${PIPESTATUS[0]}"
set -e

echo "$STATUS" > /workspace/DONE
echo "[bootstrap] abx-qsar exited with status $STATUS; wrote /workspace/DONE"
