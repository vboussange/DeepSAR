#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MUSCARI_MODEL_PATH="${MUSCARI_MODEL_PATH:-${ROOT_DIR}/scripts/results/benchmark/artifacts/MuScaRi_ClimateDEM/dae0789a3c87/ensemble_pretrained}"
export MUSCARI_FEATURES_DIR="${MUSCARI_FEATURES_DIR:-${ROOT_DIR}/data/processed/environmental_features}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "Launching MuScaRi widget"
echo "  model:    ${MUSCARI_MODEL_PATH}"
echo "  features: ${MUSCARI_FEATURES_DIR}"
echo "  url:      http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"

cd "${ROOT_DIR}/widget"
exec uv run python app.py
