#!/usr/bin/env bash
set -euo pipefail

export PAPER_QUEUE_PROFILE="${PAPER_QUEUE_PROFILE:-backbone_robustness_p0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_paper_long_experiment_queue.sh" "$@"
