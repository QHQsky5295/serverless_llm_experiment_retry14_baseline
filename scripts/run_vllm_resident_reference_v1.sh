#!/usr/bin/env bash
set -euo pipefail

# Protocol-compliant Resident-vLLM wrapper.  It reuses the ordinary fair
# runner and its frozen trace/subset; the protocol flag only adds common-clock,
# resource-domain, prepublished remote-delivery, and physical-lifecycle
# evidence.  It never edits or regenerates the shared workload.

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"

case "${MODEL_PROFILE}" in
  llama2_7b_main_v2_publicmix)
    default_endpoint="http://192.168.4.174:18081"
    ;;
  llama32_3b_main_modelscope)
    default_endpoint="http://192.168.4.174:18080"
    ;;
  *)
    default_endpoint=""
    ;;
esac

export VLLM_RESIDENT_PROTOCOL_V1=1
export VLLM_LORA_REGISTRATION_MODE=dynamic_remote
export VLLM_GENERATION_CONTRACT=fixed_length_greedy_v1
export VLLM_REMOTE_ARTIFACT_ENDPOINT="${VLLM_REMOTE_ARTIFACT_ENDPOINT:-${default_endpoint}}"
export VLLM_RESIDENT_LEAD_S="${VLLM_RESIDENT_LEAD_S:-60}"
export VLLM_RESIDENT_SERVICE_MEMORY_MAX="${VLLM_RESIDENT_SERVICE_MEMORY_MAX:-80G}"
export VLLM_RESIDENT_AUX_MEMORY_MAX="${VLLM_RESIDENT_AUX_MEMORY_MAX:-4G}"
export VLLM_RESIDENT_SERVICE_TASKS_MAX="${VLLM_RESIDENT_SERVICE_TASKS_MAX:-4096}"
export VLLM_RESIDENT_AUX_TASKS_MAX="${VLLM_RESIDENT_AUX_TASKS_MAX:-128}"
export VLLM_REMOTE_ARTIFACT_BANDWIDTH_MBPS="${VLLM_REMOTE_ARTIFACT_BANDWIDTH_MBPS:-250}"
export VLLM_RESULT_TAG="${VLLM_RESULT_TAG:-${MODEL_PROFILE}_resident_v1_r${SLLM_TOTAL_REQUESTS}_a${SLLM_SELECTED_NUM_ADAPTERS}_seed${SLLM_SAMPLING_SEED:-42}}"

# Keep the bearer token local; never print, commit, or place it in a result
# manifest.  The same private file is used by the qualified Prime runner.
if [[ -z "${PRIME_REMOTE_TOKEN:-}" && -r "/home/qhq/.config/primelora-tc-d75/artifact.token" ]]; then
  IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
  export PRIME_REMOTE_TOKEN
fi

if [[ -z "${VLLM_REMOTE_ARTIFACT_ENDPOINT}" ]]; then
  echo "[ERROR] no prepublished artifact endpoint for ${MODEL_PROFILE}" >&2
  exit 2
fi

exec bash "${ROOT_DIR}/scripts/run_vllm_fair_experiment.sh" "$@"
