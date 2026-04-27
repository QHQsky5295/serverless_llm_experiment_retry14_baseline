#!/usr/bin/env bash
set -euo pipefail

MAIN_REPO="${FAASLORA_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
BASELINES_ROOT="${FAASLORA_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RUNNER="${MAIN_REPO}/scripts/run_all_experiments_user_scope.sh"
PYTHON_BIN="${FAASLORA_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"

MODEL_PROFILE="${FAASLORA_PROFILE_MODEL:-llama2_7b_main_v2_publicmix}"
DATASET_PROFILE="${FAASLORA_PROFILE_DATASET:-azure_sharegpt_rep4000}"
WORKLOAD_PROFILE="${FAASLORA_PROFILE_WORKLOAD:-llama2_7b_auto500_formal4000_s8}"
TOTAL_REQUESTS="${FAASLORA_TOTAL_REQUESTS:-4000}"
SELECTED_NUM_ADAPTERS="${FAASLORA_SELECTED_NUM_ADAPTERS:-500}"
SAMPLING_SEED="${FAASLORA_SAMPLING_SEED:-42}"

SOURCE_RUN_TAG="${FAASLORA_SOURCE_RUN_TAG:-llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1}"
SOURCE_ROUND_DIR="${FAASLORA_SOURCE_ROUND_DIR:-${BASELINES_ROOT}/results/paper_experiments/03_main_comparison/20260424_104050_${SOURCE_RUN_TAG}}"
TRACE_PATH="${FAASLORA_SHARED_TRACE_PATH:-${SOURCE_ROUND_DIR}/shared_artifacts/${SOURCE_RUN_TAG}_trace.json}"
ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_ADAPTER_SUBSET_PATH:-${SOURCE_ROUND_DIR}/shared_artifacts/${SOURCE_RUN_TAG}_adapter_subset.json}"

RUN_TAG="${FAASLORA_PAPER_ABLATION_RUN_TAG:-llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1}"
SECTION_ID="${FAASLORA_PAPER_ABLATION_SECTION_ID:-04_ablation}"
ROUND_PURPOSE="${FAASLORA_PAPER_ABLATION_PURPOSE:-fig2_fig3_fig6_faaslora_cumulative_ablation_and_coordination}"
FIGURE_TARGETS="${FAASLORA_PAPER_ABLATION_FIGURES:-Fig2 Fig3 Fig6 CoordinationSubfigure}"
ROUND_ROOT="${FAASLORA_PAPER_ABLATION_ROOT:-${BASELINES_ROOT}/results/paper_experiments/${SECTION_ID}}"
ROUND_TIMESTAMP="${FAASLORA_PAPER_ABLATION_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ROUND_DIR="${FAASLORA_PAPER_ABLATION_ROUND_DIR:-${ROUND_ROOT}/${ROUND_TIMESTAMP}_${RUN_TAG}}"
SCENARIOS_RAW="${FAASLORA_PAPER_ABLATION_SCENARIOS:-faaslora_nvme faaslora_no_coord faaslora_full}"
FORCE_RERUN="${FAASLORA_PAPER_ABLATION_FORCE:-0}"
GPU_IDS="${FAASLORA_PAPER_ABLATION_GPU_IDS:-0,1,2,3}"
REQUIRE_GPU_IDLE="${FAASLORA_PAPER_ABLATION_REQUIRE_GPU_IDLE:-1}"
DRY_RUN="${FAASLORA_PAPER_ABLATION_DRY_RUN:-0}"
ALLOW_INTERNAL_BASELINES="${FAASLORA_PAPER_ABLATION_ALLOW_INTERNAL_BASELINES:-0}"

RAW_DIR="${ROUND_DIR}/raw/faaslora"
LOG_DIR="${ROUND_DIR}/logs"
STATE_DIR="${ROUND_DIR}/state"
SHARED_DIR="${ROUND_DIR}/shared_artifacts"

mkdir -p "${RAW_DIR}" "${LOG_DIR}" "${STATE_DIR}" "${SHARED_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

stage_done_path() {
  printf '%s/%s.done\n' "${STATE_DIR}" "$1"
}

is_done() {
  [[ "${FORCE_RERUN}" != "1" && -f "$(stage_done_path "$1")" ]]
}

mark_done() {
  date '+%F %T' >"$(stage_done_path "$1")"
}

sanitize_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

gpu_residual_pids() {
  local gpu_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  local output=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    if output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s\n' "${output}" | sed '/^[[:space:]]*$/d' | awk '{print $1}'
    fi
  done | sort -u
}

check_gpu_idle() {
  if [[ "${REQUIRE_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  local pids=()
  mapfile -t pids < <(gpu_residual_pids "${GPU_IDS}" || true)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  log "[ERROR] GPUs are not idle on ids=${GPU_IDS}; refusing to start a formal ablation stage."
  for pid in "${pids[@]}"; do
    ps -fp "${pid}" || true
  done
  log "Stop unrelated GPU jobs first, or set FAASLORA_PAPER_ABLATION_REQUIRE_GPU_IDLE=0 if this is intentional."
  return 1
}

validate_shared_artifacts() {
  "${PYTHON_BIN}" - "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
model_profile, dataset_profile, workload_profile = sys.argv[3:6]
total_requests = int(sys.argv[6])
selected_num_adapters = int(sys.argv[7])
sampling_seed = int(sys.argv[8])

if not trace_path.exists():
    raise SystemExit(f"shared trace artifact not found: {trace_path}")
if not subset_path.exists():
    raise SystemExit(f"shared adapter subset artifact not found: {subset_path}")

trace = json.loads(trace_path.read_text(encoding="utf-8"))
subset = json.loads(subset_path.read_text(encoding="utf-8"))

for field, expected in (
    ("model_profile", model_profile),
    ("dataset_profile", dataset_profile),
    ("workload_profile", workload_profile),
):
    if trace.get(field) != expected:
        raise SystemExit(f"trace {field} mismatch: expected {expected}, got {trace.get(field)}")
    if subset.get(field) != expected:
        raise SystemExit(f"subset {field} mismatch: expected {expected}, got {subset.get(field)}")

if len(trace.get("requests", [])) != total_requests:
    raise SystemExit(f"trace request count mismatch: expected {total_requests}, got {len(trace.get('requests', []))}")
if int(trace.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit("trace selected_num_adapters mismatch")
if int(subset.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit("subset selected_num_adapters mismatch")
if int(trace.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit("trace sampling_seed mismatch")
if int(subset.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit("subset sampling_seed mismatch")

subset_ids = {str(item["id"]) for item in subset.get("adapters", []) if "id" in item}
if len(subset_ids) != selected_num_adapters:
    raise SystemExit(f"subset adapter cardinality mismatch: expected {selected_num_adapters}, got {len(subset_ids)}")

trace_ids = {str(req.get("adapter_id")) for req in trace.get("requests", []) if req.get("adapter_id") is not None}
missing = sorted(trace_ids - subset_ids)
if missing:
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY
}

validate_scenarios() {
  local scenario=""
  local allowed_faaslora=" faaslora_nvme faaslora_no_coord faaslora_full "
  local allowed_internal=" cold_start slora_style serverlessllm "
  for scenario in "${SCENARIOS[@]}"; do
    [[ -z "${scenario}" ]] && continue
    if [[ "${allowed_faaslora}" == *" ${scenario} "* ]]; then
      continue
    fi
    if [[ "${ALLOW_INTERNAL_BASELINES}" == "1" && "${allowed_internal}" == *" ${scenario} "* ]]; then
      continue
    fi
    log "[ERROR] scenario=${scenario} is not allowed in this paper ablation script."
    log "Allowed by default: faaslora_nvme faaslora_no_coord faaslora_full."
    log "Internal legacy references require FAASLORA_PAPER_ABLATION_ALLOW_INTERNAL_BASELINES=1 and must not be mixed with official baseline claims."
    return 1
  done
}

validate_result_json() {
  local result_path="$1"
  local scenario="$2"
  "${PYTHON_BIN}" - "${result_path}" "${scenario}" "${TOTAL_REQUESTS}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
scenario = sys.argv[2]
expected_total = int(sys.argv[3])
obj = json.loads(path.read_text(encoding="utf-8"))

schema = obj.get("metric_schema_version")
if schema != "e2e_v3":
    raise SystemExit(f"{path}: metric_schema_version must be e2e_v3, got {schema!r}")

summaries = obj.get("scenario_summaries") or {}
if scenario not in summaries:
    raise SystemExit(f"{path}: missing scenario_summaries[{scenario!r}]")
summary = summaries[scenario]

total = int(summary.get("total_requests", -1))
completed = int(summary.get("completed_requests", -1))
failed = int(summary.get("failed_requests", 0) or 0)
if total != expected_total:
    raise SystemExit(f"{path}: total_requests mismatch: expected {expected_total}, got {total}")
if completed != total or failed != 0:
    raise SystemExit(f"{path}: invalid completion state total={total} completed={completed} failed={failed}")
if str(summary.get("backend", "")).lower() != "vllm":
    raise SystemExit(f"{path}: backend must be vllm, got {summary.get('backend')!r}")

required_metrics = [
    "avg_overall_ttft_ms",
    "p95_overall_ttft_ms",
    "avg_overall_e2e_ms",
    "p95_overall_e2e_ms",
    "avg_tpot_ms",
    "throughput_tok_per_s",
    "monetary_cost_per_request_usd",
    "monetary_ce",
]
for key in required_metrics:
    value = summary.get(key)
    if value is None:
        raise SystemExit(f"{path}: missing required metric {key}")
    try:
        f = float(value)
    except Exception as exc:
        raise SystemExit(f"{path}: non-numeric metric {key}={value!r}") from exc
    if not math.isfinite(f) or f < 0:
        raise SystemExit(f"{path}: invalid metric {key}={value!r}")

host_required = bool(summary.get("host_cache_memory_backed_required", False))
if scenario in {"faaslora_no_coord", "faaslora_full"}:
    host_required = True
if host_required and not bool(summary.get("host_cache_memory_backed", False)):
    raise SystemExit(
        f"{path}: HOST tier is not memory-backed; host_cache_memory_backed={summary.get('host_cache_memory_backed')!r}"
    )

details = obj.get("detailed_results") or {}
if scenario not in details:
    raise SystemExit(f"{path}: missing detailed_results[{scenario!r}]")
requests = details[scenario].get("requests") or []
if len(requests) != total:
    raise SystemExit(f"{path}: detailed request count mismatch: expected {total}, got {len(requests)}")

print(
    f"validated {scenario}: TTFT_avg={summary['avg_overall_ttft_ms']:.3f}ms "
    f"E2E_avg={summary['avg_overall_e2e_ms']:.3f}ms "
    f"Cost/req=${summary['monetary_cost_per_request_usd']:.6f} "
    f"CE={summary['monetary_ce']:.3f}"
)
PY
}

find_result_json() {
  local scenario="$1"
  local result_tag="$2"
  local sanitized_tag=""
  sanitized_tag="$(sanitize_label "${result_tag}")"
  # MAIN_REPO/results is a symlink in the retry14 workspace. Use -L so a
  # successfully written FaaSLoRA result is recoverable after a harness failure.
  find -L "${MAIN_REPO}/results" -type f -name "*${scenario}*${sanitized_tag}*.json" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

run_logged() {
  local stage="$1"
  shift
  local log_path="${LOG_DIR}/${stage}.log"
  log "stage=${stage} log=${log_path}"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    log "stage=${stage} failed status=${status}"
    return "${status}"
  fi
}

write_round_env() {
  {
    printf 'export FAASLORA_PAPER_ABLATION_ROUND_DIR=%q\n' "${ROUND_DIR}"
    printf 'export FAASLORA_PAPER_ABLATION_RUN_TAG=%q\n' "${RUN_TAG}"
    printf 'export FAASLORA_PAPER_ABLATION_SECTION_ID=%q\n' "${SECTION_ID}"
    printf 'export FAASLORA_PAPER_ABLATION_PURPOSE=%q\n' "${ROUND_PURPOSE}"
    printf 'export FAASLORA_PAPER_ABLATION_FIGURES=%q\n' "${FIGURE_TARGETS}"
    printf 'export FAASLORA_PROFILE_MODEL=%q\n' "${MODEL_PROFILE}"
    printf 'export FAASLORA_PROFILE_DATASET=%q\n' "${DATASET_PROFILE}"
    printf 'export FAASLORA_PROFILE_WORKLOAD=%q\n' "${WORKLOAD_PROFILE}"
    printf 'export FAASLORA_SOURCE_ROUND_DIR=%q\n' "${SOURCE_ROUND_DIR}"
    printf 'export FAASLORA_SOURCE_RUN_TAG=%q\n' "${SOURCE_RUN_TAG}"
    printf 'export FAASLORA_SHARED_TRACE_PATH=%q\n' "${TRACE_PATH}"
    printf 'export FAASLORA_SHARED_ADAPTER_SUBSET_PATH=%q\n' "${ADAPTER_SUBSET_PATH}"
    printf 'export FAASLORA_PAPER_ABLATION_SCENARIOS=%q\n' "${SCENARIOS_RAW}"
  } >"${ROUND_DIR}/round.env"
}

write_manifest() {
  "${PYTHON_BIN}" - \
    "${ROUND_DIR}" \
    "${RUN_TAG}" \
    "${SCENARIOS_RAW}" \
    "${TRACE_PATH}" \
    "${ADAPTER_SUBSET_PATH}" \
    "${SOURCE_ROUND_DIR}" \
    "${SOURCE_RUN_TAG}" \
    "${MODEL_PROFILE}" \
    "${DATASET_PROFILE}" \
    "${WORKLOAD_PROFILE}" \
    "${SECTION_ID}" \
    "${ROUND_PURPOSE}" \
    "${FIGURE_TARGETS}" \
    "${MAIN_REPO}" <<'PY'
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

round_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
scenarios = sys.argv[3].split()
trace_path = Path(sys.argv[4])
subset_path = Path(sys.argv[5])
source_round_dir = sys.argv[6]
source_run_tag = sys.argv[7]
model_profile = sys.argv[8]
dataset_profile = sys.argv[9]
workload_profile = sys.argv[10]
section_id = sys.argv[11]
purpose = sys.argv[12]
figure_targets = sys.argv[13].split()
main_repo = Path(sys.argv[14])
raw_dir = round_dir / "raw" / "faaslora"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_value(args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(main_repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

def build_consistency_audit(entries):
    metrics = {
        "ttft_avg_ms": ("lower", 0.02),
        "ttft_p95_ms": ("lower", 0.03),
        "e2e_avg_ms": ("lower", 0.02),
        "e2e_p95_ms": ("lower", 0.03),
        "cost_per_req_usd": ("lower", 0.03),
        "ce": ("higher", 0.03),
        "gpu_hit_rate": ("higher", 0.01),
        "avg_lora_io_ms": ("lower", 0.05),
    }
    by_scenario = {
        entry["scenario"]: entry
        for entry in entries
        if entry.get("exists") and entry.get("summary")
    }
    audit = {
        "baseline": "faaslora_full",
        "policy": (
            "Compare only scenarios from the same ablation round. "
            "Warnings are not automatic failures; they require explanation before plotting."
        ),
        "status": "incomplete",
        "warnings": [],
        "comparisons": [],
    }
    full = by_scenario.get("faaslora_full")
    if not full:
        audit["missing"] = ["faaslora_full"]
        return audit
    full_summary = full["summary"]
    audit["status"] = "ok"
    for scenario, entry in sorted(by_scenario.items()):
        if scenario == "faaslora_full":
            continue
        row = {"scenario": scenario, "metrics": {}}
        for metric, (direction, tolerance) in metrics.items():
            full_value = full_summary.get(metric)
            scenario_value = entry["summary"].get(metric)
            if full_value is None or scenario_value is None:
                continue
            try:
                full_f = float(full_value)
                scenario_f = float(scenario_value)
            except Exception:
                continue
            if full_f == 0:
                rel = None
            else:
                rel = (scenario_f - full_f) / abs(full_f)
            scenario_beats_full = (
                scenario_f < full_f if direction == "lower" else scenario_f > full_f
            )
            over_tolerance = False
            if rel is not None:
                over_tolerance = (
                    rel < -tolerance if direction == "lower" else rel > tolerance
                )
            metric_row = {
                "direction": direction,
                "full": full_f,
                "scenario": scenario_f,
                "relative_delta_vs_full": rel,
                "scenario_beats_full": scenario_beats_full,
                "warning": bool(scenario_beats_full and over_tolerance),
            }
            row["metrics"][metric] = metric_row
            if metric_row["warning"]:
                audit["warnings"].append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "direction": direction,
                        "full": full_f,
                        "scenario": scenario_f,
                        "relative_delta_vs_full": rel,
                        "tolerance": tolerance,
                    }
                )
        audit["comparisons"].append(row)
    if audit["warnings"]:
        audit["status"] = "warning"
    return audit

trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
entries = []
csv_rows = []
for scenario in scenarios:
    result = raw_dir / f"{run_tag}_{scenario}_result.json"
    source = raw_dir / f"{run_tag}_{scenario}_source_path.txt"
    entry = {"scenario": scenario, "result_json": str(result), "exists": result.exists()}
    if source.exists():
        entry["source_path"] = source.read_text(encoding="utf-8").strip()
    if result.exists():
        obj = json.loads(result.read_text(encoding="utf-8"))
        summary = (obj.get("scenario_summaries") or {}).get(scenario, {})
        entry["summary"] = {
            "ttft_avg_ms": summary.get("avg_overall_ttft_ms"),
            "ttft_p95_ms": summary.get("p95_overall_ttft_ms"),
            "e2e_avg_ms": summary.get("avg_overall_e2e_ms"),
            "e2e_p95_ms": summary.get("p95_overall_e2e_ms"),
            "tpot_ms": summary.get("avg_tpot_ms"),
            "tok_per_s": summary.get("throughput_tok_per_s"),
            "cost_per_req_usd": summary.get("monetary_cost_per_request_usd"),
            "ce": summary.get("monetary_ce"),
            "gpu_hit_rate": summary.get("gpu_hit_rate"),
            "avg_lora_io_ms": summary.get("avg_lora_io_ms"),
            "host_cache_memory_backed": summary.get("host_cache_memory_backed"),
        }
        csv_rows.append({"scenario": scenario, **entry["summary"]})
    else:
        csv_rows.append({"scenario": scenario})
    entries.append(entry)

manifest = {
    "run_tag": run_tag,
    "section_id": section_id,
    "purpose": purpose,
    "figure_targets": figure_targets,
    "round_dir": str(round_dir),
    "source_round_dir": source_round_dir,
    "source_run_tag": source_run_tag,
    "model_profile": model_profile,
    "dataset_profile": dataset_profile,
    "workload_profile": workload_profile,
    "code_snapshot": {
        "repo": str(main_repo),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_status_short": git_value(["status", "--short"]),
    },
    "shared_trace": {
        "path": str(trace_path),
        "sha256": sha256(trace_path),
        "requests": len(trace_payload.get("requests", [])),
        "selected_num_adapters": trace_payload.get("selected_num_adapters"),
        "sampling_seed": trace_payload.get("sampling_seed"),
        "active_adapter_cap": trace_payload.get("active_adapter_cap")
            or (trace_payload.get("load_profile") or {}).get("active_adapter_cap"),
        "hotset_rotation_requests": trace_payload.get("hotset_rotation_requests")
            or (trace_payload.get("load_profile") or {}).get("hotset_rotation_requests"),
    },
    "shared_adapter_subset": {
        "path": str(subset_path),
        "sha256": sha256(subset_path),
        "selected_num_adapters": subset_payload.get("selected_num_adapters"),
        "sampling_seed": subset_payload.get("sampling_seed"),
        "adapter_count": len(subset_payload.get("adapters", [])),
    },
    "scenarios": scenarios,
    "entries": entries,
}
(round_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
audit = build_consistency_audit(entries)
(round_dir / "ablation_consistency_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
csv_path = round_dir / "summary_metrics.csv"
fieldnames = [
    "scenario",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "tpot_ms",
    "tok_per_s",
    "cost_per_req_usd",
    "ce",
    "gpu_hit_rate",
    "avg_lora_io_ms",
    "host_cache_memory_backed",
]
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
print(round_dir / "MANIFEST.json")
PY
}

if [[ ! -x "${RUNNER}" ]]; then
  log "[ERROR] runner not found or not executable: ${RUNNER}"
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  log "[ERROR] Python not executable: ${PYTHON_BIN}"
  exit 1
fi

validate_shared_artifacts
cp -f "${TRACE_PATH}" "${SHARED_DIR}/$(basename "${TRACE_PATH}")"
cp -f "${ADAPTER_SUBSET_PATH}" "${SHARED_DIR}/$(basename "${ADAPTER_SUBSET_PATH}")"
write_round_env

read -r -a SCENARIOS <<< "${SCENARIOS_RAW}"
validate_scenarios
log "round_dir=${ROUND_DIR}"
log "run_tag=${RUN_TAG}"
log "section=${SECTION_ID} purpose=${ROUND_PURPOSE} figures=${FIGURE_TARGETS}"
log "scenarios=${SCENARIOS[*]}"
log "trace=${TRACE_PATH}"
log "adapter_subset=${ADAPTER_SUBSET_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for scenario in "${SCENARIOS[@]}"; do
    [[ -z "${scenario}" ]] && continue
    log "[dry-run] would run scenario=${scenario} result_tag=${RUN_TAG}_${scenario}"
  done
  write_manifest
  log "[dry-run] shared artifacts and round layout validated: ${ROUND_DIR}"
  exit 0
fi

for scenario in "${SCENARIOS[@]}"; do
  [[ -z "${scenario}" ]] && continue
  stage="scenario_${scenario}"
  result_tag="${RUN_TAG}_${scenario}"
  copied_result="${RAW_DIR}/${RUN_TAG}_${scenario}_result.json"

  if is_done "${stage}" && [[ -f "${copied_result}" ]]; then
    log "stage=${stage} already done; validating copied result and skipping"
    validate_result_json "${copied_result}" "${scenario}"
    continue
  fi

  recovered_result="$(find_result_json "${scenario}" "${result_tag}")"
  if [[ -n "${recovered_result}" && -f "${recovered_result}" ]]; then
    log "stage=${stage} has existing source result; validating and marking done without rerun"
    validate_result_json "${recovered_result}" "${scenario}"
    cp -f "${recovered_result}" "${copied_result}"
    printf '%s\n' "${recovered_result}" >"${RAW_DIR}/${RUN_TAG}_${scenario}_source_path.txt"
    mark_done "${stage}"
    continue
  fi

  check_gpu_idle
  log "running scenario=${scenario} result_tag=${result_tag}"
  (
    export FAASLORA_PROFILE_MODEL="${MODEL_PROFILE}"
    export FAASLORA_PROFILE_DATASET="${DATASET_PROFILE}"
    export FAASLORA_PROFILE_WORKLOAD="${WORKLOAD_PROFILE}"
    export FAASLORA_SHARED_TRACE_PATH="${TRACE_PATH}"
    export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}"
    export FAASLORA_RESULTS_TAG="${result_tag}"
    export PYTHONUNBUFFERED=1
    cd "${MAIN_REPO}"
    run_logged "${stage}" "${RUNNER}" \
      --config configs/experiments.yaml \
      --scenario "${scenario}" \
      --backend vllm \
      --model-profile "${MODEL_PROFILE}" \
      --dataset-profile "${DATASET_PROFILE}" \
      --workload-profile "${WORKLOAD_PROFILE}"
  )

  result_path="$(find_result_json "${scenario}" "${result_tag}")"
  if [[ -z "${result_path}" || ! -f "${result_path}" ]]; then
    log "[ERROR] unable to locate result JSON for scenario=${scenario} result_tag=${result_tag}"
    exit 1
  fi
  validate_result_json "${result_path}" "${scenario}"
  cp -f "${result_path}" "${copied_result}"
  printf '%s\n' "${result_path}" >"${RAW_DIR}/${RUN_TAG}_${scenario}_source_path.txt"
  mark_done "${stage}"
  log "stage=${stage} done result=${copied_result}"
done

write_manifest
log "FaaSLoRA paper ablation round complete: ${ROUND_DIR}"
