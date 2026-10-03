#!/usr/bin/env python3
"""Generate a PrimeLoRA-only control-path overhead audit.

This script intentionally does not compare control-path overhead across
different serving systems. Baselines expose different wrapper, runtime, and
admission boundaries, so the paper-facing audit is scoped to PrimeLoRA's
additional online decisions.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

OPERATIONS = [
    (
        "Routing +\ntier lookup",
        "routing_decision_us",
        "avg_routing_decision_us",
        "p95_routing_decision_us",
        False,
    ),
    (
        "Adapter-path\nresolution",
        "adapter_path_resolution_us",
        "avg_adapter_path_resolution_us",
        "p95_adapter_path_resolution_us",
        False,
    ),
    (
        "GPU-admission\ncheck",
        "gpu_admission_decision_us",
        "avg_gpu_admission_decision_us",
        "p95_gpu_admission_decision_us",
        True,
    ),
    (
        "Online control\ntotal",
        "control_path_total_us",
        "avg_control_path_total_us",
        "p95_control_path_total_us",
        False,
    ),
]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def percentile(values: Sequence[float], p: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    rank = max(0.0, min(1.0, p / 100.0)) * (len(finite) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return finite[lo]
    frac = rank - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def iter_json_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.rglob("*.json")):
        yield path


def load_payloads(input_path: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in iter_json_files(input_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            yield path, payload


def scenario_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    detailed = payload.get("detailed_results")
    if isinstance(detailed, dict):
        for scenario_name, scenario in detailed.items():
            if isinstance(scenario, dict):
                item = dict(scenario)
                item.setdefault("scenario_name", scenario_name)
                records.append(item)
    elif isinstance(detailed, list):
        for item in detailed:
            if isinstance(item, dict):
                records.append(dict(item))
    if not records and isinstance(payload.get("requests"), list):
        records.append(dict(payload))
    return records


def scenario_summary(payload: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    summaries = payload.get("scenario_summaries")
    if not isinstance(summaries, dict):
        return {}
    if scenario_name in summaries and isinstance(summaries[scenario_name], dict):
        return summaries[scenario_name]
    for item in summaries.values():
        if isinstance(item, dict) and item.get("scenario_name") == scenario_name:
            return item
    return {}


def select_scenario(
    payloads: Iterable[Tuple[Path, Dict[str, Any]]],
    scenario_filter: Optional[str],
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    candidates: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    for path, payload in payloads:
        for scenario in scenario_records(payload):
            name = str(scenario.get("scenario_name") or path.stem)
            baseline = str(scenario.get("baseline_type") or "").lower()
            if scenario_filter and scenario_filter.lower() not in name.lower():
                continue
            if scenario_filter is None and baseline not in {"faaslora_full", "primelora"}:
                continue
            summary = scenario_summary(payload, name)
            candidates.append((path, scenario, summary))
    if not candidates:
        raise SystemExit(
            "No PrimeLoRA/FaaSLoRA scenario with control-path fields found. "
            "Run a new PrimeLoRA result after the control-path instrumentation."
        )
    def score(candidate: Tuple[Path, Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int]:
        _, scenario, summary = candidate
        requests = scenario.get("requests") if isinstance(scenario.get("requests"), list) else []
        has_request_fields = int(
            any(_finite_float(req.get("control_path_total_us"), -1.0) >= 0.0 for req in requests)
        )
        has_summary_fields = int(_finite_float(summary.get("avg_control_path_total_us"), -1.0) >= 0.0)
        return has_request_fields + has_summary_fields, len(requests)
    return max(candidates, key=score)


def summarize_operation(
    scenario: Dict[str, Any],
    summary: Dict[str, Any],
    request_field: str,
    avg_field: str,
    p95_field: str,
    positive_only: bool = False,
) -> Tuple[float, float, int]:
    requests = scenario.get("requests")
    values = []
    if isinstance(requests, list):
        for req in requests:
            if not isinstance(req, dict) or not bool(req.get("success", True)):
                continue
            value = _finite_float(req.get(request_field), -1.0)
            if value < 0.0:
                continue
            if positive_only and value <= 0.0:
                continue
            values.append(value)
    if values:
        return sum(values) / len(values), percentile(values, 95), len(values)
    avg = _finite_float(summary.get(avg_field), 0.0)
    p95 = _finite_float(summary.get(p95_field), 0.0)
    count = int(_finite_float(summary.get("completed_requests"), 0.0))
    if avg <= 0.0 and p95 <= 0.0:
        raise SystemExit(
            f"Missing control-path field {request_field}; archived results cannot produce this audit."
        )
    return avg, p95, count


def summarize_background(summary: Dict[str, Any]) -> Tuple[float, float, int]:
    avg = _finite_float(summary.get("avg_background_planning_us"), 0.0)
    p95 = _finite_float(summary.get("p95_background_planning_us"), 0.0)
    count = int(_finite_float(summary.get("background_planning_event_count"), 0.0))
    return avg, p95, count


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Control-path overhead of PrimeLoRA. All values are in milliseconds. Online total is measured per request; GPU-admission and background-planning rows report triggered events.}",
        r"\label{tab:control_path_overhead}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Operation & Avg & P95 & Events \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['operation_latex']} & {row['avg_ms']:.3f} & {row['p95_ms']:.3f} & {int(row['events'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: List[Dict[str, Any]], output: Path, source: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "control_path_overhead",
        "scope": "PrimeLoRA-only diagnostic audit; not a cross-system comparison",
        "source": str(source),
        "operations": [
            {
                "operation": row["operation_label"].replace("\n", " "),
                "avg_ms": row["avg_ms"],
                "p95_ms": row["p95_ms"],
                "events": int(row["events"]),
            }
            for row in rows
        ],
        "field_note": (
            "Online control total is routing_decision_us + "
            "adapter_path_resolution_us + gpu_admission_decision_us. "
            "Background handoff planning is reported separately when present."
        ),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(rows: List[Dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["operation_label"]) for row in rows]
    avg_ms = [float(row["avg_ms"]) for row in rows]
    p95_ms = [float(row["p95_ms"]) for row in rows]
    y = list(range(len(rows)))
    y_avg = [idx - 0.055 for idx in y]
    y_p95 = [idx + 0.055 for idx in y]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 10.8,
        "xtick.labelsize": 10.2,
        "ytick.labelsize": 10.2,
        "legend.fontsize": 9.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.72))
    ax.hlines(y, avg_ms, p95_ms, color="#a8b4c4", linewidth=1.15, zorder=1)
    ax.scatter(avg_ms, y_avg, s=42, marker="o", color="#2b6cb0", label="Avg", zorder=3)
    ax.scatter(p95_ms, y_p95, s=48, marker="D", color="#c53030", label="P95", zorder=3)
    for idx, p95 in enumerate(p95_ms):
        ax.annotate(
            f"{p95:.3f}",
            (p95, y_p95[idx]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            color="#1f2937",
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Overhead (ms)")
    xmax = max(max(p95_ms), max(avg_ms), 0.01) * 1.28
    ax.set_xlim(left=0, right=xmax)
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.35, borderaxespad=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def provisional_timing_bound(rows, thresholds, *, offered_count):
    """Timing-only upper bound, never substitute for adapter correctness.

    All quantities are unrounded seconds. Success is an observed terminal, not
    z_r. Unknown correctness is deliberately retained as None. Missing successful
    measurements are errors; known failures contribute zero before timing checks.
    Threshold groups must be a contiguous, disjoint partition of positive input
    lengths. This does not freeze a common reference or qualify G1/G2.
    """
    def finite(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {name}')
        return value

    if type(offered_count) is not int or offered_count <= 0 or len(rows) != offered_count:
        raise ValueError('incomplete offered population')
    if len({r['request_id'] for r in rows}) != offered_count:
        raise ValueError('duplicate offered population')
    if not thresholds or thresholds[0]['lower_exclusive'] is not None or thresholds[-1]['upper_inclusive'] is not None:
        raise ValueError('threshold partition must cover all lengths')
    ids = set()
    for i, t in enumerate(thresholds):
        if t['group_id'] in ids:
            raise ValueError('duplicate threshold group')
        ids.add(t['group_id'])
        lo, hi = t['lower_exclusive'], t['upper_inclusive']
        if i and lo != thresholds[i-1]['upper_inclusive']:
            raise ValueError('noncontiguous threshold partition')
        if i and (type(lo) is not int or lo < 1):
            raise ValueError('invalid length boundary')
        if i < len(thresholds)-1 and (type(hi) is not int or hi <= (lo or 0)):
            raise ValueError('invalid length boundary')
        for name in ('ttft_s', 'tpot_s'):
            if finite(t[name], name) == 0:
                raise ValueError('zero threshold')
    decisions = []
    for r in rows:
        if type(r['success']) is not bool:
            raise ValueError('success must be boolean')
        d = dict(request_id=r['request_id'], outcome='failure', group_id=None,
                 ttft_pass=False, tpot_pass=None, timing_joint_pass=False)
        if r['success']:
            length, n = r['prompt_tokens'], r['token_count']
            if type(length) is not int or length < 1 or type(n) is not int or n < 1:
                raise ValueError('invalid native token count')
            t = next(t for t in thresholds if
                     (t['lower_exclusive'] is None or length > t['lower_exclusive']) and
                     (t['upper_inclusive'] is None or length <= t['upper_inclusive']))
            tp = finite(r['ttft_s'], 'TTFT') <= t['ttft_s']
            if n == 1:
                if r['tpot_s'] is not None:
                    raise ValueError('single-token TPOT must be N/A')
                pp = None
            else:
                pp = finite(r['tpot_s'], 'TPOT') <= t['tpot_s']
            joint = tp and pp is not False
            outcome = ('timing_pass' if joint else 'both_miss' if not tp and pp is False
                       else 'ttft_only_miss' if not tp else 'tpot_only_miss')
            d.update(group_id=t['group_id'], outcome=outcome, ttft_pass=tp,
                     tpot_pass=pp, timing_joint_pass=joint)
        decisions.append(d)
    counts = Counter(d['outcome'] for d in decisions)
    eligible = [d for d in decisions if d['tpot_pass'] is not None]
    return dict(kind='provisional_timing_only_upper_bound_v1', offered_count=offered_count,
                correctness_qualified=False, n_correct=None, formal_joint_slo=None,
                formal_g1_g2_qualified=False,
                timing_joint_pass_count=counts['timing_pass'],
                joint_upper_bound=counts['timing_pass']/offered_count,
                ttft_upper_bound=sum(d['ttft_pass'] for d in decisions)/offered_count,
                timing_tpot_eligible_count=len(eligible),
                timing_tpot_conditional_rate=(sum(d['tpot_pass'] for d in eligible)/len(eligible)
                                              if eligible else None),
                required_joint_count=math.ceil(0.95*offered_count),
                minimum_additional_timing_passes=max(0, math.ceil(0.95*offered_count)-counts['timing_pass']),
                outcome_counts={k: counts[k] for k in ('timing_pass', 'ttft_only_miss',
                                'tpot_only_miss', 'both_miss', 'failure')}, decisions=decisions)


def analyze_native_timeline(projection: Path, deployment_path: Path,
                            terminals_path: Path, watchdog_path: Path,
                            output: Path, *, allow_failed: bool = False,
                            control_outcome_path: Optional[Path] = None) -> Dict[str, Any]:
    """Audit a completed bounded projection, without loading the huge source tree.

    These are request occupancy intervals, NOT GPU resource billing. Native
    last-token/controller completion/outer terminal are distinct observations.
    No extrapolation, missing-field defaults, legacy percentile or paper copy.
    """
    import hashlib
    if output.exists():
        raise FileExistsError(output)
    if projection.stat().st_size >= 256 * 1024**2:
        raise ValueError('bounded request projection required')
    rows = json.loads(projection.read_text())['requests']
    deployment = json.loads(deployment_path.read_text())
    with terminals_path.open() as handle:
        terminals = [json.loads(line) for line in handle]
    count = deployment['plan']['count']
    by_id = {r['request_id']: r for r in terminals}
    if (len(rows) != count or len(terminals) != count or len(by_id) != count
            or {r['request_id'] for r in rows} != set(by_id)
            or any(type(r['success']) is not bool or r['success'] != by_id[r['request_id']]['success'] for r in rows)
            or any(type(r['success']) is not bool or (r['success'] and r['native_contract_matched'] is not True) for r in terminals)
            or (not allow_failed and any(r['success'] is not True for r in rows))):
        raise ValueError('complete native-contract population required')
    if any(r['clock_id'] != deployment['clock_id'] for r in terminals):
        raise ValueError('mixed timing domains')
    start = deployment['arrival_start_s']
    end = max(r['at'] for r in terminals)
    if count <= 0 or not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('nonempty finite observation window required')
    phase_names = ('arrival_to_gate', 'gate_to_source_admission',
                   'source_to_native_dispatch', 'native_dispatch_to_last_token',
                   'last_token_to_controller_completion', 'controller_completion_to_terminal')
    intervals = {name: [] for name in phase_names}
    intervals['gate_to_terminal'] = []
    native_by_replica = {}
    request_rows = []
    failures = []
    for row in rows:
        terminal = by_id[row['request_id']]
        arrival = start + row['scheduled_arrival_offset_s']
        if not math.isfinite(arrival) or not start <= arrival <= terminal['at'] <= end:
            raise ValueError('invalid request terminal interval')
        if not row['success']:
            observation = row['failure_observation']
            if observation['clock_id'] != deployment['clock_id']:
                raise ValueError('mixed failure timing domains')
            observed = observation['observed_monotonic_s']
            if not math.isfinite(observed) or observed < arrival:
                raise ValueError('invalid failure observation interval')
            # Task-exception collection happens AFTER the outer finally emits
            # terminal; returned execution errors are built BEFORE that finally.
            kind = observation['kind']
            if ((kind == 'controller_task_exception_v1' and observed < terminal['at'])
                    or (kind == 'native_request_execution_error_v1' and observed > terminal['at'])
                    or kind not in ('controller_task_exception_v1','native_request_execution_error_v1')):
                raise ValueError('failure observation does not match its producer boundary')
            failures.append(dict(request_id=row['request_id'], exception_type=observation['exception_type'],
                observation_kind=kind,observed_at=observed,terminal_at=terminal['at'],
                observation_minus_terminal_s=observed-terminal['at']))
            # Default zero/empty stages in failed results are NOT measurements.
            request_rows.append(dict(request_id=row['request_id'], instance_id=row.get('instance_id'),
                arrival_offset_s=arrival-start, terminal_offset_s=terminal['at']-start,
                **{name+'_s':None for name in phase_names}, recorded_e2e_s=None,
                outer_terminal_e2e_s=terminal['at']-arrival, success=False))
            continue
        t = row['native_token_timing']
        if t['native_clock_id'] != deployment['clock_id'] or terminal['clock_id'] != deployment['clock_id']:
            raise ValueError('mixed timing domains')
        if terminal['instance_id'] != row['instance_id']:
            raise ValueError('terminal replica identity mismatch')
        gate = start + row['arrival_released_offset_s'] + row['dispatch_window_wait_ms']/1000
        bounds = (arrival, gate, t['controller_admitted_monotonic_s'],
                  t['native_dispatch_monotonic_s'], t['native_last_token_monotonic_s'],
                  t['controller_completed_monotonic_s'], terminal['at'])
        if bounds[0] < start or any(not math.isfinite(v) for v in bounds) or any(a > b for a,b in zip(bounds,bounds[1:])):
            raise ValueError(('invalid ordered native intervals', row['request_id'], bounds))
        record = dict(request_id=row['request_id'], instance_id=row['instance_id'],
                      arrival_offset_s=arrival-start, terminal_offset_s=terminal['at']-start)
        for name, left, right in zip(phase_names, bounds, bounds[1:]):
            intervals[name].append((left,right))
            record[name+'_s'] = right-left
        intervals['gate_to_terminal'].append((gate,terminal['at']))
        native_by_replica.setdefault(row['instance_id'], []).append((bounds[3],bounds[4]))
        record['recorded_e2e_s'] = row['overall_e2e_ms']/1000
        record['outer_terminal_e2e_s'] = terminal['at']-arrival
        record['success'] = True
        request_rows.append(record)

    def summary(spans):
        lengths = sorted(b-a for a,b in spans)
        events = sorted([(a,1) for a,b in spans if a < b] + [(b,-1) for a,b in spans if a < b])
        active = peak = 0
        for _,delta in events:
            active += delta
            peak = max(peak,active)
        if active != 0:
            raise ValueError('unclosed occupancy events')
        area = math.fsum(lengths)
        return dict(n=len(lengths), total_request_seconds=area,
                    mean_duration_s=area/len(lengths) if lengths else None,
                    p95_duration_s=lengths[math.ceil(.95*len(lengths))-1] if lengths else None,
                    mean_concurrent_requests=area/(end-start), max_concurrent_requests=peak)

    indexes = {name:(sorted(a for a,b in spans),sorted(b for a,b in spans)) for name,spans in intervals.items()}
    samples = []
    with watchdog_path.open() as handle:
        for line in handle:
            sample = json.loads(line)
            if sample['event'] != 'resource_sample' or not start <= sample['monotonic'] <= end:
                continue
            at = sample['monotonic']
            held = set(sample['gpu']['service_held_gpu_uuids'])
            rates = [d['gpu_utilization_percent'] for d in sample['gpu']['devices'] if d['gpu_uuid'] in held]
            if len(rates) != len(held) or any(type(v) not in (float,int) or not math.isfinite(v) for v in rates):
                raise ValueError('missing held-GPU activity observations')
            record = dict(offset_s=at-start, held_gpus=len(held),
                          held_gpu_utilization_mean_percent=sum(rates)/len(rates) if rates else None)
            record.update({name:bisect_right(left,at)-bisect_right(right,at) for name,(left,right) in indexes.items()})
            samples.append(record)
    if not samples:
        raise ValueError('no overlapping resource observations')
    gpu_rates = [r['held_gpu_utilization_mean_percent'] for r in samples if r['held_gpus']]
    if not gpu_rates:
        raise ValueError('no held-GPU resource observations')
    result = dict(kind='native_control_occupancy_audit_v1', formal_performance_result=False,
        count=count, observation_s=end-start, phase_summaries={k:summary(v) for k,v in intervals.items()},
        population=dict(planned=count,terminal=count,native_success=count-len(failures),failed=len(failures)),
        phase_population='native_success_only' if failures else 'complete_native_success_population',
        failures=failures,
        native_by_replica={k:summary(v) for k,v in native_by_replica.items()},
        resource_samples=len(samples), sampled_mean_held_gpu_utilization_percent=sum(gpu_rates)/len(gpu_rates),
        samples_with_pre_gate_backlog=sum(r['arrival_to_gate']>0 for r in samples),
        sampled_gate_occupancy_counts=dict(Counter(r['gate_to_terminal'] for r in samples)),
        caveats=['Occupancy is not GPU billing or kernel busy time.',
                 'Gate end uses outer terminal after release; this is an upper envelope, not an instrumented gate-release timestamp.',
                 'Controller completion is an earlier boundary than outer request terminal; report their gap without relabeling either.',
                 'GPU utilization averages are sampled descriptive observations, not causal attribution or confidence intervals.'])
    if failures:
        result['caveats'].append('All offered IDs and failures are retained. Phase intervals and reconstructed occupancy cover ONLY native successes, not all active work. Failed work may persist beyond its request terminal; absent native events are not proof of no dispatch or release.')
    paths = [projection,deployment_path,terminals_path,watchdog_path,Path(__file__)]
    controls = []
    if control_outcome_path is not None:
        if control_outcome_path.stat().st_size >= 128 * 1024**2:
            raise ValueError('bounded normal control outcome required')
        outcome = json.loads(control_outcome_path.read_text())
        events = outcome['mechanism_events']['_ieee_control_events']
        quarantines = outcome['mechanism_events']['_ieee_runtime_quarantine_events']
        if not events or any(q['clock_id'] != deployment['clock_id'] for q in quarantines):
            raise ValueError('missing control events or mixed quarantine clocks')
        first_failure = min((f['terminal_at'] for f in failures), default=None)
        first_failure_observation = min((f['observed_at'] for f in failures), default=None)
        first_quarantine = min((q['started_monotonic_s'] for q in quarantines), default=None)
        # Control sampling and request completion have distinct lifetimes.
        # Validate the entire chronological source history, then retain samples
        # outside the request window separately rather than moving its boundary.
        previous_at = -math.inf
        outside_window = []
        for index, event in enumerate(events):
            at = event['observed_at']
            if not math.isfinite(at) or at < previous_at:
                raise ValueError('invalid control observation order')
            previous_at = at
            names = ('queue_depth','active_requests','ready_capacity','ready_instances','pending_instances')
            if any(type(event[k]) is not int or event[k] < 0 for k in names):
                raise ValueError('invalid observed control counts')
            if event['active_requests'] > event['ready_capacity']:
                raise ValueError('observed active requests exceed ready capacity')
            if not start <= at <= end:
                outside_window.append(dict(source_index=index, offset_s=at-start,
                    boundary='before_request_window' if at < start else 'after_request_window',
                    event=event))
                continue
            phase = ('after_first_quarantine' if first_quarantine is not None and at >= first_quarantine
                     else 'after_first_failure_before_quarantine' if first_failure is not None and at >= first_failure
                     else 'before_first_failure')
            controls.append(dict(offset_s=at-start,phase=phase,**{k:event[k] for k in names},
                action=event['action'],outcome=event['outcome'],
                successful_native_occupancy=bisect_right(indexes['native_dispatch_to_last_token'][0],at)
                    -bisect_right(indexes['native_dispatch_to_last_token'][1],at)))
        result['control_observations'] = dict(
            source_event_count=len(events), in_request_window_count=len(controls),
            outside_request_window=outside_window,
            first_failed_terminal_offset_s=first_failure-start if first_failure is not None else None,
            first_failure_observation_offset_s=first_failure_observation-start if first_failure_observation is not None else None,
            first_quarantine_offset_s=first_quarantine-start if first_quarantine is not None else None,
            by_phase={phase:dict(samples=len(group),
                mean_active_requests=math.fsum(e['active_requests'] for e in group)/len(group),
                mean_queue_depth=math.fsum(e['queue_depth'] for e in group)/len(group),
                positive_queue_samples=sum(e['queue_depth'] > 0 for e in group),
                positive_queue_below_capacity_samples=sum(e['queue_depth'] > 0 and e['active_requests'] < e['ready_capacity'] for e in group),
                active_count_histogram=dict(Counter(e['active_requests'] for e in group)),
                ready_capacity_histogram=dict(Counter(e['ready_capacity'] for e in group)))
                for phase in sorted({e['phase'] for e in controls})
                for group in [[e for e in controls if e['phase']==phase]]},
            caveat='Direct controller samples, not continuous occupancy or kernel activity. All source samples are validated; outside-request-window samples are retained separately and excluded from by_phase and the in-window CSV. active_requests counts bound requests in currently routable slots; queue_depth also includes requests outside those slots. Successful native occupancy is conditional; never subtract it to infer all non-native work.')
        paths.append(control_outcome_path)
    result['source_refs'] = []
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):
                h.update(chunk)
        result['source_refs'].append(dict(path=str(path),sha256=h.hexdigest()))
    output.mkdir(parents=True,exist_ok=False)
    write_csv(request_rows,output/'request_occupancy.csv')
    write_csv(samples,output/'sampled_occupancy.csv')
    if controls:
        write_csv(controls,output/'control_observations.csv')
    with (output/'summary.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    return result


def analyze_rpc_breakdown(projection: Path, deployment_path: Path,
                          terminals_path: Path, sealed_summary: Path,
                          sealed_sha256: str, output: Path) -> Dict[str, Any]:
    """Recover retained RPC observations, not a new performance measurement.

    Pin the previous curator and only read its bounded, SHA-verified sources.
    Unknown/invalid diagnostic values are counted, never imputed as zero.
    Wall-clock and clamped producer measurements are explicitly descriptive.
    """
    import hashlib

    if output.exists():
        raise FileExistsError(output)

    def bounded_bytes(path):
        if path.stat().st_size >= 256 * 1024**2:
            raise ValueError('bounded input required')
        return path.read_bytes()

    sealed_bytes = bounded_bytes(sealed_summary)
    if hashlib.sha256(sealed_bytes).hexdigest() != sealed_sha256:
        raise ValueError('sealed summary SHA mismatch')
    sealed = json.loads(sealed_bytes)
    refs = {str(Path(r['path']).resolve()): r['sha256'] for r in sealed['source_refs']}
    inputs = []
    payloads = []
    for path in (projection, deployment_path, terminals_path):
        data = bounded_bytes(path)
        sha = hashlib.sha256(data).hexdigest()
        if refs.get(str(path.resolve())) != sha:
            raise ValueError('input not pinned by sealed summary')
        inputs.append(dict(path=str(path.resolve()), sha256=sha))
        payloads.append(data)
    rows = json.loads(payloads[0])['requests']
    deployment = json.loads(payloads[1])
    terminals = [json.loads(line) for line in payloads[2].splitlines()]
    count = deployment['plan']['count']
    by_id = {r['request_id']: r for r in terminals}
    if (type(count) is not int or count <= 0 or len(rows) != count
            or len(terminals) != count or len(by_id) != count
            or {r['request_id'] for r in rows} != set(by_id)
            or not sealed['native_contract_completion_pass']
            or sealed['population']['native_contract_matched'] != count):
        raise ValueError('complete native population required')
    # These are the actual producers' names; the former parent_response_* names
    # do not exist in RequestResult. Neither aliasing nor defaulting is allowed.
    rpc_names = ('worker_rpc_handler_wall_ms', 'worker_rpc_queue_ms',
                 'parent_rpc_wall_ms', 'parent_rpc_overhead_ms',
                 'parent_rpc_channel_acquire_ms',
                 'parent_rpc_response_pickup_delay_ms',
                 'parent_rpc_thread_resume_delay_ms')
    native_names = (*rpc_names, 'parent_rpc_send_flush_ms',
                    'parent_rpc_wait_response_ms', 'parent_rpc_request_bytes',
                    'parent_rpc_response_bytes', 'admission_to_engine_dispatch_ms',
                    'engine_entry_to_queue_ms', 'native_engine_queue_ms',
                    'native_prefill_ms', 'native_decode_ms',
                    'worker_completion_notification_ms',
                    'worker_to_controller_completion_ms')
    outer_names = ('routing_decision_us', 'adapter_path_resolution_us',
                   'gpu_admission_decision_us', 'control_path_total_us')
    observations = {name: [] for name in (*native_names, *outer_names)}
    coverage = {name: Counter() for name in observations}
    invalid_ids = {name: [] for name in observations}
    diagnostics = []
    for row in sorted(rows, key=lambda r: r['request_id']):
        terminal = by_id[row['request_id']]
        timing = row['native_token_timing']
        if (row['success'] is not True or terminal['success'] is not True
                or terminal['native_contract_matched'] is not True
                or row['output_contract_match'] is not True
                or row['generation_contract'] != 'fixed_length_greedy_v1'
                or row['completion_token_source'] != 'vllm_token_ids'
                or timing['native_terminal_observed'] is not True
                or timing['timing_contract'] != 'ieee_tc_native_v1'
                or timing['parent_rpc_transport'] != 'native_async_socket_v1'
                or terminal['clock_id'] != deployment['clock_id']
                or timing['native_clock_id'] != deployment['clock_id']
                or terminal['instance_id'] != row['instance_id']):
            raise ValueError('native timing/transport/identity contract mismatch')
        for name in rpc_names:
            if name in timing and name in row and timing[name] != row[name]:
                raise ValueError('outer/native RPC values disagree')
        record = dict(request_id=row['request_id'], instance_id=row['instance_id'])
        for name in observations:
            source = timing if name in native_names else row
            value = source.get(name)
            status = ('missing' if name not in source else 'null' if value is None
                      else 'invalid' if type(value) not in (int, float)
                      or not math.isfinite(value) or value < 0 else 'valid')
            coverage[name][status] += 1
            if status == 'valid':
                observations[name].append(value)
                coverage[name]['zero'] += int(value == 0)
            else:
                invalid_ids[name].append(row['request_id'])
            record[name] = value if status == 'valid' else None
        diagnostics.append(record)
    metrics = []
    for name, values in observations.items():
        values.sort()
        unit = 'bytes' if name.endswith('_bytes') else 'us' if name.endswith('_us') else 'ms'
        metrics.append(dict(field=name, unit=unit, offered=count,
            **{key: coverage[name][key] for key in ('valid','missing','null','invalid','zero')},
            mean=math.fsum(values)/len(values) if values else None,
            p50=values[math.ceil(.5*len(values))-1] if values else None,
            p95=values[math.ceil(.95*len(values))-1] if values else None,
            max=max(values) if values else None))
    result = dict(kind='retained_rpc_observation_audit_v1', formal_performance_result=False,
        new_experiment=False, population=dict(planned=count, terminal=count, native_success=count),
        complete_field_coverage=all(c['valid'] == count for c in coverage.values()),
        quantile='Type-1', metric_protocol_sha256=sealed['metric_protocol_sha256'],
        metrics=metrics, invalid_or_absent_request_ids=invalid_ids,
        source_refs=[*inputs,dict(path=str(sealed_summary.resolve()),sha256=sealed_sha256),
                     dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())],
        caveats=[
            'Native contract completion is not numerical adapter or common-SLO qualification.',
            'Pickup and worker RPC queue use same-host wall timestamps and clamp negative differences; raw parent read timestamps are not retained, so clock effects cannot be reconstructed.',
            'Pickup spans response serialization/loopback/event-loop read availability; it is neither remote artifact network delay nor an isolated CPU function duration.',
            'Native async transport sets thread resume to structural zero; this is not an independently measured absence of scheduling delay.',
            'Legacy outer control timers can be structural/default zero; a recorded zero is not proof of zero mechanism overhead.',
            'RPC metrics are generation-call observations, not durations of source_snapshot, preparation or retirement RPCs.',
            'Overlapping RPC/native/control spans and their percentiles must not be summed as disjoint phases.',
            'Missing/invalid fields retain their request IDs and counts; available-value summaries are diagnostic only.'])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(metrics, output/'rpc_metrics.csv')
    write_csv(diagnostics, output/'request_rpc_observations.csv')
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def summarize_admission_capacity(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Describe retained admission snapshots, never infer continuous KV use.

    These records are selected by preparation/admission events. A repeated
    snapshot is counted once; its multiplicity is retained. Missing scheduler
    pool/preemption fields cannot be recovered from the admission projection.
    """
    snapshots = {}
    occurrences = 0
    pending = [outcome['mechanism_events']]
    required = ('replica_id', 'epoch', 'captured_at', 'admitted',
                'kv_tokens_per_block', 'kv_bytes_per_block',
                'kv_unreserved_free_blocks', 'scheduled_tokens',
                'iteration_token_budget')
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if 'kv_unreserved_free_blocks' in node:
                if any(key not in node for key in required):
                    raise ValueError('incomplete retained admission snapshot')
                if not isinstance(node['replica_id'], str) or not node['replica_id']:
                    raise ValueError('missing replica identity')
                if type(node['epoch']) is not int or node['epoch'] < 0:
                    raise ValueError('invalid admission epoch')
                if (type(node['captured_at']) not in (int, float)
                        or not math.isfinite(node['captured_at'])):
                    raise ValueError('invalid snapshot time')
                for key in required[4:]:
                    if type(node[key]) is not int or node[key] < 0:
                        raise ValueError('invalid native capacity value')
                if min(node['kv_tokens_per_block'], node['kv_bytes_per_block'],
                       node['iteration_token_budget']) <= 0:
                    raise ValueError('zero native capacity unit')
                if node['scheduled_tokens'] > node['iteration_token_budget']:
                    raise ValueError('native token budget exceeded')
                admitted = node['admitted']
                if not isinstance(admitted, list):
                    raise ValueError('missing admitted demand population')
                ids = [r['request_id'] for r in admitted]
                if len(ids) != len(set(ids)):
                    raise ValueError('duplicate admitted demand')
                key = (node['replica_id'], node['epoch'], node['captured_at'])
                if key in snapshots and snapshots[key] != node:
                    raise ValueError('conflicting duplicate snapshot identity')
                snapshots[key] = node
                occurrences += 1
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    if not snapshots:
        raise ValueError('no retained admission snapshots; capacity unknown')
    rows = []
    for replica in sorted({key[0] for key in snapshots}):
        group = [s for key, s in snapshots.items() if key[0] == replica]
        geometry = {(s['kv_tokens_per_block'], s['kv_bytes_per_block'],
                     s['iteration_token_budget']) for s in group}
        if len(geometry) != 1:
            raise ValueError('capacity geometry changed within replica')
        tokens, size, budget = geometry.pop()
        free = sorted(s['kv_unreserved_free_blocks'] for s in group)
        rows.append(dict(replica_id=replica, distinct_snapshots=len(group),
            first_captured_at=min(s['captured_at'] for s in group),
            last_captured_at=max(s['captured_at'] for s in group),
            kv_tokens_per_block=tokens, kv_bytes_per_block=size,
            iteration_token_budget=budget, free_blocks_min=free[0],
            free_blocks_p50=free[math.ceil(len(free)*.5)-1],
            free_blocks_max=free[-1],
            admitted_demand_max=max(len(s['admitted']) for s in group),
            scheduled_tokens_max=max(s['scheduled_tokens'] for s in group)))
    return dict(kind='retained_admission_capacity_audit_v1',
        source_occurrences=occurrences, distinct_snapshots=len(snapshots),
        duplicate_occurrences=occurrences-len(snapshots), replicas=rows,
        continuous_kv_exhaustion=None, preemption_total=None,
        qualified_runtime_capacity=None,
        caveat='Event-selected admission snapshots, not continuous scheduler samples. '
            'Admitted includes controller-pending demand, not only executing sequences. '
            'Free blocks are not total pool capacity. No safety or speedup claim for a new configuration.')


def analyze_admission_capacity(source: Path, output: Path) -> Dict[str, Any]:
    import hashlib
    if source.stat().st_size >= 128 * 1024**2:
        raise ValueError('bounded normal control outcome required')
    if output.exists():
        raise FileExistsError(output)
    encoded = source.read_bytes()
    result = summarize_admission_capacity(json.loads(encoded))
    result['source_ref'] = dict(path=str(source), sha256=hashlib.sha256(encoded).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    write_csv(result['replicas'], output/'capacity_by_replica.csv')
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


CONTROL_BOUNDARIES = (
    'parent_begin', 'parent_send', 'worker_received', 'frontend_begin',
    'frontend_native_send', 'native_begin', 'native_ready',
    'frontend_native_received', 'frontend_ready', 'parent_received', 'parent_terminal',
)


def analyze_control_boundaries(directory: Path, receipt_path: Path,
                               deployment_path: Path, output: Path) -> Dict[str, Any]:
    """Join diagnostic RPC boundaries, never infer queue-only or request latency.

    The producer writes one scalar event at a time under the qualified prefix
    gate. Read bounded lines; retain incomplete/cancelled attempts and partial
    final lines. Only complete successful chains enter interval summaries.
    """
    import hashlib
    import stat
    import uuid

    if output.exists():
        raise FileExistsError(output)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('diagnostic directory must be a real directory')
    if directory.resolve() != receipt_path.resolve().parent / 'diagnostic_stacks':
        raise ValueError('diagnostic directory must belong to receipt')
    source_refs = []

    def bounded_json(path, limit=4*1024**2):
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('regular nonsymlink input required')
        with path.open('rb') as handle:
            data = handle.read(limit+1)
        if len(data) > limit:
            raise ValueError('bounded JSON input required')
        source_refs.append(dict(path=str(path.resolve()), bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest()))
        return json.loads(data)

    def positive_int(value):
        return type(value) is int and value > 0

    def finite(value, positive=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))

    def owned_path(value, owner):
        path = Path(value)
        return (path.is_absolute() and '..' not in path.parts
                and path.is_relative_to(owner))

    receipt = bounded_json(receipt_path)
    deployment = bounded_json(deployment_path)
    replay = receipt.get('external_replay', {})
    owner = Path(receipt['service_identity']['path'])
    count = replay.get('diagnostic_prefix_count')
    clock_id, start = deployment.get('clock_id'), deployment.get('arrival_start_s')
    if (receipt.get('allow_exec') is not True or not positive_int(count)
            or replay.get('replay_scope') != 'diagnostic_prefix_v1'
            or not owned_path(str(owner), Path('/sys/fs/cgroup'))
            or owner == Path('/sys/fs/cgroup')
            or deployment.get('contract') != 'physical_gpu_deployment_v1'
            or type(deployment.get('plan', {}).get('count')) is not int
            or deployment['plan']['count'] != count
            or not isinstance(clock_id, str) or not clock_id or len(clock_id) > 256
            or not finite(start, positive=True)):
        raise ValueError('qualified prefix receipt/deployment required')

    metadata_paths = sorted(directory.glob('*.json'))
    event_paths = sorted(directory.glob('*.control.jsonl'))
    if (not metadata_paths or len(metadata_paths) > 256
            or {p.stem for p in metadata_paths}
            != {p.name.removesuffix('.control.jsonl') for p in event_paths}):
        raise ValueError('complete bounded metadata/event file population required')
    by_attempt, partial_lines = {}, []
    event_count, total_bytes = 0, 0
    expected_keys = {'event','attempt_id','boundary','pid','thread_ident','clock_id',
                     'monotonic_s','thread_cpu_s','byte_count','outcome'}
    for meta_path in metadata_paths:
        meta = bounded_json(meta_path, 64*1024)
        if (meta.get('kind') != 'diagnostic_python_frames_v1'
                or meta.get('formal_performance_result') is not False
                or meta.get('control_boundaries') != 'request_source_control_boundary_v1'
                or meta.get('control_overhead_included') is not True
                or not positive_int(meta.get('pid')) or not positive_int(meta.get('start_ticks'))
                or meta_path.stem != f"{meta['pid']}-{meta['start_ticks']}"
                or not owned_path(meta['cgroup'], owner)
                or not finite(meta.get('capture_start_monotonic_s'), positive=True)):
            raise ValueError('qualified process metadata required')
        path = directory / (meta_path.stem+'.control.jsonl')
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('regular nonsymlink event input required')
        before = path.stat()
        total_bytes += before.st_size
        if total_bytes > 128*1024**2:
            raise ValueError('bounded total event bytes exceeded')
        digest, size, previous_by_thread = hashlib.sha256(), 0, {}
        with path.open('rb') as handle:
            line_number = 0
            while True:
                line = handle.readline(8193)
                if not line:
                    break
                line_number += 1
                if len(line) > 8192:
                    raise ValueError('bounded event line exceeded')
                digest.update(line)
                size += len(line)
                if not line.endswith(b'\n'):
                    partial_lines.append(dict(path=str(path.resolve()), line=line_number, bytes=len(line)))
                    break
                event_count += 1
                if event_count > 262144:
                    raise ValueError('bounded event population exceeded')
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != expected_keys:
                    raise ValueError('invalid control event schema')
                aid, boundary = row['attempt_id'], row['boundary']
                if (not isinstance(aid, str) or len(aid) != 32 or uuid.UUID(aid).hex != aid
                        or boundary not in CONTROL_BOUNDARIES
                        or row['event'] != 'request_source_control_boundary_v1'
                        or type(row['pid']) is not int or row['pid'] != meta['pid']
                        or not positive_int(row['thread_ident']) or row['clock_id'] != clock_id
                        or not finite(row['monotonic_s'], positive=True)
                        or row['monotonic_s'] < meta['capture_start_monotonic_s']
                        or not finite(row['thread_cpu_s'])):
                    raise ValueError('invalid control event identity/clock/time')
                if boundary in ('parent_send', 'parent_received'):
                    if type(row['byte_count']) is not int or row['byte_count'] < 0:
                        raise ValueError('invalid byte measurement')
                elif row['byte_count'] is not None:
                    raise ValueError('unexpected byte measurement')
                if boundary == 'parent_terminal':
                    if row['outcome'] not in ('success', 'error', 'cancelled'):
                        raise ValueError('invalid terminal outcome')
                elif row['outcome'] is not None:
                    raise ValueError('unexpected terminal outcome')
                previous = previous_by_thread.get(row['thread_ident'])
                if previous and (row['monotonic_s'] < previous[0] or row['thread_cpu_s'] < previous[1]):
                    raise ValueError('nonmonotonic thread event stream')
                previous_by_thread[row['thread_ident']] = (row['monotonic_s'], row['thread_cpu_s'])
                attempt = by_attempt.setdefault(aid, {})
                if boundary in attempt:
                    raise ValueError('duplicate attempt boundary')
                # Process incarnation, not PID alone, protects against reuse.
                row['process_identity'] = meta_path.stem
                attempt[boundary] = row
        after = path.stat()
        if ((before.st_ino,before.st_size,before.st_mtime_ns) !=
                (after.st_ino,after.st_size,after.st_mtime_ns) or size != before.st_size):
            raise ValueError('event input changed during analysis')
        source_refs.append(dict(path=str(path.resolve()), bytes=size, sha256=digest.hexdigest()))
    if not by_attempt:
        raise ValueError('no control boundary evidence')

    spans = [(a+'__to__'+b, a, b) for a,b in zip(CONTROL_BOUNDARIES, CONTROL_BOUNDARIES[1:])]
    field_names = [name for name,_,_ in spans] + ['parent_total_ms', 'native_thread_cpu_ms']
    samples = {(phase,name): [] for phase in ('all','prebusiness','business') for name in field_names}
    records, statuses = [], Counter()
    for aid, events in sorted(by_attempt.items()):
        # In-role monotonicity is required even when cancellation leaves a
        # native utility operation finishing after the parent's terminal event.
        for role in (('parent_begin','parent_send','parent_received','parent_terminal'),
                     ('worker_received','frontend_begin','frontend_native_send',
                      'frontend_native_received','frontend_ready'), ('native_begin','native_ready')):
            seen = [events[b] for b in role if b in events]
            if len({(r['process_identity'],r['thread_ident']) for r in seen}) > 1:
                raise ValueError('attempt role changed process/thread')
            if any(a['monotonic_s'] > b['monotonic_s'] or a['thread_cpu_s'] > b['thread_cpu_s']
                   for a,b in zip(seen,seen[1:])):
                raise ValueError('invalid role boundary order')
        # Parent terminal is not ordered after late worker/native completion on
        # cancelled/error calls. Every other observed edge remains causal.
        chain = [events[b]['monotonic_s'] for b in CONTROL_BOUNDARIES[:-1] if b in events]
        if chain != sorted(chain):
            raise ValueError('invalid cross-process boundary order')
        terminal = events.get('parent_terminal')
        outcome = terminal['outcome'] if terminal else None
        missing = [b for b in CONTROL_BOUNDARIES if b not in events]
        complete = not missing and outcome == 'success'
        if outcome == 'success' and chain and terminal['monotonic_s'] < chain[-1]:
            raise ValueError('successful terminal precedes completion')
        status = ('success_complete' if complete else 'success_incomplete' if outcome == 'success'
                  else outcome if outcome else 'orphan' if 'parent_begin' not in events else 'incomplete')
        statuses[status] += 1
        begin = events.get('parent_begin')
        phase = ('prebusiness' if begin['monotonic_s'] < start else 'business') if begin else 'unknown'
        record = dict(attempt_id=aid, status=status, phase=phase,
            missing_boundaries=';'.join(missing), observed_boundaries=len(events),
            parent_begin_s=begin['monotonic_s'] if begin else None,
            request_bytes=events.get('parent_send',{}).get('byte_count'),
            response_bytes=events.get('parent_received',{}).get('byte_count'))
        record.update({name: None for name in field_names})
        if complete:
            for name,a,b in spans:
                record[name] = 1000*(events[b]['monotonic_s']-events[a]['monotonic_s'])
            record['parent_total_ms'] = 1000*(terminal['monotonic_s']-begin['monotonic_s'])
            record['native_thread_cpu_ms'] = 1000*(events['native_ready']['thread_cpu_s']-events['native_begin']['thread_cpu_s'])
            if not math.isclose(math.fsum(record[n] for n,_,_ in spans), record['parent_total_ms'], abs_tol=1e-6):
                raise ValueError('boundary decomposition identity failed')
            for name in field_names:
                for population in ('all',phase):
                    samples[population,name].append(record[name])
        records.append(record)
    metrics = []
    for (phase,name), values in samples.items():
        values.sort()
        metrics.append(dict(phase=phase, field=name, unit='ms', count=len(values),
            mean=math.fsum(values)/len(values) if values else None,
            p50=values[math.ceil(.5*len(values))-1] if values else None,
            p95=values[math.ceil(.95*len(values))-1] if values else None,
            max=values[-1] if values else None))
    result = dict(kind='diagnostic_control_boundary_audit_v1', formal_performance_result=False,
        new_experiment=False, quantile='Type-1', clock_id=clock_id, diagnostic_prefix_count=count,
        events=event_count, attempts=len(records), statuses=dict(statuses), partial_lines=partial_lines,
        complete_success_coverage=(not partial_lines and statuses['success_complete'] == len(records)),
        metrics=metrics, source_refs=source_refs, caveats=[
            'Intervals include diagnostic emission overhead; this is not ordinary Full performance or model/SLO qualification.',
            'Utility-send to native-begin includes transport, serialization and scheduling; it is not isolated queue or network time.',
            'Native wall interval includes physical inventory and its instrumentation; synchronous native thread CPU is separate, not async request CPU.',
            'Concurrent replica queries overlap. Do not sum attempts as request latency; RPC IDs are not request IDs.',
            'Only complete successful chains enter interval summaries. Missing/cancelled/error/partial evidence is retained, not imputed zero.',
            'Business classification uses parent begin versus arrival start; it includes drain and does not identify a request critical path.',
            'Within-attempt adjacent means are additive for the same population; their quantiles and overlapping parent totals are not additive.'])
    result['source_refs'].append(dict(path=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(records, output/'control_attempts.csv')
    write_csv(metrics, output/'control_intervals.csv')
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Result JSON file or directory")
    parser.add_argument("--output", default=None, help="Output directory; mandatory/fresh for native timeline")
    parser.add_argument("--native-timeline", action="store_true", help="Strict bounded native projection audit; no legacy paper copies")
    parser.add_argument("--rpc-breakdown", action="store_true", help="Audit retained native-async RPC fields, no new experiment or legacy paper copies")
    parser.add_argument("--admission-capacity", action="store_true", help="Audit retained admission KV snapshots; no serving or capacity configuration change")
    parser.add_argument("--control-boundaries", action="store_true", help="Strict diagnostic source-RPC boundary audit, not serving performance")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--sealed-summary", type=Path)
    parser.add_argument("--sealed-sha256")
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--terminals", type=Path)
    parser.add_argument("--watchdog", type=Path)
    parser.add_argument("--allow-failed", action="store_true", help="Retain all offered IDs; compute conditional success phases without imputing failed stages")
    parser.add_argument("--control-outcome", type=Path, help="Bounded normal outcome with independent control/quarantine observations")
    parser.add_argument("--scenario", default=None, help="Optional scenario-name substring")
    args = parser.parse_args()

    if sum((args.control_boundaries,args.admission_capacity,args.rpc_breakdown,args.native_timeline)) > 1:
        parser.error('choose one audit mode')
    if args.control_boundaries:
        if not all((args.output,args.deployment,args.launch_receipt)):
            parser.error('control boundaries require explicit fresh output/deployment/launch-receipt')
        result = analyze_control_boundaries(Path(args.input),args.launch_receipt,args.deployment,Path(args.output))
        print(json.dumps(result,indent=2,allow_nan=False))
        return

    if args.admission_capacity:
        if args.native_timeline or args.rpc_breakdown or not args.output:
            parser.error('admission capacity audit requires explicit fresh output and no other audit mode')
        print(json.dumps(analyze_admission_capacity(Path(args.input), Path(args.output)),
                         indent=2, allow_nan=False))
        return

    if args.rpc_breakdown:
        if args.native_timeline or not all((args.output,args.deployment,args.terminals,args.sealed_summary,args.sealed_sha256)):
            parser.error('RPC audit requires explicit fresh output/deployment/terminals/sealed-summary/sealed-sha256 and no native-timeline')
        result = analyze_rpc_breakdown(Path(args.input),args.deployment,args.terminals,
                                      args.sealed_summary,args.sealed_sha256,Path(args.output))
        print(json.dumps(result,indent=2,allow_nan=False))
        return

    if args.native_timeline:
        if not all((args.output,args.deployment,args.terminals,args.watchdog)):
            parser.error('native timeline requires explicit output/deployment/terminals/watchdog')
        result = analyze_native_timeline(Path(args.input),args.deployment,args.terminals,args.watchdog,Path(args.output),
            allow_failed=args.allow_failed,control_outcome_path=args.control_outcome)
        print(json.dumps(result,indent=2,allow_nan=False))
        return

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output or "figs/paper/control_path").expanduser().resolve()
    source, scenario, summary = select_scenario(load_payloads(input_path), args.scenario)

    rows: List[Dict[str, Any]] = []
    for label, request_field, avg_field, p95_field, positive_only in OPERATIONS:
        avg_us, p95_us, count = summarize_operation(
            scenario,
            summary,
            request_field,
            avg_field,
            p95_field,
            positive_only=positive_only,
        )
        rows.append({
            "operation_label": label,
            "operation_latex": label.replace("\n", " "),
            "avg_us": avg_us,
            "p95_us": p95_us,
            "avg_ms": avg_us / 1000.0,
            "p95_ms": p95_us / 1000.0,
            "events": count,
            "source": str(source),
        })
    bg_avg_us, bg_p95_us, bg_count = summarize_background(summary)
    if bg_count > 0 or bg_avg_us > 0.0 or bg_p95_us > 0.0:
        rows.append({
            "operation_label": "Background\nhandoff plan",
            "operation_latex": "Background handoff plan",
            "avg_us": bg_avg_us,
            "p95_us": bg_p95_us,
            "avg_ms": bg_avg_us / 1000.0,
            "p95_ms": bg_p95_us / 1000.0,
            "events": bg_count,
            "source": str(source),
        })

    write_csv(rows, output_dir / "control_path_overhead_summary.csv")
    write_latex(rows, output_dir / "tables" / "table_control_path_overhead.tex")
    write_manifest(rows, output_dir / "control_path_overhead_manifest.json", source)
    plot(rows, output_dir / "fig_control_path_overhead.pdf")
    paper_copy = Path("figs/fig_control_path_overhead.pdf").resolve()
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, paper_copy)

    print(f"source={source}")
    for row in rows:
        print(
            f"{row['operation_label'].replace(chr(10), ' ')}: "
            f"avg={row['avg_ms']:.3f}ms p95={row['p95_ms']:.3f}ms events={int(row['events'])}"
        )
    print(f"wrote {output_dir / 'fig_control_path_overhead.pdf'}")


if __name__ == "__main__":
    main()
