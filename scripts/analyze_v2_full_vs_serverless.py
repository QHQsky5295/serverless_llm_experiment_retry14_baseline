#!/usr/bin/env python3
"""Formal PrimeLoRA Full versus ServerlessLLM-new publication analyzer.

Formal mode is intentionally fail-closed: inputs must be completed held-out
campaign manifests, the matrix and frozen workload axes must match exactly,
and all statistics use the seed-level paired difference as their unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
    from plot_paper_figures import _v2_mean_ci95
    from plot_paper_sensitivity import (
        V2SensitivityObservation,
        _sha256_field,
        load_v2_sensitivity_observations,
    )
except ImportError:  # Package imports used by tests.
    from scripts.eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
    from scripts import plot_paper_figures as _plot_paper_figures

    sys.modules.setdefault("plot_paper_figures", _plot_paper_figures)
    from scripts.plot_paper_figures import _v2_mean_ci95
    from scripts.plot_paper_sensitivity import (
        V2SensitivityObservation,
        _sha256_field,
        load_v2_sensitivity_observations,
    )


FORMAL_MODELS = ("llama2_7b", "llama32_3b")
FORMAL_SEEDS = (43, 44, 45)
FORMAL_SYSTEMS = ("faaslora", "serverlessllm")
FORMAL_SCENARIOS = {
    "faaslora": "v2_full",
    # The official ServerlessLLM-new harness deliberately retains the
    # historical result-schema scenario name.  "new" identifies the pinned
    # implementation path/system, not a second scenario row.
    "serverlessllm": "serverlessllm_fair",
}
METRICS = (
    "ttft_avg_ms",
    "ttft_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
    "tok_s",
    "cost_req_usd",
    "ce",
)
HIGHER_IS_BETTER = {"tok_s", "ce"}


@dataclass(frozen=True)
class FormalRun:
    model: str
    seed: int
    system: str
    scenario: str
    source: Path
    total: int
    completed: int
    generation_contract: str
    selected_num_adapters: int | None
    bandwidth_mib_s: float | None
    configured_time_scale_factor: float | None
    effective_time_scale_factor: float | None
    zipf_exponent: float | None
    active_adapter_cap: int | None
    hotset_rotation_requests: int | None
    hotset_rotation_mode: str
    hotset_overlap_fraction: float | None
    trace_sha256: str
    adapter_subset_sha256: str
    system_resolved_config_sha256: str
    metrics: Mapping[str, float]


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _sha(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SystemExit(f"formal Full-vs-Serverless {label}: invalid SHA-256")
    return digest


def _read_trace_axes(observation: V2SensitivityObservation) -> Dict[str, Any]:
    raw_path = _first(
        observation.metadata.get("shared_trace_path"),
        observation.metadata.get("trace_source"),
        observation.manifest.get("shared_trace_path"),
    )
    if raw_path is None:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = observation.source.parent / path
    if not path.is_file():
        raise SystemExit(
            f"formal Full-vs-Serverless {observation.source}: shared trace is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot read shared trace axes from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"shared trace must be a JSON object: {path}")
    return payload


def _exact_model(model: str) -> str:
    if model in FORMAL_MODELS:
        return model
    raise SystemExit(
        f"formal Full-vs-Serverless unsupported model {model!r}; expected {FORMAL_MODELS}"
    )


def _run_from_observation(observation: V2SensitivityObservation) -> FormalRun:
    trace = _read_trace_axes(observation)
    metadata = observation.metadata
    manifest = observation.manifest
    profile = observation.profile
    contract_values = {
        str(value).strip().lower()
        for value in (
            metadata.get("generation_contract"),
            manifest.get("generation_contract"),
        )
        if value is not None and str(value).strip()
    }
    if len(contract_values) != 1:
        raise SystemExit(
            f"formal Full-vs-Serverless {observation.source}: missing/conflicting "
            f"generation contract {sorted(contract_values)}"
        )
    config_sha = _sha(
        _first(
            metadata.get("system_resolved_config_sha256"),
            manifest.get("system_resolved_config_sha256"),
        ),
        f"{observation.source}.system_resolved_config_sha256",
    )
    return FormalRun(
        model=_exact_model(observation.model),
        seed=int(observation.seed),
        system=observation.system_key,
        scenario=observation.scenario,
        source=observation.source.resolve(),
        total=observation.total,
        completed=observation.completed,
        generation_contract=next(iter(contract_values)),
        selected_num_adapters=int(
            _first(
                metadata.get("selected_num_adapters"),
                metadata.get("num_adapters"),
                manifest.get("selected_num_adapters"),
                profile.get("nominal_adapter_pool_size"),
            )
        ) if _first(
            metadata.get("selected_num_adapters"),
            metadata.get("num_adapters"),
            manifest.get("selected_num_adapters"),
            profile.get("nominal_adapter_pool_size"),
        ) is not None else None,
        bandwidth_mib_s=float(
            _first(metadata.get("bandwidth_mib_s"), manifest.get("bandwidth_mib_s"))
        ) if _first(metadata.get("bandwidth_mib_s"), manifest.get("bandwidth_mib_s")) is not None else None,
        configured_time_scale_factor=float(
            _first(
                metadata.get("configured_time_scale_factor"),
                metadata.get("shared_trace_configured_time_scale_factor"),
                trace.get("configured_time_scale_factor"),
            )
        ) if _first(
            metadata.get("configured_time_scale_factor"),
            metadata.get("shared_trace_configured_time_scale_factor"),
            trace.get("configured_time_scale_factor"),
        ) is not None else None,
        effective_time_scale_factor=float(
            _first(
                metadata.get("effective_time_scale_factor"),
                metadata.get("shared_trace_effective_time_scale_factor"),
                trace.get("effective_time_scale_factor"),
            )
        ) if _first(
            metadata.get("effective_time_scale_factor"),
            metadata.get("shared_trace_effective_time_scale_factor"),
            trace.get("effective_time_scale_factor"),
        ) is not None else None,
        zipf_exponent=float(profile.get("zipf_exponent")) if profile.get("zipf_exponent") is not None else None,
        active_adapter_cap=int(profile.get("active_adapter_cap")) if profile.get("active_adapter_cap") is not None else None,
        hotset_rotation_requests=int(profile.get("hotset_rotation_requests")) if profile.get("hotset_rotation_requests") is not None else None,
        hotset_rotation_mode=str(profile.get("rotation_mode") or "").strip().lower(),
        hotset_overlap_fraction=float(profile.get("hotset_overlap_fraction")) if profile.get("hotset_overlap_fraction") is not None else None,
        trace_sha256=_sha256_field(metadata, manifest, "shared_trace_sha256"),
        adapter_subset_sha256=_sha256_field(
            metadata, manifest, "shared_adapter_subset_sha256"
        ),
        system_resolved_config_sha256=config_sha,
        metrics=dict(observation.metrics),
    )


def load_formal_runs(inputs: Sequence[Path]) -> List[FormalRun]:
    return [
        _run_from_observation(observation)
        for observation in load_v2_sensitivity_observations(inputs)
    ]


def _identity(run: FormalRun) -> tuple[str, int, str]:
    return run.model, run.seed, run.system


def validate_formal_matrix(runs: Sequence[FormalRun]) -> None:
    expected = {
        (model, seed, system)
        for model in FORMAL_MODELS
        for seed in FORMAL_SEEDS
        for system in FORMAL_SYSTEMS
    }
    counts: Dict[tuple[str, int, str], int] = defaultdict(int)
    for run in runs:
        counts[_identity(run)] += 1
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicate = sorted(identity for identity, count in counts.items() if count != 1)
    if missing or extra or duplicate:
        raise SystemExit(
            "formal Full-vs-Serverless matrix identity mismatch: "
            f"missing={missing}, extra={extra}, duplicate={duplicate}"
        )

    expected_axes: Mapping[str, Any] = {
        "total": 4000,
        "completed": 4000,
        "generation_contract": "legacy",
        "selected_num_adapters": 500,
        "bandwidth_mib_s": 250.0,
        "configured_time_scale_factor": 8.0,
        "effective_time_scale_factor": 8.0,
        "zipf_exponent": 1.0,
        "active_adapter_cap": 48,
        "hotset_rotation_requests": 500,
        "hotset_rotation_mode": "legacy",
        "hotset_overlap_fraction": 0.75,
    }
    config_hashes: Dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_hashes: Dict[tuple[str, int], Dict[str, set[str]]] = defaultdict(
        lambda: {"trace": set(), "subset": set()}
    )
    for run in runs:
        expected_scenario = FORMAL_SCENARIOS.get(run.system)
        if run.scenario != expected_scenario:
            raise SystemExit(
                f"formal Full-vs-Serverless {run.source}: system={run.system} must use "
                f"scenario={expected_scenario}, observed={run.scenario}"
            )
        for name, expected_value in expected_axes.items():
            value = getattr(run, name)
            if isinstance(expected_value, float):
                matches = value is not None and math.isclose(
                    float(value), expected_value, rel_tol=0.0, abs_tol=1e-9
                )
            else:
                matches = value == expected_value
            if not matches:
                raise SystemExit(
                    f"formal Full-vs-Serverless {run.source}: wrong {name}; "
                    f"expected={expected_value!r}, observed={value!r}"
                )
        for metric in METRICS:
            value = run.metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise SystemExit(
                    f"formal Full-vs-Serverless {run.source}: missing metric {metric}"
                )
        config_hashes[(run.model, run.system)].add(
            _sha(run.system_resolved_config_sha256, str(run.source))
        )
        pair_hashes[(run.model, run.seed)]["trace"].add(run.trace_sha256)
        pair_hashes[(run.model, run.seed)]["subset"].add(run.adapter_subset_sha256)
    drift = {group: hashes for group, hashes in config_hashes.items() if len(hashes) != 1}
    if drift:
        raise SystemExit(
            f"formal Full-vs-Serverless frozen configuration drift: {drift}"
        )
    mismatched_pairs = {
        pair: values
        for pair, values in pair_hashes.items()
        if len(values["trace"]) != 1 or len(values["subset"]) != 1
    }
    if mismatched_pairs:
        raise SystemExit(
            f"formal Full-vs-Serverless trace/subset mismatch: {mismatched_pairs}"
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    runs: Sequence[FormalRun],
    out_dir: Path,
    *,
    inputs: Sequence[Path],
    manifest_paths: Sequence[Path],
) -> None:
    validate_formal_matrix(runs)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty Full-vs-Serverless output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run: List[Dict[str, Any]] = []
    lookup = {_identity(run): run for run in runs}
    for run in sorted(runs, key=lambda item: (item.model, item.seed, item.system)):
        per_run.append(
            {
                "model": run.model,
                "seed": run.seed,
                "system": run.system,
                "scenario": run.scenario,
                "source": str(run.source),
                "total": run.total,
                "completed": run.completed,
                "generation_contract": run.generation_contract,
                "shared_trace_sha256": run.trace_sha256,
                "shared_adapter_subset_sha256": run.adapter_subset_sha256,
                "system_resolved_config_sha256": run.system_resolved_config_sha256,
                **run.metrics,
            }
        )

    paired: List[Dict[str, Any]] = []
    for model in FORMAL_MODELS:
        for seed in FORMAL_SEEDS:
            prime = lookup[(model, seed, "faaslora")]
            baseline = lookup[(model, seed, "serverlessllm")]
            for metric in METRICS:
                prime_value = float(prime.metrics[metric])
                baseline_value = float(baseline.metrics[metric])
                difference = prime_value - baseline_value
                if baseline_value == 0.0:
                    raise SystemExit(f"cannot compute paired improvement for zero {metric}")
                improvement = (
                    (prime_value - baseline_value) / baseline_value * 100.0
                    if metric in HIGHER_IS_BETTER
                    else (baseline_value - prime_value) / baseline_value * 100.0
                )
                paired.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "higher_is_better": metric in HIGHER_IS_BETTER,
                        "prime_value": prime_value,
                        "serverlessllm_value": baseline_value,
                        "paired_difference_prime_minus_serverlessllm": difference,
                        "prime_improvement_pct": improvement,
                    }
                )

    summary: List[Dict[str, Any]] = []
    for model in FORMAL_MODELS:
        for metric in METRICS:
            rows = [row for row in paired if row["model"] == model and row["metric"] == metric]
            differences = [float(row["paired_difference_prime_minus_serverlessllm"]) for row in rows]
            improvements = [float(row["prime_improvement_pct"]) for row in rows]
            diff_mean, diff_std, diff_half = _v2_mean_ci95(differences)
            imp_mean, imp_std, imp_half = _v2_mean_ci95(improvements)
            summary.append(
                {
                    "model": model,
                    "metric": metric,
                    "higher_is_better": metric in HIGHER_IS_BETTER,
                    "paired_seed_count": len(rows),
                    "paired_seeds": ";".join(str(row["seed"]) for row in rows),
                    "paired_difference_mean": diff_mean,
                    "paired_difference_std": diff_std,
                    "paired_difference_ci95_half_width": diff_half,
                    "prime_improvement_pct_mean": imp_mean,
                    "prime_improvement_pct_std": imp_std,
                    "prime_improvement_pct_ci95_half_width": imp_half,
                }
            )

    per_run_path = out_dir / "full_vs_serverless_per_run.csv"
    paired_path = out_dir / "full_vs_serverless_paired_per_seed.csv"
    summary_path = out_dir / "full_vs_serverless_paired_summary.csv"
    _write_csv(per_run_path, per_run)
    _write_csv(paired_path, paired)
    _write_csv(summary_path, summary)
    manifest = {
        "analysis": "v2_full_vs_serverlessllm_new_formal",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_matrix": True,
        "inputs": [str(Path(path).expanduser().resolve()) for path in inputs],
        "provenance_manifests": [str(path) for path in manifest_paths],
        "identity_count": len(runs),
        "statistical_unit": "independent held-out seed",
        "ci": "two-sided 95% Student-t over three paired seed-level differences",
        "frozen_workload": {
            "requests": 4000,
            "adapters": 500,
            "bandwidth_mib_s": 250.0,
            "time_scale": 8.0,
            "generation_contract": "legacy",
            "zipf_exponent": 1.0,
            "active_adapter_cap": 48,
            "hotset_rotation_requests": 500,
            "hotset_rotation_mode": "legacy",
            "hotset_overlap_fraction": 0.75,
        },
        "csvs": [per_run_path.name, paired_path.name, summary_path.name],
    }
    (out_dir / "full_vs_serverless_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def analyze(inputs: Sequence[Path], out_dir: Path) -> None:
    provenance = build_formal_provenance_index(inputs)
    runs = load_formal_runs(inputs)
    validate_formal_matrix(runs)
    validate_formal_analysis_sources(
        provenance,
        (
            FormalAnalysisIdentity(
                source=run.source,
                model=run.model,
                variant=run.system,
                seed=run.seed,
            )
            for run in runs
        ),
        analysis_label="V2 Full versus ServerlessLLM-new",
    )
    write_outputs(
        runs,
        out_dir,
        inputs=inputs,
        manifest_paths=provenance.manifest_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal V2 PrimeLoRA Full versus ServerlessLLM-new analyzer"
    )
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output_dir.resolve())
    print(f"generated formal Full-vs-Serverless analysis -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
