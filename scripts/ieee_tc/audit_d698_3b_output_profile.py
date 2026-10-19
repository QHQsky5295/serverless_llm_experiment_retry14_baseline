"""D698 bounded audit for 3B declared-output service-class coverage.

This is an offline, fail-closed coverage check.  It never estimates a missing
class and never writes a runtime profile.  The candidate is launchable only
after the missing native measurements have been collected and independently
sealed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from faaslora.experiment.instance_pool import ServiceClassBins
from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
RAW = ROOT / "results/ieee_tc/p2_backend_qualification/d644_20261017"
SAMPLES = RAW / "completed_service_samples.json"
PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d645_3b_profiles/initialization/service.json"
OUT = ROOT / "paper_results/ieee_tc/p2_backend/d698_3b_output_profile_coverage"


def _read(path: Path):
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _class_key(value):
    return tuple(asdict(value).items())


def _quartile_edges(samples, output_cap: int):
    values = sorted(row["class_features"]["declared_output_tokens"] for row in samples)
    if not values or any(type(x) is not int or not 1 <= x <= output_cap for x in values):
        raise ValueError("completed declared-output samples must be within the fixed cap")
    return sorted({values[math.ceil(len(values) * q) - 1] for q in (0.25, 0.5, 0.75)} | {output_cap})


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUT}")
    samples_doc = _read(SAMPLES)
    profile = _read(PROFILE)
    samples = samples_doc["samples"]
    retained = profile["samples"]
    if samples_doc.get("production_profile_frozen") is not False:
        raise ValueError("D644 source samples are not marked development-only")
    if profile.get("development_only") is not True or profile.get("full_qualified") is not False:
        raise ValueError("D645 profile is not the expected development profile")
    if len(samples) != 360 or len(retained) != 360:
        raise ValueError("unexpected D644/D645 service sample count")
    source_sha = _digest(RAW / "source1.json")
    if samples_doc.get("source_sha256") != source_sha:
        raise ValueError("completed sample source SHA does not bind D644 source1.json")
    if [row["request_id"] for row in samples] != [row["request_id"] for row in retained]:
        raise ValueError("D645 retained service samples differ from D644 completed samples")

    output_cap = 256
    edges = _quartile_edges(samples, output_cap)
    old_bins = ServiceClassBins(**{key: tuple(value) for key, value in profile["bins"].items()})
    candidate_bins = dict(profile["bins"])
    candidate_bins["declared_output_tokens"] = edges
    new_bins = ServiceClassBins(**{key: tuple(value) for key, value in candidate_bins.items()})

    # The non-output domain comes only from the existing measured profile.  A
    # missing class is therefore an explicit native measurement obligation.
    non_output = {
        _class_key(old_bins.classify(**row["class_features"]))
        for row in retained
    }
    required = {
        _class_key(new_bins.classify(
            **{**row["class_features"], "declared_output_tokens": edge}
        ))
        for row in retained
        for edge in edges
    }
    observed = Counter(
        _class_key(new_bins.classify(**row["class_features"]))
        for row in retained + samples
    )
    missing = sorted(required - set(observed))
    if not missing:
        raise ValueError("coverage is unexpectedly complete; do not seal a gap-only candidate")

    def unpack(key):
        return dict(key)

    bin_counts = Counter()
    for row in retained + samples:
        cls = new_bins.classify(**row["class_features"])
        bin_counts[cls.output_limit_bin] += 1

    result = {
        "kind": "d698_3b_declared_output_profile_coverage_v1",
        "campaign": "D698",
        "status": "coverage_only_missing_native_measurements",
        "launchable": False,
        "performance_accepted": False,
        "production_profile_frozen": False,
        "numerical_adapter_correctness_qualified": False,
        "n_correct": None,
        "new_gpu_runs": 0,
        "no_default_fills": True,
        "no_interpolation_or_pooling": True,
        "rule": "historical_declared_output_type1_quartiles_cap_v1",
        "output_cap": output_cap,
        "output_edges": edges,
        "retained_profile_classes": len(non_output),
        "required_classes": len(required),
        "observed_classes": len(observed),
        "missing_classes": [unpack(key) for key in missing],
        "observed_class_counts": [
            {"service_class": unpack(key), "n": count}
            for key, count in sorted(observed.items())
        ],
        "observed_samples": len(samples) + len(retained),
        "source_refs": [
            {"path": str(path), "sha256": _digest(path)}
            for path in (
                SAMPLES,
                RAW / "source1.json",
                PROFILE,
                ROOT / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md",
                ROOT / "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md",
                ROOT / "scripts/ieee_tc/audit_d698_3b_output_profile.py",
            )
        ],
        "source_execution_sha256": source_sha,
        "source_plan_sha256": p.check_plan(),
        "bin_sample_counts": dict(sorted(bin_counts.items())),
        "coverage_interpretation": (
            "Existing 3B native intervals show output-length heterogeneity, but "
            "30 of 160 required output/non-output classes have no measured interval. "
            "Only those classes may be collected; until then no runtime profile or "
            "performance claim is authorized."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "coverage.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": True,
        "output": str(OUT / "coverage.json"),
        "output_edges": edges,
        "required_classes": len(required),
        "observed_classes": len(observed),
        "missing_classes": len(missing),
    }, indent=2))


if __name__ == "__main__":
    main()
