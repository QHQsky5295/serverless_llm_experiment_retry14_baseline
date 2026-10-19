"""Prepare the missing 3B native output-profile classes (D699).

This is deliberately a CPU-only, append-only preparation step.  It does not
measure or relabel a profile, create a trace, create an adapter, or launch a
GPU worker.  Every request reference is resolved against the frozen 1,000
request prefix and the existing 500-adapter pool.  The resulting source spec
is launchable only after the native collector measures the waves and a separate
coverage/export audit closes all 30 D698 gaps.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path

import yaml

from faaslora.datasets.workload_generator import FrozenReplayPlan
from faaslora.experiment.instance_pool import ServiceClassBins
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
BASE = ROOT / "paper_results/ieee_tc/p2_backend"
SOURCE_SPEC = BASE / "d645_3b_profiles/source_spec.json"
COVERAGE = BASE / "d698_3b_output_profile_coverage/coverage.json"
SAMPLES = Path(
    "/home/qhq/serverless_llm_experiment/results/ieee_tc/"
    "p2_backend_qualification/d644_20261017/completed_service_samples.json"
)
TRACE = Path(
    "/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/"
    "20260509_134259_llama32_3b_main_s8/shared_artifacts/"
    "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json"
)
POOL = Path("/home/qhq/serverless_llm_experiment/artifacts/frozen/llama32_3b_a500_v1_modelscope")
CONFIG = ROOT / "configs/ieee_tc/20261017_d641_3b_native_source.yaml"
OUT = BASE / "d699_3b_output_profile_gaps"
RAW_ROOT = Path(
    "/home/qhq/serverless_llm_experiment/results/ieee_tc/"
    "p2_backend_qualification/d699_20261019"
)
OUTPUT_EDGES = [89, 152, 179, 256]
RANK_EDGES = [8, 16, 64]
ADMISSION_EDGES = [1, 2, 4, 8]
OUTPUT_CAP = 256
TRACE_PREFIX = 1000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def index_for(edges, value: int) -> int:
    return bisect_left(edges, value)


def target_bin(target: int) -> int:
    return index_for(OUTPUT_EDGES, target)


def rank_bin(rank: int) -> int:
    return index_for(RANK_EDGES, rank)


def source_name(service_class: dict) -> str:
    key = (service_class["tier"], service_class["representation"])
    mapping = {
        ("gpu", "native_gpu_dense_slot_v1:torch.float16"): "gpu",
        ("host", "native_cpu_dense_ab_v1:torch.float16:pinned:packed"): "native_host",
        ("host", "verified_regular_file_tree_v1"): "file_host",
        ("nvme", "verified_regular_file_tree_v1"): "nvme",
        ("remote", "tar_gzip_verified_file_tree_v1"): "remote",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unsupported D698 source class: {key}") from exc


def source_for_name(name: str) -> tuple[str, str]:
    return {
        "gpu": ("gpu", "native_gpu_dense_slot_v1:torch.float16"),
        "native_host": ("host", "native_cpu_dense_ab_v1:torch.float16:pinned:packed"),
        "file_host": ("host", "verified_regular_file_tree_v1"),
        "nvme": ("nvme", "verified_regular_file_tree_v1"),
        "remote": ("remote", "tar_gzip_verified_file_tree_v1"),
    }[name]


def load_population(spec: dict):
    """Resolve existing request rows and rank identities without running a model."""
    index = read(ROOT / spec["content_index"]["path"])
    artifacts = {row["id"]: row for row in index["artifacts"]}
    if len(artifacts) != 500 or spec["content_index"]["sha256"] != digest(ROOT / spec["content_index"]["path"]):
        raise ValueError("3B content index is not the frozen 500-adapter index")
    plan = FrozenReplayPlan.load(TRACE, count=TRACE_PREFIX)
    if plan.source_sha256 != spec["trace_sha256"]:
        raise ValueError("trace SHA does not match the D645 source specification")
    entries = {entry.request_id: entry for entry in plan.entries}
    client = HttpArtifactStoreClient(endpoint="http://127.0.0.1:1")
    client.configure_content_manifest(index)
    population = []
    identities = {}
    for source_index, entry in enumerate(plan.entries):
        row = json.loads(entry.source_json)
        aid = row["adapter_id"]
        if aid not in artifacts:
            raise ValueError(f"trace adapter is outside frozen content index: {aid}")
        adapter_dir = (POOL / aid).resolve(strict=True)
        config_path = adapter_dir / "adapter_config.json"
        if adapter_dir.parent != POOL.resolve() or not config_path.is_file():
            raise ValueError(f"adapter config is outside the frozen pool: {aid}")
        config = read(config_path)
        rank = config.get("r")
        if type(rank) is not int or rank <= 0:
            raise ValueError(f"adapter rank is not an observed positive integer: {aid}")
        target = min(row["expected_output_tokens"], OUTPUT_CAP)
        if type(target) is not int or target < 1:
            raise ValueError(f"invalid declared output for {entry.request_id}")
        # The source profile's only prompt boundary is 759.  Restrict the
        # candidate pool to the existing prefix requests whose declared input
        # is already within that bound; the collector still records the
        # canonical prompt length at runtime and refuses any actual mismatch.
        if row["expected_input_tokens"] > 759:
            continue
        identities.setdefault(aid, client.routing_identity(aid, config_path.read_bytes()))
        population.append({
            "source_index": source_index,
            "source_request_id": entry.request_id,
            "adapter_id": aid,
            "rank": rank,
            "target_tokens": target,
            "output_bin": target_bin(target),
            "rank_bin": rank_bin(rank),
            "expected_input_tokens": row["expected_input_tokens"],
        })
    if not population:
        raise ValueError("no existing prefix request is eligible for gap preparation")
    return plan, entries, index, artifacts, population, identities


def make_waves(missing: list[dict], population: list[dict], *, max_lanes: int = 8):
    """Make one natural wave per missing class, placing its target last.

    A wave of size 1/2/4/8 puts the target at admitted_after_accept 1/2/4/8,
    exactly the frozen inclusive upper-bin edges.  Other lanes are real,
    existing requests and are measured too; they are not held open or delayed
    artificially.  Reuse across waves is allowed, but a wave never duplicates
    an adapter.
    """
    by_class = defaultdict(list)
    for row in population:
        by_class[(row["output_bin"], row["rank_bin"])].append(row)
    for rows in by_class.values():
        rows.sort(key=lambda r: (r["source_index"], r["adapter_id"]))
    all_rows = sorted(population, key=lambda r: (r["source_index"], r["adapter_id"]))

    def ref(row):
        return {"source_request_id": row["source_request_id"], "adapter_id": row["adapter_id"]}

    waves = []
    target_manifest = []
    for number, wanted in enumerate(sorted(missing, key=lambda x: tuple(x[k] for k in (
            "tier", "representation", "output_limit_bin", "rank_bin", "admitted_bin")))):
        source = source_name(wanted)
        output_rank = (wanted["output_limit_bin"], wanted["rank_bin"])
        candidates = by_class.get(output_rank, [])
        if not candidates:
            raise ValueError(f"no existing request for output/rank class {output_rank}")
        target = candidates[0]
        admitted_edge = ADMISSION_EDGES[wanted["admitted_bin"]]
        if admitted_edge > max_lanes:
            raise ValueError("target admission bin exceeds the qualified runtime lane count")
        used = {target["adapter_id"]}
        background = []
        # Prefer other classes, then deterministic source order.  Their only
        # purpose is genuine concurrent occupancy; all are preserved in raw
        # output and checked by the collector.
        required_background = admitted_edge - 1
        if required_background:
            for row in all_rows:
                if row["adapter_id"] in used:
                    continue
                background.append(row)
                used.add(row["adapter_id"])
                if len(background) == required_background:
                    break
        if len(background) != admitted_edge - 1:
            raise ValueError(f"insufficient distinct existing background for gap {wanted}")
        requests = [ref(row) for row in background] + [ref(target)]
        wave = {
            "role": "representative_measurement",
            "round": 0,
            "source": source,
            "requests": requests,
            "target_source_request_id": target["source_request_id"],
            "target_adapter_id": target["adapter_id"],
            "target_position": len(requests) - 1,
            "target_admission_edge": admitted_edge,
            "intended_missing_class": copy.deepcopy(wanted),
            "background_request_count": len(background),
            "actual_admission_coverage_required": True,
        }
        waves.append(wave)
        target_manifest.append({
            "wave": number + 1,
            "source": source,
            "missing_class": copy.deepcopy(wanted),
            "target": dict(source_request_id=target["source_request_id"],
                            adapter_id=target["adapter_id"],
                            declared_output_tokens=target["target_tokens"],
                            rank=target["rank"],
                            expected_input_tokens=target["expected_input_tokens"]),
            "wave_size": len(requests),
            "target_position": len(requests) - 1,
        })

    # The collector starts each wave from a controlled empty state.  Retain a
    # single <=8-lane GPU warmup; it is not part of representative coverage.
    warm = []
    seen = set()
    for wave in waves:
        for request in wave["requests"]:
            if request["adapter_id"] not in seen:
                warm.append(request)
                seen.add(request["adapter_id"])
            if len(warm) == max_lanes:
                break
        if len(warm) == max_lanes:
            break
    if not warm:
        raise ValueError("gap source spec has no warmup requests")
    return [dict(role="kernel_warmup_retained", round=-1, source="gpu", requests=warm)] + waves, target_manifest


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUT}")
    spec = read(SOURCE_SPEC)
    coverage = read(COVERAGE)
    if coverage.get("launchable") is not False or len(coverage["missing_classes"]) != 30:
        raise ValueError("D698 coverage audit is not the expected 30-class closed obligation")
    if coverage["output_edges"] != OUTPUT_EDGES:
        raise ValueError("D698 output edges changed; no new gap contract is allowed")
    plan, entries, index, artifacts, population, identities = load_population(spec)
    waves, target_manifest = make_waves(coverage["missing_classes"], population)

    candidate = copy.deepcopy(spec)
    candidate.update(
        kind="native_source_profile_spec_v1",
        purpose="D699_only_D698_missing_3B_declared_output_classes",
        bins=dict(spec["bins"], declared_output_tokens=OUTPUT_EDGES),
        waves=waves,
        representatives=[],
        rounds=1,
        selection_window=dict(prefix=TRACE_PREFIX, source_trace_sha256=plan.source_sha256),
        order_rule=(
            "Existing first1000 prefix only; target selected by source order within "
            "the frozen output/rank class; target is last in a natural wave of "
            "size 1/2/4/8; no synthetic request, wait, interpolation or default."
        ),
        derivation=(
            "D698 frozen quartile edges [89,152,179,256]. One wave per missing "
            "source/output/rank/admission class; backgrounds are existing prefix "
            "requests and all actual classes remain measured."
        ),
        supplementation=dict(
            source_coverage=dict(path=str(COVERAGE.relative_to(ROOT)), sha256=digest(COVERAGE)),
            retained_profile=dict(path=str(SOURCE_SPEC.relative_to(ROOT)), sha256=digest(SOURCE_SPEC)),
            missing_classes=coverage["missing_classes"],
            no_runtime_profile_until_measurement=True,
            no_gpu_runs_in_preparation=True,
        ),
    )
    # Keep an explicit representative identity list for every adapter used by
    # the new waves.  This is metadata only; source_profile_inputs remains the
    # authoritative static resolver.
    used = {q["adapter_id"] for wave in waves for q in wave["requests"]}
    for aid in sorted(used):
        item = identities[aid]
        candidate["representatives"].append(dict(
            adapter_id=aid,
            source_request_id=next(q["source_request_id"] for wave in waves for q in wave["requests"]
                                   if q["adapter_id"] == aid),
            rank=item["rank"],
            file_tree_sha256=item["content_sha256"],
            logical_bytes=item["remote_payload_bytes"],
            selection="existing_prefix_gap_wave_identity_only",
        ))

    # Resolve the exact same static validator used immediately before a native
    # source-matrix launch.  No remote endpoint is contacted by this check.
    model_key = "llama32_3b_main_modelscope"
    cfg = yaml.safe_load(CONFIG.read_text())["model_profiles"][model_key]["model"]
    cfg.update(
        generation_contract="fixed_length_greedy_v1",
        canonical_prompt_renderer="role_lines_v1",
        max_input_len=759,
        max_output_tokens_cap=256,
        ieee_physical_allocation=True,
        timing_contract="ieee_tc_native_v1",
        ieee_worker_observation=True,
        ieee_gpu_references=True,
        ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],
    )
    cfg.update(candidate["model_overrides"])
    spec_path = OUT / "source_spec.json"
    OUT.mkdir(parents=True, exist_ok=False)
    spec_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    _, bins, resolved_identities, resolved_waves, static_index = p.source_profile_inputs(
        spec_path, plan, cfg, POOL.resolve(strict=True)
    )
    if len(resolved_waves) != len(waves) or len(resolved_identities) != len(used):
        raise ValueError("static source resolver changed the prepared wave identity")
    if len(waves) != 31 or sum(len(w["requests"]) for w in waves) <= 31:
        raise ValueError("unexpected D699 wave cardinality")
    # Every target remains a genuine existing request and all target class
    # fields are statically derivable from the frozen bins.
    for item in target_manifest:
        if item["target"]["source_request_id"] not in entries:
            raise ValueError("target is not in the frozen trace prefix")
        if item["target"]["adapter_id"] not in artifacts:
            raise ValueError("target is not in the frozen content index")
        if item["missing_class"]["output_limit_bin"] != target_bin(
                min(item["target"]["declared_output_tokens"], OUTPUT_CAP)):
            raise ValueError("target output class does not match the frozen edge")
        if item["missing_class"]["rank_bin"] != rank_bin(item["target"]["rank"]):
            raise ValueError("target rank class does not match the frozen edge")

    preparation = {
        "kind": "d699_3b_output_profile_gap_preparation_v1",
        "campaign": "D699",
        "preparation_only": True,
        "launchable": False,
        "gpu_runs": 0,
        "n_correct": None,
        "numerical_adapter_correctness_qualified": False,
        "dynamic_measurement_required": True,
        "missing_class_count": len(coverage["missing_classes"]),
        "prepared_wave_count": len(waves),
        "prepared_measurement_wave_count": len(waves) - 1,
        "prepared_request_count": sum(len(w["requests"]) for w in waves),
        "target_manifest": target_manifest,
        "static_validation": {
            "source_profile_inputs": True,
            "resolved_wave_count": len(resolved_waves),
            "resolved_adapter_count": len(resolved_identities),
            "bins": asdict(bins) if hasattr(bins, "__dataclass_fields__") else candidate["bins"],
            "trace_prefix": TRACE_PREFIX,
            "trace_sha256": plan.source_sha256,
            "content_index_sha256": digest(ROOT / spec["content_index"]["path"]),
            "pool": str(POOL),
        },
        "source_refs": [
            {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
             "sha256": digest(path)}
            for path in (SOURCE_SPEC, COVERAGE, SAMPLES, TRACE, CONFIG,
                         ROOT / "scripts/ieee_tc/prepare_d699_3b_output_profile_gaps.py",
                         ROOT / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md",
                         ROOT / "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md")
        ],
        "raw_root_reserved_for_future_dynamic_run": str(RAW_ROOT),
        "safety": {
            "no_new_trace": True,
            "no_new_adapter": True,
            "no_remote_contact": True,
            "no_synthetic_wait": True,
            "no_interpolation": True,
            "append_only": True,
        },
        "source_plan_sha256": p.check_plan(),
    }
    (OUT / "prepare.json").write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n")
    (OUT / "target_manifest.json").write_text(json.dumps(target_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": True,
        "campaign": "D699",
        "output": str(OUT),
        "missing_classes": len(coverage["missing_classes"]),
        "measurement_waves": len(waves) - 1,
        "requests": sum(len(w["requests"]) for w in waves),
        "launchable": False,
    }, indent=2))


if __name__ == "__main__":
    main()
