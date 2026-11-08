"""Project completed-group ledger: project sealed summaries, never replay models.

This is a one-off composition of existing audit outputs, not a metric engine.
No latency is recomputed and no old artifact is rewritten. Unknown correctness
stays unknown. Outputs describe observations, not independent audit acceptance.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "result1"
sources = {}
points = []


def read(path):
    p = ROOT / path
    data = p.read_bytes()
    sources[str(p)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return data.decode()


def load(path):
    return json.loads(read(path))


def add(model, workload, group, variant, run, x, path):
    n = x["counts"].get("N", x["counts"].get("n", x.get("requests", 4000)))
    joint = x["counts"]["joint_pass"] / n
    u = x["physical_gpu_seconds"]
    metrics = x["metrics"]
    assert n == 4000 and u > 0
    points.append(dict(model=model, workload=workload, group=group, variant=variant,
                       run=run, offered=n, joint_slo=joint, physical_gpu_seconds=u,
                       gpu_seconds_per_offered=u/n,
                       p95_ttft_ms=metrics["ttft_ms"]["p95_type1"],
                       mean_ttft_ms=metrics["ttft_ms"]["mean"],
                       mean_tpot_ms=metrics["tpot_ms"]["mean"],
                       p95_tpot_ms=metrics["tpot_ms"]["p95_type1"],
                       mean_e2e_ms=metrics["e2e_ms"]["mean"],
                       n_failed=x.get("n_failed", 0), n_correct=None,
                       independent_numerical_proof=False, independent_holdout=False,
                       source=path))


# Reuse retained external points, not the obsolete PrimeLoRA rows in these tables.
external = {}
for model, path in [
    ("7B", "paper_results/ieee_tc/main_comparison/d593_complete_7b/comparison.json"),
    ("3B", "paper_results/ieee_tc/main_comparison/d810_available_3b/comparison.json"),
]:
    external[model] = load(path)
    for workload, systems in external[model]["results"].items():
        for system, x in systems.items():
            if system != "PrimeLoRA":
                add(model, workload, "external" if system != "dLoRA" else "B6", system,
                    x["run"], x, path)

refs = {"7B": dict(external["7B"]["references"]),
        "3B": load("paper_results/ieee_tc/main_comparison/d657_3b_references/resident_references.json")}
refs["7B"].update(load("paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json"))

for workload, run, path in [
    ("W0", "D1114", "paper_results/ieee_tc/main_comparison/d1114_same_version_w0/result1/comparison.json"),
    ("W1", "D1115", "paper_results/ieee_tc/main_comparison/d1115_same_version_w1/result1/comparison.json"),
    ("W2", "D1100", "paper_results/ieee_tc/main_comparison/d1113_contract_reconciliation/result1/reconciliation.json"),
]:
    add("7B", workload, "current_resource", "Full", run, load(path)["prime"], path)

for workload, run, directory in [
    ("W0", "D1083", "d1083_same_version_w0"),
    ("W1", "D1082", "d1082_host_budget_projection"),
    ("W2", "D1084", "d1084_same_version_w2"),
]:
    path = f"paper_results/ieee_tc/p2_backend/{directory}/completion/full4000.json"
    add("3B", workload, "current_resource", "Full", run, load(path), path)

for workload, run, directory, result in [
    ("W1", "D1117", "d1117_budget_cap3_w1", "result1"),
    ("W2", "D1118", "d1118_budget_cap3_w2", "result2"),
]:
    path = f"paper_results/ieee_tc/main_comparison/{directory}/{result}/comparison.json"
    add("7B", workload, "budget_cap3", "Full", run, load(path)["prime"], path)

# Exact prescribed A1 variants; no directory-wide latest/winner selection.
ablation_7b = [(1119, "elastic_only", "ElasticOnly"), (1120, "elastic_locality", "ElasticLocality"),
               (1121, "fixed_full", "FixedFull"), (1122, "last_known", "LastKnown"),
               (1123, "no_routing", "NoRouting"), (1124, "no_handoff", "NoHandoff"),
               (1125, "r0h0", "R0H0"), (1126, "delayed_handoff", "DelayedHandoff")]
old_3b = [(1050, "no_handoff", "NoHandoff"), (1052, "capacity_only", "CapacityOnly"),
          (1053, "fixed_full", "FixedFull"), (1054, "delayed_handoff", "DelayedHandoff"),
          (1055, "no_routing", "NoRouting"), (1056, "r0h0", "R0H0"),
          (1057, "elastic_only", "ElasticOnly"), (1061, "elastic_locality", "ElasticLocality"),
          (1062, "last_known", "LastKnown"), (1063, "nohost", "NoHOST")]
for model, group, entries in [("7B", "current_A1", ablation_7b), ("3B", "historical_A1", old_3b)]:
    for number, tag, variant in entries:
        path = f"paper_results/ieee_tc/ablation/d{number}_{model.lower()}_{tag}_w1/comparison/metrics.csv"
        rows = list(csv.DictReader(io.StringIO(read(path))))
        selected = [r for r in rows if r["variant"] == variant]
        assert len(selected) == 1
        if number == 1050:
            selected += [r for r in rows if r["variant"] == "Full"]
        for r in selected:
            x = {"counts": {"N": 4000, "joint_pass": int(r["joint_good"])},
                 "physical_gpu_seconds": float(r["gpu_seconds"]),
                 "metrics": {"ttft_ms": {"mean": float(r["mean_ttft_ms"]), "p95_type1": float(r["p95_ttft_ms"])},
                             "tpot_ms": {"mean": float(r["mean_tpot_ms"]), "p95_type1": float(r["p95_tpot_ms"])},
                             "e2e_ms": {"mean": float(r["mean_e2e_ms"])}}}
            add(model, "W1", group, r["variant"], "D1047" if r["variant"] == "Full" else f"D{number}", x, path)

for number, tag, variant in [(1085,"elastic_only","ElasticOnly"), (1087,"elastic_locality","ElasticLocality"),
                            (1088,"last_known","LastKnown"), (1089,"nohost","NoHOST")]:
    path = f"paper_results/ieee_tc/ablation/d{number}_3b_{tag}_w1/completion/full4000.json"
    add("3B", "W1", "current_A1", variant, f"D{number}", load(path), path)

path = "paper_results/ieee_tc/p2_backend/d1112_same_version_w2/completion/full4000.json"
add("3B", "W2", "extra_duplicate_not_canonical", "Full", "D1112", load(path), path)

for p in points:
    p["budget75_gpu_seconds"] = refs[p["model"]][p["workload"]]["B_gpu_seconds"]
    p["within_budget75"] = p["physical_gpu_seconds"] <= p["budget75_gpu_seconds"]
    p["timing_minimum95"] = p["joint_slo"] >= .95 and p["n_failed"] == 0
    p["observed_budget_and_timing"] = p["within_budget75"] and p["timing_minimum95"]

comparisons = []
for p in points:
    if p["group"] not in ("external", "B6", "current_A1", "historical_A1") or p["variant"] == "Full":
        continue
    wanted = "historical_A1" if p["group"] == "historical_A1" else "current_resource"
    full = next(x for x in points if x["model"] == p["model"] and x["workload"] == p["workload"]
                and x["group"] == wanted and x["variant"] == "Full")
    du = (p["physical_gpu_seconds"] - full["physical_gpu_seconds"]) / p["physical_gpu_seconds"]
    dq = (p["p95_ttft_ms"] - full["p95_ttft_ms"]) / p["p95_ttft_ms"]
    da = 100 * (full["joint_slo"] - p["joint_slo"])
    comparisons.append(dict(model=p["model"],workload=p["workload"],group=p["group"],
                            full_run=full["run"], comparator=p["variant"], comparator_run=p["run"],
                            full_gpu_reduction_pct=100*du, full_p95_reduction_pct=100*dq,
                            full_joint_difference_pp=da,
                            all_three_primary_observations_better=du>0 and dq>0 and da>=0,
                            both_observed_budget_and_timing=full["observed_budget_and_timing"] and p["observed_budget_and_timing"],
                            formal_ranking_eligible=False))

# Fixed measured-point classification, not a new configuration or interpolation.
sensitivity = []
for p in points:
    for q in (.90,.95,.99):
        for b in (.5,.75,1):
            sensitivity.append(dict(run=p["run"],model=p["model"],workload=p["workload"],group=p["group"],
                                    q=q,budget_fraction=b,
                                    meets_timing=p["joint_slo"]>=q and p["n_failed"]==0,
                                    within_budget=p["physical_gpu_seconds"]<=b*refs[p["model"]][p["workload"]]["U_ref_gpu_seconds"],
                                    actual_joint_slo=p["joint_slo"],actual_gpu_seconds=p["physical_gpu_seconds"],
                                    formal_ranking_eligible=False))

# Bind the narrative's source documents without rehashing multi-GB raw journals.
for name in json.loads(read("paper_results/ieee_tc/metric_review/closure_20261108/document_sources.json")):
    read(name)
read("docs/ieee_tc/PLAN_APPROVED_20261108_QUALITY_RESOURCE_CLOSURE.md")
read("docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md")
read("docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md")
read("paper_results/ieee_tc/metric_review/closure_20261108/collect.py")

assert len({(p['model'],p['run']) for p in points}) == len(points)
assert len([p for p in points if p['group']=='external']) == 30
assert len([p for p in points if p['group']=='B6']) == 2
assert len([p for p in points if p['group']=='current_resource']) == 6
OUT.mkdir(exist_ok=False)
for name, data in [("points",points),("comparisons",comparisons),("fixed_point_sensitivity",sensitivity)]:
    with (OUT/f"{name}.csv").open("x") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (OUT/f"{name}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
(OUT/"manifest.json").write_text(json.dumps(dict(
    kind="completed_group_observation_ledger", source_refs=sources,
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())},
    points=len(points), comparisons=len(comparisons), fixed_point_sensitivity_rows=len(sensitivity),
    new_gpu_runs=0, raw_reaudit=False, reuse="sealed summaries; immediate input SHA captured",
    independent_integrity_review="unavailable", independent_numerical_proof=False,
    changed_primary_metrics=False, new_workload_or_weights=False,
    caveats=["All source42 known development observations; no CI or holdout claims.",
             "Current and historical A1 Full sources kept separate.",
             "D1112 is an extra duplicate, not a replacement or pooled average.",
             "Frozen q/budget slices do not change measured resource consumption.",
             "Correctness remains null; per-offered resources are not per-correct proof."]
),ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"points":len(points),"comparisons":len(comparisons),"sources":len(sources),"output":str(OUT)}))
