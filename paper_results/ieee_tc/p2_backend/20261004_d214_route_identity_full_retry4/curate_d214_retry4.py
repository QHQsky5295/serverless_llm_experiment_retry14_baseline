"""Compact, bounded curator for immutable D214 retry4 evidence."""
import csv, hashlib, json, re, sys
from pathlib import Path

root, out, launch_p, trace, subset, config, remote = map(Path, sys.argv[1:])
raw = root / "launch.launch"
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def load(p): return json.loads(p.read_text())

launch = load(launch_p)
summary = load(raw / "physical_deployment/summary.json")
terminals = raw / "physical_deployment/request_terminals.jsonl"
seen, success, errors, native, target_non256, target_sum, missing = set(), 0, 0, 0, 0, 0, 0
required = ("canonical_prompt_sha256", "source_request_sha256", "completion_token_ids_sha256", "clock_id", "instance_id")
with terminals.open() as f:
    for line in f:
        r = json.loads(line); rid = r["request_id"]
        if rid in seen: raise RuntimeError(f"duplicate request id: {rid}")
        seen.add(rid); success += int(bool(r.get("success"))); errors += int(not r.get("success")); native += int(bool(r.get("native_contract_matched")))
        missing += int(any(r.get(k) in (None, "") for k in required))
        t = r.get("target_tokens")
        if isinstance(t, (int, float)): target_sum += int(t); target_non256 += int(int(t) != 256)

journals = []
for p in sorted((raw / "physical_allocations").glob("*.jsonl")):
    events = [json.loads(x)["event"] for x in p.read_text().splitlines() if x.strip()]
    expected = ["acquire", "worker_spawn", "native_workers", "native_workers_exited", "release"]
    journals.append({"path": str(p), "events": events, "event_order_valid": events == expected, "sha256": sha(p)})
if not all(j["event_order_valid"] for j in journals): raise RuntimeError("journal order failed")

main = load(raw / "physical_deployment/main_outcome.json")
transfers = main.get("remote_transfers", [])
main_ids = {r.get("http_transfer_id") for r in transfers}
remote_rows_all = [json.loads(x) for x in remote.open() if x.strip()]
remote_rows = [r for r in remote_rows_all if r.get("transfer_id") in main_ids]
if len(remote_rows) != len(transfers) or len({r.get("transfer_id") for r in remote_rows}) != len(remote_rows):
    raise RuntimeError("retry4 outcome/remote transfer join is not one-to-one")
mech = main.get("mechanism_events", {})
text = (raw / "service.log").read_text(errors="replace")
def last(p):
    m = re.findall(p, text)
    if not m: return ()
    x = m[-1] if isinstance(m[-1], tuple) else (m[-1],)
    return tuple(float(v) for v in x)
metrics = {}
m = last(r"ttft_e2e\(avg/p95/p99\)=([0-9.]+)/([0-9.]+)/([0-9.]+)ms ttft_service\(avg/p95/p99\)=([0-9.]+)/([0-9.]+)/([0-9.]+)ms tpot=([0-9.]+)ms e2e_e2e\(avg/p95/p99\)=([0-9.]+)/([0-9.]+)/([0-9.]+)ms")
if m: metrics.update(dict(zip(("ttft_e2e_avg_ms","ttft_e2e_p95_ms","ttft_e2e_p99_ms","ttft_service_avg_ms","ttft_service_p95_ms","ttft_service_p99_ms","tpot_ms","e2e_avg_ms","e2e_p95_ms","e2e_p99_ms"), m)))
m = last(r"req/s=([0-9.]+) tok/s=([0-9.]+) slo@5000ms=([0-9.]+)% slo_goodput=([0-9.]+)rps/([0-9.]+)tok/s")
if m: metrics.update(dict(zip(("req_per_s","tok_per_s","legacy_ttft_only_slo_5000_pct","legacy_slo_goodput_rps","legacy_slo_goodput_tok_per_s"), m)))
m = last(r"scaleup_ttft=([0-9.]+)ms runtime=([0-9.]+)ms runtime_e2e=([0-9.]+)ms worker_wall=([0-9.]+)ms .*?pre_runtime=([0-9.]+)ms svc_residual=([0-9.]+)ms rpc_ovh=([0-9.]+)ms .*?gpu_ready=([0-9.]+)ms .*?arrival_late=([0-9.]+)ms dispatch_wait=([0-9.]+)ms")
if m: metrics.update(dict(zip(("scaleup_ttft_ms","runtime_ms","runtime_e2e_ms","worker_wall_ms","pre_runtime_ms","service_residual_ms","rpc_overhead_ms","gpu_ready_ms","arrival_late_ms","dispatch_wait_ms"), m)))

source_sha = {"plan_sha256": launch.get("plan_sha256"), "launch_sha256": sha(launch_p), "main_outcome_sha256": sha(raw / "physical_deployment/main_outcome.json"), "physical_summary_sha256": sha(raw / "physical_deployment/summary.json"), "service_log_sha256": sha(raw / "service.log"), "request_terminals_sha256": sha(terminals), "remote_transfer_copy_sha256": sha(remote), "config_sha256": sha(config), "trace_sha256": sha(trace), "adapter_subset_sha256": sha(subset)}
delivery = {"main_outcome_transfer_count":len(transfers),"copied_transfer_count_joined":len(remote_rows),"copied_lines_total":len(remote_rows_all),"unmatched_retained_lines":len(remote_rows_all)-len(remote_rows),"copied_bytes_written_joined":sum(int(r.get("bytes_written",0)) for r in remote_rows),"pack_performed_values":sorted(set(bool(r.get("pack_performed")) for r in remote_rows)),"delivery_modes":sorted(set(r.get("delivery_mode") for r in remote_rows)),"main_outcome_pack_performed_values":sorted(set(bool(r.get("remote_pack_performed")) for r in transfers)),"main_outcome_pack_ms_values":sorted(set(r.get("remote_pack_ms") for r in transfers)),"main_outcome_pack_duration_basis_values":sorted(set(r.get("remote_pack_duration_basis") for r in transfers)),"main_outcome_spans_additive_values":sorted(set(r.get("remote_spans_additive_to_client_total") for r in transfers))}
curated = {"kind":"d214_retry4_curated_v1", "run_id":"d214_7b_route_identity_full_retry4", "qualification":{"launch_pass":launch.get("pass"), "returncodes":{k:launch.get(k) for k in ("service_returncode","replay_returncode","watchdog_returncode")}, "native_gpu_context_release_confirmed":launch.get("native_gpu_context_release_confirmed"), "service_path_removed":launch.get("service_path_removed"), "formal_comparison_qualified":False, "correctness_qualification":summary.get("correctness_qualification"), "eligible_correctness":summary.get("eligible_correctness")}, "population":{"planned":summary["n_plan"],"terminals":summary["n_terminal"],"success":success,"errors":errors,"native_contract_complete":native,"unique_request_ids":len(seen),"target_non256":target_non256,"target_tokens_sum":target_sum,"missing_required_terminal_fields":missing}, "physical_gpu":{"gpu_seconds_observed":summary["gpu_seconds_observed"],"gpu_seconds_per_offered_request":summary["gpu_seconds_per_offered_observed"],"n_correct":summary["n_correct"],"open_lease_ids":summary["open_lease_ids"],"owner_event_count":summary["owner_event_count"],"window_gpu_seconds":summary["window_gpu_seconds"],"journals":journals}, "delivery":delivery, "mechanism":{"source_observation_stats":mech.get("_ieee_source_observation_stats",{}),"activations":len(mech.get("_ieee_activations",[])),"residency_epochs":len(mech.get("_ieee_residency_epochs",[])),"control_events":len(mech.get("_ieee_control_events",[])),"runtime_quarantine_events":len(mech.get("_ieee_runtime_quarantine_events",[]))}, "descriptive_service_metrics":metrics,"source_sha256":source_sha,"caveat":"Descriptive replay only: numerical adapter correctness is not inferred from the token contract; the legacy 5000-ms display is TTFT-only, not the frozen joint SLO; no G1/G2 or cross-system ranking."}
out.mkdir(parents=True, exist_ok=True)
(out / "d214_retry4_curated.json").write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
with (out / "d214_retry4_metrics.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["run","planned","terminals","success","native_contract","target_non256","gpu_seconds_observed","gpu_seconds_per_offered","ttft_e2e_avg_ms","ttft_e2e_p95_ms","ttft_service_avg_ms","ttft_service_p95_ms","tpot_ms","e2e_avg_ms","legacy_ttft_only_slo_5000_pct","formal_comparison_qualified"]); w.writerow(["d214_7b_route_identity_full_retry4",summary["n_plan"],summary["n_terminal"],success,native,target_non256,summary["gpu_seconds_observed"],summary["gpu_seconds_per_offered_observed"],metrics.get("ttft_e2e_avg_ms"),metrics.get("ttft_e2e_p95_ms"),metrics.get("ttft_service_avg_ms"),metrics.get("ttft_service_p95_ms"),metrics.get("tpot_ms"),metrics.get("e2e_avg_ms"),metrics.get("legacy_ttft_only_slo_5000_pct"),False])
(out / "d214_retry4_status.md").write_text(f"# D214 7B route identity-recheck full replay (retry4)\n\n- Population: {summary['n_terminal']}/{summary['n_plan']} terminal, {success} successful, {native} native-contract complete; target values different from 256: {target_non256}; unique request IDs: {len(seen)}.\n- Lifecycle: physical GPU-s observed **{summary['gpu_seconds_observed']:.6f}**, {len(journals)} journals with exact acquire → worker_spawn → native_workers → native_workers_exited → release order; open leases: {len(summary['open_lease_ids'])}.\n- Delivery: {len(transfers)} outcome transfers joined one-to-one to the copied remote journal; the file contains {len(remote_rows_all)} lines, of which {len(remote_rows_all)-len(remote_rows)} are retained out-of-run records excluded from retry4 bytes/counts. Joined records report prepublished gzip and no request-time packing; joined server bytes: {sum(int(r.get('bytes_written',0)) for r in remote_rows)}.\n- Descriptive service values: E2E TTFT mean/P95/P99 {metrics.get('ttft_e2e_avg_ms')}/{metrics.get('ttft_e2e_p95_ms')}/{metrics.get('ttft_e2e_p99_ms')} ms; service TTFT {metrics.get('ttft_service_avg_ms')}/{metrics.get('ttft_service_p95_ms')}/{metrics.get('ttft_service_p99_ms')} ms; TPOT {metrics.get('tpot_ms')} ms; E2E {metrics.get('e2e_avg_ms')}/{metrics.get('e2e_p95_ms')}/{metrics.get('e2e_p99_ms')} ms.\n- Qualification: launch and cleanup passed, but formal comparison remains **not qualified**: `n_correct` is unknown and the displayed 5000-ms SLO is TTFT-only, not the frozen joint TTFT/TPOT SLO.\n- Provenance: `d214_retry4_curated.json` contains source hashes; raw evidence remains immutable under the qualification result directory.\n")
print("curated", out / "d214_retry4_curated.json")
