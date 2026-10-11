## Advisory verdict: WARN

No evidence error blocks reuse of D419 as a **100-request affected-path/measurement qualification**. It does not support reuse as a complete W0 performance point, `U_ref`, G2 budget, whole-500-ID qualification, or ranking.

Review class: `same-family`; semantic findings are **provisional**. Deterministic file, hash, count, denominator, and numerical checks are reported separately.

Citation roots:

- `M=/home/qhq/serverless_llm_experiment_retry14_baseline`
- `R=/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d419_20261011`

### A. Reference/target provenance and labeling — PASS

- Target length is trace-derived as `min(source expected, 256)`, not generated from evaluated outputs (`M/docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md:53-66`; `M/scripts/ieee_tc/audit_d402_serverlessllm_path.py:68-101`).
- Request, prompt, adapter, target, native token IDs, and independent-reference identities are checked separately (`M/scripts/ieee_tc/audit_d402_serverlessllm_path.py:104-199`).
- The HF/PEFT comparison is an independently generated **bounded numerical reference**, but still model-generated proxy evidence rather than dataset ground truth. It is correctly labeled `semantic_full_pool_qualification=false`, `pass=false`, and covers only 100 requests (`R/../p2_backend_qualification/d385_20261010/7b_peft_sequence_reference_100_retry1.json:1-15`, `:2442-2444`).
- D419 retains `full_semantic_correctness_proof=false` and `full_performance_qualification=false` (`M/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/audit.json:1716-1718`).
- The 92 exact / 8 differing sequences are therefore a bounded diagnostic, not fake GT or a promoted correctness denominator. V1 explicitly does not require cross-backend token equality (`M/docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md:68-71`). The five differences without reusable first-divergence margin remain unresolved rather than being mislabeled as rounding.

Claim impact: adapter/input/target provenance is supported for these 100 requests; full semantic correctness is not.

### B. Metric denominators and normalization — PASS

- SLO thresholds come from the common D394 freeze, not the evaluated run’s maxima or averages (`M/paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json:22-32`).
- Attainment divides by the full offered population; conditional TPOT uses its explicitly labeled eligible denominator (`M/scripts/ieee_tc/audit_d402_serverlessllm_path.py:202-250`).
- The result records 47/100 TTFT, 100/100 conditional TPOT, and 47/100 joint, with `observed_feasible=false` (`M/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/audit.json:236-251`).
- `n_correct` remains null, so the physical result does not invent per-correct GPU cost (`M/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/audit.json:108-120`; `R/model7b/physical/summary.json:58-64`).
- No evaluated-output max/min/mean is used as a normalization denominator. V1’s separate Resident `Q_ref` normalization is not used here (`M/docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md:208-216`).

Claim impact: the descriptive prefix metrics are denominator-safe; they are not G1/G2 feasibility results.

### C. Result existence and numerical agreement — WARN

Deterministic core checks passed:

- All 103 unique entries in `evidence_manifest.json:3-519` existed and matched both recorded byte count and SHA-256 when checked.
- Direct replay recomputation agreed exactly with `audit.json`:

  - 100 contracts, arrivals, terminals, and responses; zero failures.
  - 29 actual adapter IDs.
  - 17,369 target/native output tokens.
  - TTFT mean/P95: `18634.907506 / 74980.162867 ms`.
  - TPOT mean/P95: `25.349100 / 27.612873 ms`.
  - Mean E2E: `23055.655692 ms`.
  - Frozen-SLO counts: `47 / 100 / 47`.
  - Maximum E2E, TPOT, and native recomputation errors: zero.

  These values appear at `M/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/audit.json:4-121` and `:236-251`, and agree with `M/docs/ieee_tc/D419_VLLM_7B_ENVIRONMENT_PATH.md:38-53`.

- The four device intervals and four lifecycle windows both sum exactly to `1019.2586401619483 GPU-s`; all leases are closed (`R/model7b/physical/summary.json:32-80`).
- The 29 transfer rows independently total 50,650,865 wire bytes; all are prepublished sends with no request-time packing. The 29 receipts total 724,665,100 verified materialized bytes, agreeing with `audit.json:116-120`.

WARN is limited to the documentation claim that “27 CPU tests passed in 0.090 s” (`M/docs/ieee_tc/D419_VLLM_7B_ENVIRONMENT_PATH.md:72-74`). The manifest retains the test source (`M/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/evidence_manifest.json:505-507`) but no test-execution receipt/log. This does not invalidate the directly checked run evidence, but that test-result sentence should remain unverified until a receipt is captured after the live gate.

### D. Actually executed metric paths — PASS

- The retained command invokes `native-http-service` (`R/run7b.sh:18-24`), and the dispatcher enters that path before importing legacy replay code (`/home/qhq/serverless_llm_baselines/scripts/replay_vllm_engine_trace.py:623-627`).
- The worker installs the v0.30 native snapshot and constructs `V030NativeRequestObservation` (`replay_vllm_engine_trace.py:231-264`).
- The observation hook snapshots cumulative native IDs and engine timestamps and checks output-event LoRA name/int-ID/path (`/home/qhq/serverless_llm_baselines/scripts/ieee_tc_serverless_measurement.py:496-537`, `:540-589`).
- Worker receipts record the executed runner hash, vLLM version, snapshot schema, and observation-source hashes (`R/model7b/worker_receipts/664de20c95fc4f4287cbc13d3e060256.json:74-100`).
- The audit then revalidates service responses, native observations, worker receipts, and the common-SLO function (`M/scripts/ieee_tc/audit_d402_serverlessllm_path.py:367-399`, `:439-453`, `:545-570`).

Claim impact: TTFT/TPOT/token-count and adapter-binding fields come from the executed native path. Legacy in-process timing code was not used and should not be cited for D419.

### E. Scope and justified reuse — WARN

Observed scope is:

- One run (`audit.json:17`).
- 100-request unscaled W0 prefix from a 4,000-request source (`audit.json:97-104`).
- 29 actually exercised logical adapter IDs (`audit.json:84`; `R/input_identity.json:2-16`).
- Four distinct adapter-weight hashes across those 29 IDs (`audit.json:259-312`).
- Nominal pool: 500 logical IDs, not 500 distinct trained weights (`M/docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md:39`, `:68-71`).

The files label this correctly: prefix-only, no comparison or optimum (`audit.json:229-250`). D419 also explicitly notes that the older complete replays exercised only 132 IDs and do not prove all 500 (`M/docs/ieee_tc/D419_VLLM_7B_ENVIRONMENT_PATH.md:77-80`).

Safe reuse:

- Exact trace/prefix identity and target derivation.
- Current vLLM 0.30/config/toolchain/source identities.
- RR routing, pre-ready ingress, remote published-object resolution, native timing, adapter-binding, and physical lease mechanisms.
- The reported 100-request descriptive observations.
- D394 common thresholds.
- The bounded 29-ID/four-weight-class HF/PEFT comparison.

Unsupported reuse:

- Complete 4,000-request W0 performance.
- `N_correct`, G1 feasibility, `U_ref`, `Q_ref`, 75% G2 budget, or rankings.
- Whole-500-ID engine execution.
- Cross-scenario optimum or robustness claims.
- D395/D396 as a resource reference; their timing/resource-contract defects remain explicitly disqualifying (`M/docs/ieee_tc/D397_RESIDENT_AUDIT_CORRECTION.md:7-20`, `:49-57`).

The single-run V2 amendment requires one complete preregistered qualifying reference—not three new repeats (`M/docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md:19-33`). No extra repetition or stricter cross-backend equality requirement should be introduced by this audit.

### F. Evaluation type — PASS

- D419: **real-system fixed-trace contract/measurement qualification**, closest checklist category `self_supervised_proxy` because no task-level semantic GT exists by design. It is not simulation-only.
- D385 comparison: explicitly bounded `synthetic_proxy`/independent numerical-reference evidence.
- No `real_gt` or human evaluation is present.

The HTTP config labels the run `qualification_only`, `affected_native_path_qualification`, `formal_run=false`, and numerical correctness unqualified (`R/http_7b_100.json:16-22`, `:31-32`).

### Bounded post-gate checks

After the current live gate ends:

1. Preserve a test-execution receipt if the “27 tests passed” sentence will be retained.
2. Verify the complete result’s source/config/protocol hashes and `full4000`, unscaled W0 identity.
3. Report actual distinct executed adapter IDs separately from the nominal 500-ID pool.
4. Keep failures in the 4,000 offered denominator and require closed physical leases.
5. Do not promote any result with null `n_correct` to G1/G2 or per-correct resource claims.
6. Do not require cross-backend exact sequence equality beyond frozen V1; retain and disclose mismatches.

Limitation: the skill-referenced shared files `reviewer-independence.md`, `experiment-integrity.md`, `reviewer-routing.md`, `review-tracing.md`, and `integration-contract.md` were absent under `/home/qhq/.codex/skills/shared-references/`. Their additional policy details could not be applied.
