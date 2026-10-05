# D227: explicit admission-gate diagnostic configuration

Status: the original configuration/CPU-only pre-run note is retained below for
provenance; the bounded execution and post-run interpretation are appended in
the D237 audit.  This file is not a formal performance result.
Date: 2026-10-05 (Asia/Shanghai)

## Purpose

D226 showed that D219's `tc_ieee_full` configuration omitted
`effective_capacity_admission_enabled`.  Both the runner and
`ResourceCoordinator` therefore used their documented default `false`, and the
D219 metadata correctly reported zero admission decisions.  D227 makes this
single mechanism gate explicit and uses a bounded 22-request prefix to test the
admission path before authorizing another full 7B replay.

This is a diagnostic configuration, not a performance result, SLO result, or
G1/G2 candidate.  It does not change the IEEE equations, model, vLLM backend,
trace, adapter subset, generation contract, service/preparation profiles, or
resource envelope.

## Intended differences from D219

| Field | D219 | D227 |
|---|---|---|
| `resource_coordination.effective_capacity_admission_enabled` | absent, runner default `false` | explicit `true` |
| `workload.total_requests` | 4,000 | 22 bounded diagnostic requests |
| output directory | `.../d219_20261005/...` | `.../d227_20261006/...` |
| NVMe cache directory | D219-owned path | fresh D227-owned path |
| memory-backed HOST path | D219-owned path | fresh D227-owned path |
| scenario name | `tc_ieee_full` | `tc_ieee_admission_gate_22` |

All other parsed YAML values are identical.  The D219 file and its result tree
remain immutable.

## CPU-only audit

The audit performed the following checks without importing the GPU service:

1. Both YAML files parse successfully.
2. A recursive comparison found no value changes outside the five operational
   fields above; the scenario name is an additional identity-only label.
3. The merged scenario coordination map used by
   `scripts/run_all_experiments.py` contains the top-level
   `resource_coordination` values because the scenario has no overriding
   `resource_coordination` map.
4. The merged feature-gate metadata therefore evaluates
   `effective_capacity_admission_enabled=true`.
5. D227 output, NVMe, and `/dev/shm` paths were absent before this audit; no
   workspace or result directory was created.
6. The old D219 configuration hash is
   `5072d681dc961a44b03ce9f3794b5c13d878f902c6f962cfba13999d5d3a0b63` and the
   new D227 configuration hash is
   `76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73`.

The source-level gate is unchanged and fail-closed: when true,
`ResourceCoordinator.prepare_for_inference` calls
`evaluate_gpu_admission` and records admit/defer/reject decisions; when false,
it uses only the basic available-memory check.  This audit confirms intended
control flow but does not establish that the 22-request trace will produce
pressure or any particular decision count.

## Run gate and acceptance

D227 may be launched only after the unchanged physical-memory, disk, GPU-idle,
remote-health, resource-domain, and numerical-correctness preparation gates
pass.  The run must remain bounded at 22 requests and use fresh owned scopes.
After cleanup, the result is admissible only if request identity, fixed-output
native token contract, physical allocation/release, dispatch fields, remote
content SHA, and admission decision metadata are complete.  It remains
diagnostic even if all checks pass.  A full 7B replay is considered only if the
bounded run supplies the pre-registered admission evidence without a new
correctness or lifecycle failure and the host again passes the unchanged
startup gate.

No baseline, 3B, Resident budget, SLO threshold, figure, or ranking may be
derived from this configuration alone.

## D227 retry2 execution record (2026-10-06)

The first launch attempt is retained as a protocol/launcher failure: the
guarded environment did not contain `FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE`,
so the input setup rejected the run before model construction.  It produced no
GPU service and is not a performance observation.  The retry used the exact
same config and trace, with the approved prepublished read-only gzip delivery
cache enabled on the remote artifact node.  Its launch receipt passed and the
service, replay, watchdog, and cleanup return codes were all zero.

| Item | Evidence |
|---|---|
| Plan SHA | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| D227 config SHA | `76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73` |
| Trace | W0 prefix 22 of the immutable 4,000-request trace; source SHA `efb903254fdcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`; view SHA `452bbb124b7ad73e1f337adf58e20fabc797ab74d06cdc6499e534ee3b0e39d0` |
| Result SHA | `3f3f70f678fc196d02f7345aded579e0528d9806057b5f8ccc40b349a98ea656` |
| Launch SHA | `d8e2ae1b3c34909e043fa0981e214b8de395a6667a69feb07363de89c1d388d8` |
| Correctness/contract | 22/22 completed; native fixed-length output contract matched for all terminals; no request error |
| Scope | bounded diagnostic only; `formal_comparison_qualified=false` |

The run also shows why the explicit YAML gate cannot yet be interpreted as
effective native GPU admission.  The scenario metadata has
`effective_capacity_admission_enabled=true`, but the native route recorded
zero GPU-admission decisions, zero admits/defers/rejects, and zero physical KV
reservations.  Every request's source-admission record has
`physical_capacity_qualified=false`; the native pending receipt explicitly
declares `physical_kv_reservation=false`.  Thus the run is evidence that this
bounded request-driven prefix produced no proactive HOST→GPU candidate for
the configured admission path.  It is not evidence of a request-route
integration defect, and it does not qualify the gate as an effective
physical-admission policy or show that the policy is ineffective.

The user-facing latency is correspondingly dominated by startup and queue
wait (mean TTFT 101,272.7 ms; mean dispatch/admission wait 100,514.4 ms),
while mean service TTFT is 758.2 ms.  These values must not enter a baseline
ranking.  The physical deployment journal reports 348.72 owner-union GPU-s;
the separate 602.51 GPU-s infrastructure billing field includes a 300-s idle
tail and is not the lifecycle metric.  All allocation journals were released,
and the launch receipt confirms native GPU-context release.

The detailed root-cause analysis, source-line evidence, and next action are in
`D237_D227_RETRY2_ADMISSION_AUDIT.md`.  No full replay is authorized until a
first-principles native physical-capacity transaction is integrated and passes
a new bounded pressure diagnostic.

### Semantic boundary correction (D238)

The final sentence is a qualification requirement, not a claim that ordinary
request-driven loading must invoke the proactive E(t) budget.  The IEEE
semantics give request-driven loading priority over proactive preparation.
D227 submitted no proactive residency candidate, so its zero decision count is
insufficient coverage of the proactive path.  See
`D238_ADMISSION_SEMANTIC_REASSESSMENT.md`; no raw D227 artifact or scope label
is changed.
