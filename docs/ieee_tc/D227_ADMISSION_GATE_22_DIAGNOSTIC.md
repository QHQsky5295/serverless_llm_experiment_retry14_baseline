# D227: explicit admission-gate diagnostic configuration

Status: configuration and CPU audit prepared; no GPU or remote service run.
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
