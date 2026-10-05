# D241 — Canonical physical-resource sidecar gate

Date: 2026-10-06

Status: **implemented and unit-tested; no GPU replay**.

## Scope

D240 found that the old `infra_gpu_seconds_total` value is a logical billing
projection and cannot support the frozen G1 physical owner-union metric. This
change implements the measurement gate without changing any existing result
file, trace, adapter subset, threshold, or paper source.

## Sealed evidence

After the guarded runner writes its result JSON and the outer launch wrapper
has completed service cleanup, `PhysicalGPUDeployment.finalize()` now writes
`physical_resource_summary.json` beside `physical_deployment/summary.json`.
The sidecar records:

- `physical_gpu_seconds_total` and the observed/correct-request views copied
  from the finalized owner-union report;
- completion, denominator, interruption, and open-lease fields;
- every allocation-journal path and SHA-256 digest from the reducer;
- the request-terminal digest and finalized summary digest; and
- the exact result-file path and SHA-256 digest, when the runner has bound it.

The sidecar explicitly lists `infra_gpu_seconds_total` and
`instance_lifecycle_log` as rejected legacy fields. A missing result file is
recorded as missing evidence; it is never replaced by a synthetic value.

The runner binds the scaled result path before model execution. Scenario
metadata publishes both the summary path and sidecar path, with the canonical
field name `physical_gpu_seconds_total`. The result JSON is not rewritten after
finalization, so the sidecar is the immutable cross-file binding.

## Publication gate

The formal V2 Fig. 9 entry point (`plot_paper_figures.py --figure
fig9_v2_ablation --formal-matrix`) now rejects a record unless the sidecar is a
regular file with the expected contract, a complete positive owner-union
measurement, no open leases, a complete request denominator, matching summary
SHA-256, and a matching result-file path and SHA-256. Exploratory loading keeps
its previous behavior; it cannot be used as a formal G1 figure.

## Verification

The focused physical lifecycle suite passes 32/32 tests, including a new test
that checks the canonical field, release-complete denominator, summary hash,
and result-file hash. Python compilation and whitespace checks pass. A broader
launch suite was also run; its seven failures are pre-existing environment
limitations (NVML unavailable in the current Python process and the optional
`vllm` test dependency absent), not failures of this sidecar change.

No model, remote artifact service, baseline, or formal replay was started by
this gate. The next allowed action remains the existing PrimeLoRA 7B
qualification/optimization path, subject to this physical-resource gate.
