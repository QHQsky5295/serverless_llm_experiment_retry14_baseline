# D238 retry3 experiment-integrity audit

**Input:** bounded D238 retry3 diagnostic  
**Date:** 2026-10-06  
**Audit mode:** deterministic executor checks plus requested fresh-agent review

## Overall verdict: WARN (diagnostic only)

The artifact is internally consistent as a bounded native-contract diagnostic,
but it is not a formal evaluation point. A fresh same-family reviewer was
requested under the experiment-audit protocol, but the read-only reviewer did
not return within the audit window and was interrupted. The reviewer status is
therefore **BLOCKED**, not a provisional PASS. The deterministic checks below
are the only accepted checks for this report.

### A. Ground-truth provenance: PASS for the stated evaluation type

This run does not claim task correctness or model quality. It uses a frozen
trace and native terminal/generation contracts; the result explicitly leaves
`eligible_correctness=false` and `n_correct` unavailable. The correct
classification is `self_supervised_proxy` / bounded native-contract
diagnostic, not `real_gt`.

Evidence: the result metadata and deployment summary in the immutable result
file and `launch.launch/physical_deployment/summary.json`.

### B. Score normalization: PASS for claim boundary; not a performance score

The retained TTFT, E2E, GPU-second and cost fields are raw diagnostic fields.
No normalized score is used to claim superiority. CE/cost-effectiveness is
kept only as a diagnostic field and is explicitly excluded from the claim
boundary.

### C. Result existence and consistency: PASS

The result file, launch receipt, main outcome, deployment summary, terminal
ledger, ingress log and replay log all exist. Counts agree at 22 planned,
22 terminal, 22 native-contract-complete, and zero failed requests. Launch,
service, replay and watchdog return codes are zero; leases are settled and
cleanup is complete. SHA-256 values are sealed in
`D238_RETRY3_DIAGNOSTIC_AUDIT.json`.

### D. Mechanism coverage: WARN / not qualified

The run records zero proactive-admission decisions and zero physical KV
reservations, while the diagnostic itself reports
`state=not_eligible` with reason
`requires_one_initialized_ieee_runtime_and_profiles`. This is a coverage gap,
not evidence that the ordinary demand route is defective. The current run does
not provide a paired Full/CapacityOnly counterfactual or post-load physical
allocator/workspace evidence.

### E. Scope: WARN

The run is one Llama-2-7B seed-42 22-request prefix from a 4,000-request trace.
It supports only bounded execution, cleanup, and diagnostic-coverage claims;
it cannot support comprehensive, robust, SLO, lifecycle or ranking language.

### F. Evaluation type: `self_supervised_proxy`

The evaluation has no independent numerical ground truth and is intentionally
limited to native request/terminal/resource contracts. It is not a formal
performance comparison.

## Claim impact

- bounded native execution and fixed-output contract: **supported**;
- cleanup/resource release: **supported**;
- proactive-admission coverage: **not qualified / needs a follow-up diagnostic**;
- physical admission correctness: **unsupported**;
- G1/G2, SLO superiority, lifecycle ranking, baseline comparison: **unsupported**;
- ordinary demand-route defect: **not established**.

