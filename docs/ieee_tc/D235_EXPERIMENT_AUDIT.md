# D235 experiment-integrity audit

**Audit input:** D235 scale-up diagnostic and the retained D219 7B JSON  
**Reviewer:** fresh same-family GPT-5.6-Sol, ultra; provisional independence  
**Date:** 2026-10-05

## Overall verdict: WARN — valid diagnostic evidence, not a formal comparison

### A. Ground-truth provenance — PASS

The D235 numbers are recomputed from the retained D219 request records and
scale-up events. They are system-serving observations (confirmed routing
snapshots and native source tiers), not semantic model ground truth. No value is
derived from a model output and called a correctness reference.

### B. Score normalization — PASS

The audit uses absolute tier counts, counts/denominators, raw TTFT and the
stored result SHA. No metric is divided by a model-specific maximum, minimum or
mean.

### C. Result existence and numerical consistency — PASS with protocol caveat

The result path exists and its SHA-256 is
`e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`.
The three scale-up events, activation times, 3,003 scale-up-runtime requests,
confirmed-dispatch count, first-20 tier counts, and TTFT values in D235 match
the JSON. The Plan SHA is recorded in D235 but is not re-derived from the
reviewer's independent artifact set; this does not affect the JSON-derived
counts.

### D. Metric activation — WARN (vacuous legacy labels)

The native confirmed path does not publish the legacy handoff plan. Therefore
`scaleup_first_service_*` is mechanically zero and cannot be interpreted as a
failed first-service experiment. The authoritative `confirmed_dispatch_snapshot`
and `readiness_tier_before_dispatch` fields are populated and are the valid
diagnostic fields.

### E. Scope — WARN by design

This is one exploratory 4,000-request 7B run with three scale-up events. The
JSON explicitly marks it as non-formal, with no effective physical-capacity
admission qualification and no common-warm reference. It supports the causal
diagnostic in D235 only; it does not support superiority, G1/G2, SLO, or
baseline-ranking claims.

### F. Evaluation type — `self_supervised_proxy`

The run uses real native GPU/file-owner telemetry and a frozen serving trace,
but has no semantic dataset ground truth. It is therefore a serving-system
diagnostic, not a model-accuracy evaluation.

## Claim impact

- Confirmed dispatch is recorded before adapter resolution: **supported as a
  diagnostic claim**.
- First-service legacy counters are zero: **not interpretable on this path**.
- New replicas often dispatch with non-GPU source tiers: **supported as a
  descriptive diagnostic claim**.
- Prime satisfies G1/G2 or beats a baseline: **unsupported by D235**.

