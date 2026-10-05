# D238 retry4--6 experiment-integrity audit

**Date:** 2026-10-06  
**Input:** `D238_RETRY456_DIAGNOSTIC_AUDIT.json` and its three immutable raw runs  
**Audit mode:** deterministic evidence checks plus fresh same-family reviewer request

## Overall verdict: WARN (diagnostic-only; reviewer status pending)

The three raw artifacts exist, their bounded request counts agree, and cleanup
evidence is present. The scope and mechanism coverage are insufficient for a
formal experiment: no run contains a typed proactive decision, physical KV
reservation, independent correctness denominator, or a paired Full/CapacityOnly
counterfactual. Any fresh reviewer that does not return within the audit window
will be recorded as `BLOCKED`, never converted into PASS.

### A. Ground-truth provenance: PASS for the stated evaluation type

These runs are explicitly `self_supervised_proxy` diagnostics. They claim only
native terminal/generation/resource contracts, not semantic adapter correctness.
The result metadata keeps `eligible_correctness=false`; no model output is used
as an independent ground truth. This supports bounded execution only.

### B. Score normalization: PASS for claim boundary

TTFT, E2E, throughput, GPU-s and legacy cost/CE fields are retained as raw
diagnostic values. None is normalized by a model's own output to claim a win.
The legacy SLO display and CE fields are excluded from the claim boundary and
must not be promoted to the frozen joint SLO or G1/G2.

### C. Result existence and consistency: PASS with bounded scope

Each result and launch receipt exists; SHA-256 values are sealed in the
diagnostic audit. All three have 22 terminal requests and zero failed requests.
Launch, replay, and watchdog return codes are zero, and cleanup receipts confirm
no remaining owned targets. The run-level values in the audit are copied from
the scenario summaries; they are not inferred from the figure or from CE.

### D. Mechanism coverage: WARN / not qualified

The opt-in path was entered under the intended conditions, but all runs report
zero observed/decided/admit/defer/reject proactive admission events and
`physical_kv_reservation=false`. Retry4 had no selected GPU target; retry5 had
no native HOST source on its selected owner; retry6 had a mixed Remote→GPU plan
and correctly failed closed. This is a coverage/eligibility result, not evidence
that the native request-driven route is defective and not evidence that
admission is unnecessary.

### E. Scope assessment: WARN

The evidence is three one-model, one-seed, 22-request prefixes from a 4,000
request trace. It supports no comprehensive, robust, cross-system, SLO, tail,
lifecycle, or ranking wording. The measured TTFT/E2E values remain diagnostic
because numerical correctness and the frozen joint SLO are unavailable.

### F. Evaluation type: `self_supervised_proxy`

The runs use a frozen trace and native request/terminal/resource contracts, not
an independent semantic ground truth. This classification is appropriate only
for the bounded diagnostic claim.

## Claim impact

- bounded native execution: **supported**;
- resource cleanup and remote-service release: **supported**;
- natural in-flight trigger/coverage conditions: **supported**;
- typed proactive admission: **unsupported / needs a new diagnostic**;
- physical KV/admission correctness: **unsupported**;
- G1/G2, joint SLO superiority, lifecycle ranking, baseline comparison:
  **unsupported**;
- ordinary request-route defect: **not established**.

## Required next action

Do not repeat arbitrary short prefixes. Pre-register a legitimate frozen-trace
window and a paired Full/CapacityOnly frozen-state design. Require owner/epoch,
snapshot, proposal, decision and reservation receipts, plus backend allocator,
workspace, cancellation and release evidence. If that candidate is not
naturally produced, retain the coverage result and move to the next mainline
qualification task without inventing a decision count.

