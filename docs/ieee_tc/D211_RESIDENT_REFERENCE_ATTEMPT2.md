# D211 Resident reference attempt 2 — protocol-invalid partial run

**Status:** rejected as a Resident-vLLM reference; retained as an incomplete
qualification attempt. It is not used for (U_{\rm ref}), G2 budget selection,
SLO calibration, or a system ranking.

## What was tested

The candidate used four dedicated vLLM 0.30.0 runtimes on the existing 7B W0
trace (4,000 planned requests, 500 logical adapters, seed 42, and the frozen
`fixed_length_greedy_v1` contract). Handoff, hierarchical residency, dynamic
forwarding, scale control, and effective-capacity admission were disabled in
the candidate YAML. The measured movement-owner limit was restored to 3 after
the first prelaunch rejection; this is a qualification contract, not tuning.

The immutable candidate configuration SHA is
`da13a1ec5c9e0954138ec9287b7ff9c341ab718ea7985ed07609f0c57c19bae9`.
The run used plan SHA
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` and
metric-protocol V1 SHA
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.

## Why it is invalid as the independent reference

The run still entered `faaslora_full` through the IEEE native deployment path
with `routing_policy=ieee_confirmed`. Thus its selected replica and dispatch
path used Prime's confirmed-tier routing/physical ownership machinery, even
though the other Prime mechanisms were gated off. The service log identifies
the stack as `ResidencyManager + PreloadingManager + scale-aware` and records
`ieee-activation-*` replicas. This is not the protocol-defined ordinary
Resident-vLLM path, which must use a native vLLM cache and ordinary routing
without Prime planner, confirmed-tier propagation, handoff, or admission.

Continuing the replay would contaminate the reference GPU lifecycle and tail
latency with the very mechanism that G2 is intended to compare. The run was
therefore stopped at the first semantic audit finding rather than converted
into a budget reference.

## Partial evidence retained

The attempt reached the serving phase with four physical GPUs held by the
service cgroup. The replay publisher recorded 1,642 requests received and the
service ingress recorded 1,642 received/dequeued requests; the last live line
reported 1,624 successful completions and zero request failures before the
stop. The planned denominator remained 4,000. These are observed partial
counts only, not a completion or performance result.

The gated launch ended with `service_returncode=-15`,
`replay_returncode=-2`, `classification=safety_abort_unattributed`, and
`watchdog_returncode=1` because the forced stop did not permit a positive
native-context-release proof. The physical GPUs were subsequently observed
empty and the owned service scopes were gone. The watchdog, launch receipt,
replay, ingress, and service logs remain under:

`results/ieee_tc/p2_backend_qualification/d211_20261004_resident_ref1/7b_resident_ref1/launch.launch.attempt2_semantic_invalid/`

The corresponding launch receipt SHA is
`f2eb5c1aac936cdf23df59ff8f6008000ff67ae9f7d8b35a43cc7318dc9ee0e4`.
The earlier owner-binding rejection remains separately archived as
`attempt1_binding_mismatch`; neither attempt is a successful reference.

The remote artifact services were stopped only after local GPU/scope release.
The exact transfer directories and monitor log were copied to
`results/ieee_tc/p2_backend_qualification/d211_20261004_resident_ref1/remote_transfer_journal_attempt2/`.
Post-stop state records both artifact services and the monitor as inactive;
the remote NIC remained 1000/full. No remote packing or reconfiguration was
performed.

## Decision and next step

This is a **protocol/qualification rejection**, not evidence that vLLM or
Prime is slow. No figure is generated: the plan requires a status table for an
incomplete/invalid candidate rather than a misleading performance plot.

The next reference candidate must be a separately identifiable ordinary
vLLM execution path whose routing, cache, and resource lifetime do not import
`ieee_confirmed` or `faaslora_full` control semantics. It must first pass a
read-only semantic gate, then a short qualification replay, before any full
three-run Resident reference series is started. The failed attempt is not
deleted, reused as a denominator, or used to relax the frozen V1 protocol.
