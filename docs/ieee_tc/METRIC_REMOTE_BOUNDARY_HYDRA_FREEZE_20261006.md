# PrimeLoRA IEEE TC: metric and remote-boundary freeze

**Status:** frozen for the next PrimeLoRA optimization cycle; this note is
subordinate to `METRIC_PROTOCOL_FROZEN_V1.md` and does not alter its nine
paper formulas.  A later change to an online SLO, resource envelope, delivery
contract, workload, or generation contract requires a new protocol version and
execution key.

## 1. Why HydraServe is a useful reference, but not a template to copy

HydraServe evaluates the problem it actually claims to solve: serverless cold
start.  It defines TTFT and TPOT as the two service objectives, calibrates
end-to-end SLOs from warm measurements, uses a global TTFT threshold of five
times warm TTFT and a TPOT threshold of two times warm TPOT, and reports the
fraction of requests satisfying those SLOs under controlled request rates and
burstiness ([HydraServe, NSDI '26, Sections 2 and 8](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf)).
It also separates cold-start stages, reports the TPOT and resource penalty of
pipeline parallelism, and uses a GPU-memory-time proportional cost ratio for
the end-to-end trade-off ([HydraServe, Figures 8--13](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf)).
The released implementation confirms that its evaluation is centered on
TTFT, TPOT, SLO attainment, and cost rather than a custom CE score
([HydraServe repository](https://github.com/LLMServe/hydraserve)).

This is a strong conceptual match for PrimeLoRA: PrimeLoRA also changes
startup/readiness, remote artifact work, scale-out, and residency, so a raw
TTFT-only claim would hide the lifecycle resource trade-off.  We therefore
adopt HydraServe's *measurement principles*—shared warm-derived TTFT/TPOT
SLOs, cold/warm phase separation, SLO attainment, tail latency, and cost or
resource trade-off—but do not copy its absolute warm numbers, A10/V100
hardware, request-rate range, or application-specific SLO doubling.  Those
values are not portable to our RTX 3090, 7B/3B, LoRA, or 100-Mbps-link
conditions.

## 2. Primary comparison objectives

Every system is first required to execute the same offered request set under
the same prompt/target/generation contract:

\[
N_{\mathrm{correct}}=N_{\mathrm{plan}},
\qquad
A_{\mathrm{joint}}\ge 0.95,
\]

where a request contributes to `A_joint` only if it is correct **and** its
end-to-end TTFT and TPOT both satisfy the common thresholds.  A failed,
truncated, empty, fallback-token, or output-contract-mismatched request is not
silently removed from the denominator.

### G1: resource efficiency at a common service level

For a valid run, let `U` be the physical-card GPU-second union over the
request-serving lifecycle (startup, ready-idle, active service, and any
request-triggered scale transition), not a logical replica count.  The
primary resource metric is

\[
G_1=\frac{U}{N_{\mathrm{correct}}}
\quad\text{(physical GPU-s/request)},
\]

minimized only among runs satisfying the correctness and joint-SLO gate.
The result is also reported as GPU-s per offered request, peak/mean physical
GPU count, and the startup/active/idle-ready decomposition.  This is the
primary test of PrimeLoRA's lifecycle-residency claim; CE or a cloud price is
not a substitute for physical resource evidence.

### G2: tail latency at a common resource budget

For each model/workload/block, the Resident reference budget is

\[
U^{\mathrm{ref}}_{\mu,w}=\frac{1}{3}\sum_{j=1}^{3}U^{\mathrm{Resident}}_{\mu,w,j},
\qquad
B_{\mu,w}=0.75U^{\mathrm{ref}}_{\mu,w}.
\]

This budget remains **unfrozen** until three complete Resident references are
available.  Under the common budget, CPU/HOST/GPU envelope, correct execution,
and `A_joint\ge0.95`, G2 is the P95 end-to-end TTFT in the pre-registered
scale-out/churn workload.  We report absolute P50/P95/P99 TTFT, TPOT, E2E,
and SLO attainment; the headline comparison is the paired P95 difference,
with the joint-SLO attainment difference checked for non-inferiority.  No
system is allowed to change the budget or SLO after seeing formal results.

## 3. Common warm-derived SLOs

HydraServe's five-times/two-times calibration is adopted as a transparent
starting rule, not as an unmeasured assertion.  For each model and frozen
input-length group, the same backend version, batch protocol, resource
envelope, and generation contract are used to obtain a common warm reference
mean (equal-weighted across the pre-registered measurement rounds):

\[
\theta_T=5\,\mu^{\mathrm{warm}}_{\mathrm{TTFT}},
\qquad
\theta_P=2\,\mu^{\mathrm{warm}}_{\mathrm{TPOT}}.
\]

The reference is shared by all systems in the comparison; each system may not
calibrate its own easier threshold.  We do not import HydraServe's
application-specific summarization multiplier because our primary trace is a
fixed mixed LoRA workload rather than three separately defined applications.
If a future workload has a separately registered application contract, its
relaxed SLO must be frozen before any formal run and applied to every system.

For request `i`,

\[
A_{\mathrm{joint}}
=\frac{1}{N_{\mathrm{plan}}}
\sum_i \mathbf 1\{\text{correct}_i\land
T_i^{\mathrm{TTFT}}\le\theta_T\land
T_i^{\mathrm{TPOT}}\le\theta_P\}.
\]

The scheduled trace arrival is the origin of user TTFT.  Service TTFT and
dispatch/admission wait are reported separately for diagnosis, but startup is
not subtracted from user TTFT when it is causally on the request path.

## 4. Remote delivery and latency accounting

The approved delivery contract is one read-only, prepublished gzip cache,
prepared once before the deployment notice and shared by all systems.  It is
not a Prime-only hot-set preparation, does not create a second decompressed
artifact pool, and does not change weights, the trace, or the adapter subset.

| Event | User E2E / TTFT | Treatment |
|---|---:|---|
| Scheduled request arrival through queue/admission and replica selection | included | part of the serving system |
| Request-triggered remote object read, network transfer, receive, decompression/materialization, and local adapter load | included | real work on the request path; report bytes, count, and spans |
| Local GPU/HOST/NVMe residency, eviction, handoff, and cache wait | included | PrimeLoRA's proposed mechanism is allowed to change these |
| Remote archive creation, compression, hashing, manifest generation, or bulk scan performed before deployment | excluded from request latency | deployment-time administration; record its wall time/bytes separately |
| Remote service start/stop, health polling, and unrelated maintenance | excluded | external experiment orchestration; any interference invalidates or stratifies the run |
| Container/model initialization caused by a request-driven scale-out | included | analogous to HydraServe's cold-start stages |

The first row through the request's first native output token defines user
TTFT.  The complete arrival-to-completion interval defines user E2E.  Stage
spans may overlap, so they are not added to reconstruct E2E by subtraction;
the canonical request timeline remains authoritative.  Remote transfer time
is therefore **not removed** merely because the physical link is shared.  A
request that avoids a remote miss is allowed to benefit from PrimeLoRA's
residency policy, while a request that has a miss pays the actual read,
transfer, receive, and materialization cost.

No formal system may perform request-time archive packing.  If a baseline
cannot consume the shared prepublished delivery contract without a private
packing step, it is a delivery-contract qualification failure or a separately
labelled diagnostic—not a number silently mixed into the main table.

## 5. The 100-Mbps link and external validity

The currently observed 100-Mbps negotiation is a property of the present
remote path, but its exact cause (NIC, cable, switch port, autonegotiation, or
configuration) has not yet been established.  We will record the endpoint
`ethtool`/switch-visible facts without changing the link during a formal run.
The physical link is part of the testbed stratum, not evidence of a 100-Gbps
or 400-Gbps deployment.

This condition can enlarge absolute remote-fetch latency and limits direct
generalization to modern datacenter fabrics.  It does **not** justify deleting
remote transfer from E2E: the transfer is a real, common scenario resource,
and PrimeLoRA's claimed advantage is precisely to reduce the number and
contention of such transfers through GPU/HOST/NVMe residency.  Different
systems may legitimately have different miss counts, bytes, and concurrency;
those quantities are reported rather than equalized away.

To bound external validity, the same frozen trace and subset will be evaluated
at application-layer aggregate limits around the measured path (including
the 0.1-Gbps-equivalent point, 0.25, 0.5, and 1 Gbps-equivalent points) and a
no-delay diagnostic upper bound.  The latter is explicitly labelled
`no-delay diagnostic`, not a measurement of a 100-Gbps network.  The primary
claim must hold at the measured-link stratum; sensitivity shows whether the
ordering depends on that bottleneck.

## 6. Supplementary metrics and CE boundary

We retain the existing CE for continuity:

\[
CE=\frac{1}{L\,C},
\qquad
\ln\frac{CE_P}{CE_B}=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P},
\]

where `L` is the frozen mean user E2E and `C` is the explicitly reported
lifecycle monetary cost.  We additionally report cost/token, SLO-goodput/$,
GPU-s/request, total/active/idle-ready GPU-seconds, Pareto latency-resource
frontiers, and idle-factor/price sensitivity.  CE is descriptive and cannot
override correctness, the common joint SLO, G1, or G2.  A custom metric is
never the sole basis for a headline claim.

## 7. Comparison checklist (must pass before a number is published)

1. Same trace, adapter subset, prompt hash, target map, generation contract,
   and offered-request denominator.
2. Same warm-derived SLO and resource envelope; no per-system SLO tuning.
3. Complete correctness and native-token audit; no trace-token fallback.
4. Actual physical GPU allocation/release and active resource limits recorded.
5. Remote delivery mode, fetch count/bytes/time, cache state, and link stratum
   recorded; prepublication work kept out of request latency.
6. P50/P95/P99 TTFT, TPOT, E2E, joint SLO attainment, GPU-seconds, and stage
   decomposition available before any CE or cost interpretation.
7. Incomplete, externally interfered, or delivery-contract-incompatible runs
   remain visible but cannot enter formal ranking.

The next PrimeLoRA optimization must use this document and the parent frozen
protocol as its metric contract.  The interrupted D288 Resident run does not
change any threshold or budget and remains quarantined.
