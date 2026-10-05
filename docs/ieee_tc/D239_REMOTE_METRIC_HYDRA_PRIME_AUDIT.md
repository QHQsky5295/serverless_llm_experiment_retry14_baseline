# D239 — Remote metric contract, 100-Mbps interpretation, HydraServe alignment, and PrimeLoRA offline audit

Date: 2026-10-06

Status: **read-only protocol and result audit**. No GPU replay, remote service
startup, delivery-cache mutation, baseline run, or paper-source edit was done.
This document does not replace the frozen metric protocols and does not promote
any historical run to a formal G1/G2 point.

## 1. Inputs and precedence

The calculation precedence is:

1. `PrimeLoRA-PLAN.md`, SHA-256
   `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`;
2. `METRIC_PROTOCOL_FROZEN_V1.md`, SHA-256
   `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`;
3. `METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md`, SHA-256
   `485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`;
4. `METRIC_PROTOCOL_FINAL_V2.md`, SHA-256
   `5878ed9ce4494edb02916962ecb588a2b32a5cf59e66211521b90b9af95a65c9`;
5. the state evidence in `D233_REMOTE_BOUNDARY_RECHECK.md` and the 7B
   old/new comparison in `D234_7B_OLD_NEW_PRIME_AUDIT.md`.

This document freezes the interpretation of those rules for the next PrimeLoRA
qualification step. It makes no change to a run's execution key, denominator,
threshold, or result label.

## 2. Decision in one paragraph

The primary question is not whether PrimeLoRA produces the smallest raw TTFT in
every condition. It is whether, for the same offered work and correct native
generation, it (G1) uses less physically held GPU time while meeting a common
joint TTFT/TPOT service requirement, and (G2) has lower scale-out/churn tail
TTFT under the same GPU-time budget without losing joint SLO attainment. Raw
TTFT, TPOT, E2E, completion, throughput, GPU-s, goodput, transfer volume, and
stage spans remain visible. CE and prices are secondary sensitivity analyses.

The request path uses one immutable, prepublished gzip object set for every
system. A request-triggered remote read/send/receive/write/extract/migration is
real service work and remains in the user timeline. One-time cache publication,
administrator work, full-pool scans, cleanup, and request-external packing are
recorded as deployment evidence but do not enter request latency. This boundary
preserves a real PrimeLoRA mechanism benefit—avoiding a miss can avoid a real
transfer—without allowing any system to subtract deployment work from E2E.

## 3. Remote delivery and the 100-Mbps observation

### 3.1 What is known

The read-only D233 recheck observed:

| endpoint | interface | negotiated state | errors observed |
|---|---|---|---|
| artifact node | `eno1` | 1000 Mb/s, full duplex | zero reported RX/TX errors |
| inference host | `eno1np0` | 1000 Mb/s, full duplex | zero reported RX/TX errors |

The approved immutable caches contain 500 artifacts per model. The 3B and 7B
archives contain 1,160,474,851 and 420,740,410 bytes respectively, for a total
of 1,581,215,261 bytes (about 1.47 GiB). Both manifests identify
`published_gzip_delivery_v1`, gzip level 9, and
`preparation_outside_inference=true`.

The historical D77 record observed 100 Mb/s/full on the remote side, with the
peer advertising only 10/100 modes. The exact switch-port, cable, driver,
autonegotiation, or NIC cause was not established. No later evidence proves that
the remote adapter is physically a 100-Mbps device. D78 and D233 observed
1000/full after a restart/readback. Therefore the only defensible conclusion is:

* D77 is a distinct historical link-negotiation state, not a relabelable 1-Gbps
  result and not proof of a permanent 100-Mbps hardware limit;
* formal current runs must record both endpoints' negotiated rate, duplex, MTU,
  route, error counters, and achieved application throughput;
* a run that unexpectedly negotiates 100 Mb/s is either an explicitly labelled
  0.1-Gbps sensitivity point or an environment-invalid run, never silently mixed
  with the 1-Gbps main point.

For a fixed wire payload (b), the ideal serialization lower bound is

\[
t_{wire}(b,R) \geq \frac{8b}{R}.
\]

Holding all else constant, 100 Mb/s has a ten-fold larger lower bound than
1,000 Mb/s. The observed user latency can be larger because of TCP behavior,
remote disk reads, gzip extraction, concurrency, and queueing. Thus the old
state can materially affect a cold/miss-heavy request, but it cannot be used to
claim that PrimeLoRA's current result represents a datacenter 100-Gbps or
400-Gbps deployment.

### 3.2 Sensitivity contract

The formal network sensitivity uses the existing application-layer aggregate
limiter and the frozen artifacts. It reports configured MiB/s, achieved MiB/s,
wire bytes, injected wait, concurrency, and both NIC states. The planned points
are 0.1, 0.25, 0.5, and 1.0 Gbit/s-equivalent plus a `no-delay` diagnostic upper
bound. The `no-delay` point is not a physical 100-Gbps measurement. The old D77
100-Mbps observation is not substituted for the controlled 0.1-Gbps point.

The interpretation rule is interaction-aware: if PrimeLoRA's advantage remains
under all measured bandwidth points, the mechanism is not limited to the current
link; if the advantage grows as bandwidth falls, that is a legitimate
remote-miss/transfer sensitivity and must be stated as such; if a ranking flips,
the paper must report a trade-off rather than hide the flip.

## 4. Exact latency boundary

For request (r), the offered-arrival clock starts at (a_r), and the normal
timeline is (a_r\rightarrow e_r\rightarrow d_r\rightarrow f_r\rightarrow
l_r\rightarrow c_r). The protocol computes

\[
TTFT_r=f_r-a_r,\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\;(n_r\geq2),\qquad
E2E_r=c_r-a_r.
\]

The following request-caused spans are included in those user metrics:

* remote object open/read/send and socket back-pressure;
* local receive, required write, gzip extraction, checksum/materialization;
* cache miss, tier migration, eviction, re-fetch, and contention selected by the
  serving system;
* queue/admission/dispatch delay and all GPU holding until verified release.

The following are outside the request result but must have a separate receipt:

* one-time creation of the shared read-only gzip delivery cache before the
  common deployment notice;
* administrator configuration/restart, complete-pool scans, log collection and
  cleanup;
* unrelated remote work and deliberate packing/compression stress tests;
* request-external dynamic packing.

The formal path therefore forbids request-time fallback to dynamic packing. A
system may legitimately fetch fewer objects or fewer bytes because it has better
residency or routing; the comparison must not equalize those events. Conversely,
an omitted request transfer, base-model fallback, or missing span makes the run
ineligible for the primary ranking rather than a PrimeLoRA advantage.

Cross-host service spans are not subtracted across unsynchronized clocks. The
client-side E2E identity remains the authoritative user metric, while server and
client spans are reported in their own clock domains with a `clock_id`.

## 5. HydraServe alignment: what is adopted and what is not

The HydraServe NSDI'26 paper states that it derives TTFT and TPOT SLOs from warm
requests, using five times warm TTFT and two times warm TPOT in its main setting;
it reports TTFT-SLO attainment, TPOT, cold-start behavior, and cost. It also
reports that its cost is proportional to GPU memory-time. The official source
repository describes a modified vLLM 0.4.2 implementation, remote storage, and
pipeline-based cold-start/consolidation mechanisms.

Sources:

* [HydraServe NSDI'26 paper](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_lou_prepub.pdf),
  §8.3 and Fig. 9–14;
* [HydraServe official repository](https://github.com/LLMServe/hydraserve),
  README and artifact instructions;
* [HydraServe USENIX page](https://www.usenix.org/conference/nsdi26/presentation/lou).

The alignment decision is therefore **partial and deliberate**:

| HydraServe practice | PrimeLoRA decision | Reason |
|---|---|---|
| warm-derived TTFT/TPOT thresholds | adopt 5×/2× as a disclosed reference | directly tests cold-start service quality |
| TTFT SLO attainment | report, but use joint TTFT/TPOT attainment | a TTFT-only pass can hide decode starvation |
| fixed 1024-token warm input in HydraServe | use frozen trace length quartiles | our existing workload is variable-length; changing it would break reuse and fairness |
| GPU memory-time cost | report only when its resource ownership is observable | an occupied whole GPU is not released merely because some memory is unused |
| HydraServe's application-specific relaxations (for example, summarization) | do not silently copy them | they are workload assumptions, not universal constants |
| CE (=1/(L C)) | retain as secondary sensitivity only | it allows price and latency compensation and is not HydraServe's universal primary metric |

Thus the paper can accurately say that PrimeLoRA follows the same service-quality
logic as HydraServe—cold-start latency must be judged against a warm reference
and resource cost must be shown—but it cannot say that the two experiments are
identical or that CE is an established serverless-LLM standard.

## 6. Frozen primary comparison rules

The following rules are the sole eligibility contract for future PrimeLoRA and
baseline comparisons:

1. Use the existing 7B/3B artifacts, trace, subset, prompt hashes, and
   `fixed_length_greedy_v1`; do not regenerate the pool.
2. Preserve (N_{plan}), (N_{arrived}), (N_{submitted}), (N_{terminal}),
   (N_{correct}), and (N_{good}). A failed or incomplete request stays in
   the denominator.
3. Correctness requires normal terminal status, correct adapter identity,
   exact native token count, and complete reliable timing/resource evidence.
4. Calibrate common warm references once per model and input-length group using
   the final common vLLM backend and GPU-ready adapters. Use
   \(θ^T=5T^{warm}\) and \(θ^P=2P^{warm}\) as disclosed reference
   multipliers, not as an industry standard.
5. G1 is the minimum
   \(R=U/N_{correct}\) among points with all requests correct and
   \(A_{joint}\geq0.95\), where
   \(U=\sum_d\int a_d(t)dt\) is the union of verified physical-GPU holding
   intervals.
6. G2 is absolute P95 user TTFT in W1/W2 under a pre-registered common GPU-time
   budget and the same correctness/joint-SLO constraints. The Resident-vLLM
   reference supplies the budget; it is not a per-system relaxation.
7. Report average/P95/P99 TTFT, TPOT, E2E, throughput, GPU-s, goodput,
   transfer counts/bytes, stage spans, and SLO attainment even when a point is
   not eligible for G1/G2.
8. CE, price, idle-ready factors, cost/token, and Pareto curves are secondary;
   no CE-only tuning or result selection is permitted.
9. Compute run-level values first, then paired seed/block differences and 95%
   t-CIs. A 4,000-request replay is not 4,000 independent replications.
10. The display name for the ServerlessLLM implementation remains `Serverless`;
    its internal version is retained only in the manifest.

## 7. Offline PrimeLoRA evidence currently available

### 7.1 7B confirmed-snapshot change

The D210 and D219 runs are both 4,000-request seed-42 descriptive replays under
the same broad 7B workload. They are not formal G1/G2 points because the common
correctness qualification, warm thresholds, and Resident budget were not jointly
complete.

| metric | D210 ordinary Full | D219 confirmed snapshot | change (D219−D210) |
|---|---:|---:|---:|
| mean TTFT (ms) | 1,650.5 | 1,508.4 | −8.6% |
| P95 TTFT (ms) | 3,415.8 | 3,620.0 | +6.0% |
| mean E2E (ms) | 6,034.1 | 5,848.0 | −3.1% |
| P95 E2E (ms) | 12,392.5 | 12,026.0 | −3.0% |
| mean dispatch/admission wait (ms) | 938.4 | 814.1 | −13.2% |
| mean runtime-slot wait (ms) | 615.8 | 485.9 | −21.1% |
| mean control path (ms) | 325.9 | 84.7 | −74.0% |
| completed / failed | 4,000 / 0 | 4,000 / 0 | unchanged |
| lifecycle GPU-s | 4,317.01 | 4,317.05 | unchanged |

This is a partial control-path improvement, not proof that the full system is
better. The unchanged lifecycle and higher P95 TTFT mean that the next
first-principles target remains scale-out readiness/first-service tail, not
another blind routing rewrite. The result supports a hypothesis; it does not
prove G1/G2.

### 7.2 3B evolution is diagnostic, not a paired causal comparison

The existing 3B Full records show a large improvement across D112, D118, and
D138, but the runs changed configuration and implementation state and are not a
registered paired experiment. They are useful for locating the current tail, not
for claiming a causal percentage gain.

| run | completed / failed | mean TTFT (ms) | P95 TTFT (ms) | mean E2E (ms) | P95 E2E (ms) | legacy 5-s SLO | lifecycle GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| D112 Full9 | 4,000 / 0 | 271,850.7 | 507,618.9 | 277,952.6 | 515,324.8 | 2.925% | 4,819.68 |
| D118 Full11 | 4,000 / 0 | 122,146.5 | 268,626.3 | 127,878.4 | 276,169.0 | 4.800% | 4,323.37 |
| D138 confirmed Full1 | 4,000 / 0 | 12,588.6 | 45,729.9 | 17,517.6 | 52,303.0 | 39.550% | 4,331.33 |

D138 is the latest complete 3B diagnostic in this audit. Its measured means
include 8,540.8 ms dispatch/admission wait, 4,082.0 ms runtime-slot wait, and
2,762.8 ms LoRA-I/O time, while service TTFT is 4,047.8 ms. That decomposition
shows why the remaining roughly 10–13 s user TTFT cannot be explained by raw
decoder speed or by request-time packing: queue/admission and preparation are
the dominant observable terms. This is a falsifiable optimization target. It is
not yet a qualification result because the final common warm reference, native
correctness interpretation, and 95% joint SLO were not applied to these legacy
records.

## 8. Consequence for the next PrimeLoRA action

The next action remains PrimeLoRA 7B qualification and optimization. It must first
produce one falsifiable readiness/admission hypothesis, use only observables
(arrival backlog, selected-replica confirmed tier, preparation epoch age,
transfer bytes/time, slot wait, and verified GPU ownership), and validate the
hypothesis on a minimal smoke before a full frozen replay. A change is accepted
only if it improves the pre-registered primary metrics without violating native
correctness, common joint SLO, physical-release accounting, or the remote delivery
contract. If the evidence instead shows a trade-off, the trade-off is reported;
the run is not reinterpreted as a win.

No 3B optimization or external baseline comparison is authorized by this audit
until the 7B gates are closed. No conclusion here authorizes H100, NVLink,
100-Gbps/400-Gbps, or global-optimality claims.

