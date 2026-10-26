# D889 — process-owned file receipts and the current measurement boundary

## Paper-level status

The latest PrimeLoRA full replay is not yet an accepted result: the 7B run
completed all 4,000 offered requests and satisfied the native generation
contract, but its joint TTFT/TPOT attainment was 3,750/4,000 (93.750%).
Therefore this checkpoint does not claim G1/G2 qualification, does not freeze a
new performance point, and does not advance to the external baseline matrix.

The current implementation work addresses the ownership boundary that can make
file preparation and dispatch observe different states. A separate spawned
process remains the sole owner of the managed file ledger. A cancelled caller
now rejoins the exact command receipt, adopts the returned event before
re-raising cancellation, and cannot cause a second reservation. File-change
notifications are delivered from the child receipt rather than through a
callback executed while the child owner is holding its lock. A value-only
facade provides the same context lifetime for the subsequent request-path
migration; it is not enabled in the production runner by this checkpoint.

## Independent artifact-node disk rule

The artifact-only node does not use the inference host's fixed 150/100-GiB
start/stop rule. Its read-only gate computes

\[
D_{\mathrm{free},f}\ge R_f+\left\lceil1.5(P_f+L_f)\right\rceil,
\qquad P_f=\sum_j c_j s_j^{\mathrm{remaining}},
\]

for every participating filesystem, and separately reports free inodes and
the required quota audit. `c_j` is the service-wide concurrent packing bound,
`s_j^{remaining}` is the measured upper bound on one remaining temporary
archive, `L_f` is the frozen log/cleanup growth bound, and `R_f` is the
independent safety reserve. The gate neither changes transfer concurrency nor
deletes the unique artifact set. The inference host's 150/100-GiB and memory
protections are unchanged.

The focused preflight tests for both policies pass (5 tests). No remote service
was restarted or reconfigured, and no inference result was altered.

## Evidence and next step

The process receipt and facade checks pass 11/11, including spawn identity,
atomic publication/acquisition, cancellation rejoin, single event delivery,
pidfd retirement, worker death, and child-owned context lifetime. The code is
backed up at `backup/d889-process-receipt-20261026`.

The next evidence required before any GPU replay is a complete migration of
the request, preparation, and release call sites to this facade. Until that
migration is complete, the old in-process owner remains the active production
path; mixing the two owners would invalidate the resource and latency
comparison and is explicitly not used.
