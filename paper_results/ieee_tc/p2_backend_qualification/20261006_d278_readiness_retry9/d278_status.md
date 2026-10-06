# D278 7B readiness recheck — final diagnostic status

Date: 2026-10-06 16:xx CST  
Decision: **execution complete; readiness gate not passed; no ranking claim**

## Evidence

The post-D277 same-configuration cold replay completed the entire frozen
8-phase W0 envelope: 4,000/4,000 requests arrived, submitted, terminated and
succeeded. All 4,000 native fixed-length/token-ID contracts and dispatch-time
tier snapshots were present. The tier consistency validator reports zero
invariant conflicts. The observed tier counts were GPU 1,165, HOST 1,823,
NVMe 897 and Remote 115. Ten natural scale-up and seven scale-down events
were observed.

The earlier D276 capacity-wait exception did not recur: there are zero
failure observations, zero terminal errors and zero open lease IDs. The
physical owner-union sidecar is complete at 15,349.592341 GPU-s. This quantity
is retained only as a diagnostic resource observation because `n_correct` is
unknown and the run is explicitly exploratory (`formal_run=false`).

The first-service sample count is still 10, below the pre-registered minimum
of 20. Therefore D278 does not qualify dispatch-time readiness and cannot be
used for G1/G2, SLO ranking, baseline comparison or a lifecycle claim. The
correct interpretation is narrower: the D277 lease-to-plan witness repair is
runtime-compatible under this complete replay and removes the D276 observed
failure, but it has not yet supplied enough first-service observations for the
readiness evidence gate.

Descriptive values are retained for diagnosis only: mean/P95 user TTFT
6,895.5/22,698.6 ms, mean/P95 E2E 11,204.6/27,220.7 ms, service TTFT
681.7 ms, and mean TPOT 36.84 ms. The legacy 88.28% SLO display is not the
frozen joint-SLO qualification because numerical adapter correctness is not
established. Scale-up first-service mean TTFT is 51,750.7 ms for n=10, with
zero GPU-ready samples and zero planned-adapter matches; this is descriptive,
not a ranking result.

## Resource and remote closure

Watchdog minima/maxima were 94.23 GiB host `MemAvailable`, 20.47 GiB service
memory peak, zero service `memory.high`, OOM or OOM-kill events, and zero
service swap current. The local GPU census was empty after runner completion.
The remote 7B artifact unit was then stopped by its managed user service;
`systemctl --user is-active` returned `inactive` and no 18081 listener remained.
The approved prepublished gzip cache was reused; 132 replay-linked transfer
records report no request-time packing.

Raw evidence remains outside Git at the unique D278 result root. Immutable
hashes and the compact values are in `d278_status.csv`; no prior result or
user artifact was overwritten. The non-ranking readiness preview table and
figure are under
`figs/ieee_tc/p2_backend_qualification/d278_20261006_readiness_retry9/`;
the figure omits labels for sub-four-percent segments at single-column size
to avoid overlap while retaining their exact values in the CSV/table.

## Next falsifiable action

Do not promote D278 to readiness merely because all requests succeeded. First
audit why the natural controller creates only ten first-service observations
despite ten scale-ups. If the pre-registered protocol permits one identical
cold replicate, run exactly that replicate after the same CPU/resource gates;
otherwise report the sample shortfall and proceed to the next authorized Prime
qualification gate. No baseline or 3B performance run is authorized from this
diagnostic alone.
