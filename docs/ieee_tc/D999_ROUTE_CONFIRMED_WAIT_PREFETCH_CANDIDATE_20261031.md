# D999 3B route-confirmed wait preparation candidate

## Status

This is a single falsifiable development candidate for the 3B PrimeLoRA
main-line optimization. It is not a final result, not a baseline comparison,
and not a new workload. The run key is
`d999_20261031 / d999_3b_route_confirmed_wait_prefetch_w1`.

## Evidence motivating the candidate

- D992 remains the retained 3B reference, but its frozen common TTFT/TPOT
  audit is 3678/4000 (91.95%), below the 95% feasibility threshold.
- D995 prepared an arrived remote demand before routing. It reduced the
  observed remote dispatch tier to zero but increased scale-out activity and
  worsened the frozen joint rate to 3529/4000 (88.225%). It is therefore not
  a template for this candidate.
- D998 deferred read-only scheduler observations. Its receipts prove the
  policy activated, but the frozen joint rate fell to 3612/4000 (90.30%), with
  higher dispatch/runtime waits and physical GPU-s. That policy is rejected.
- In D992, remote preparation was normally started after the per-replica
  runtime lane had already been reserved. For the 504 requests with remote
  preparation evidence, preparation elapsed time averaged about 0.89 s while
  runtime-slot wait had a median of about 71 ms and a P95 of about 499 ms.
  This exposes a bounded opportunity to overlap part of the wait without
  moving work before routing or outside the request timeline.

## Precise hypothesis

Once an arrived request has a confirmed routing snapshot and the selected
replica is temporarily unable to reserve a runtime lane, an owned NVMe
preparation for that request's selected adapter can overlap the existing
runtime-lane wait. If the lane becomes available, the request still performs
the existing source-owner revalidation, native admission, and source lease
transaction. Therefore the overlap can reduce wait on remote requests without
granting an unconfirmed source or changing the dispatch contract.

The candidate starts only when all of these are true:

1. the request is already inside the ordinary dispatch admission path;
2. routing policy is `ieee_confirmed` and the selected source tier is
   `remote`;
3. the selected replica fails the existing atomic runtime-lane reservation;
4. the existing owned movement queue and real remote artifact client are
   available.

It does not run on a pre-routing arrival, use future requests, select a
   different replica, reserve a GPU/KV slot, or remove any request-time
   transfer/materialization from TTFT/E2E.

## Implementation boundary

The opt-in control is
`resource_coordination.ieee_route_confirmed_wait_file_preparation:
route_confirmed_wait_v1`. The task carries explicit evidence and is joined on
normal completion, route conflict, request cancellation, and outer cleanup.
An exception in the optional overlap is recorded and the ordinary admitted
preparation path remains authoritative. A successful overlap only publishes a
verified file; it cannot bypass `_ieee_protect_selected_source` or native
admission.

The only source changes are the reservation-owned task, its cancellation/join
path, and the call immediately after a failed atomic lane reservation. The
core routing, source identity, admission, generation, and release formulas are
unchanged.

## Acceptance and rejection

After the full 4000-request run, re-evaluate the frozen metric protocol with
the same workload, generation contract, and resource envelope as D992:

- 4000/4000 correct completion and native timing integrity;
- common joint TTFT/TPOT SLO rate, with 95% as the feasibility threshold;
- physical lifecycle GPU-s/request and tail metrics;
- counts of route-confirmed overlap starts, owned transfers, coalesced
  subscribers, source revalidations, and any failed/cancelled overlap;
- scale-up count, remote-transfer count/bytes, and runtime-slot wait.

The candidate is retained only if it is a real improvement over D992 under the
frozen metric protocol and does not introduce correctness, ownership, or
resource-accounting violations. A lower joint rate, increased lifecycle
GPU-s/request without a compensating SLO/tail result, or any incomplete
settlement rejects the candidate. No seed is selected after the fact and the
configuration is not altered inside this execution key.

## External design context

ServerlessLLM treats model loading as part of the request-serving path, while
vLLM exposes adapter loading separately from generation. D999 uses that same
separation only as a systems boundary: preparation remains request-visible and
owned, while generation authority remains at the confirmed source/admission
linearization point. These papers motivate the boundary but do not provide
performance evidence for D999.
