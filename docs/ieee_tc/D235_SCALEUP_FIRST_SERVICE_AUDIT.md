# D235 — 7B scale-out first-service and confirmed-tier audit

**Date:** 2026-10-05  
**Plan SHA-256:** `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`  
**Branch:** `retry14_continuous_queue_v2`  
**Scope:** offline audit only; no GPU, remote service, result, or configuration was changed.

## Finding

The zero values in `scaleup_first_service_request_count` and
`scaleup_first_service_planned_match_rate` do **not** show that no requests
were served after scale-out. They are legacy handoff labels and are undefined
for the current `ieee_confirmed` path. The current path deliberately rejects a
legacy warm-up plan and removes legacy handoff fields from the published scale
event. This is a measurement-schema limitation, not evidence of a successful
or failed GPU handoff.

There is nevertheless a real performance issue that the labels hide. The
confirmed dispatch records show that newly activated replicas were selected
with a confirmed non-GPU source tier for their first requests. Thus the
scale-out tail is not merely a missing label: the selected replica often had to
load the adapter from NVMe/HOST/Remote after dispatch.

## Code evidence

1. In `scripts/run_all_experiments.py:13866–13870`, an
   `ieee_confirmed` activation rejects any legacy `handoff_plan` and enters
   `_add_ieee_dedicated_instance` without it.
2. In `scripts/run_all_experiments.py:14522–14526`, the event registrar clears
   `handoff_plan` and `preload_budget` for `ieee_confirmed`, explicitly avoiding
   stale legacy prefixes.
3. In `scripts/run_all_experiments.py:14597–14640`, the first-service label is
   derived only from the per-runtime legacy plan's
   `first_service_request_count`; with no such plan its request budget is zero.
4. In `faaslora/experiment/instance_pool.py:1781–1794`, confirmed routing
   selects the minimum measured service-class key among feasible replicas. A
   confirmed NVMe/HOST/Remote source is a valid feasible class; GPU readiness
   is not imposed as a precondition.
5. In `scripts/run_all_experiments.py:16893–16962`, the request first commits a
   fresh received-owner snapshot, records the selected source tier, registers a
   pending load when that tier is not GPU, and only then proceeds to source
   protection/admission. This is the authoritative pre-dispatch evidence path.
6. In `faaslora/preloading/preloading_planner.py:645–774`, activation planning
   freezes the demand window available when the plan is created. A 45-second
   startup can therefore make that demand window stale unless a new epoch is
   planned after startup or the ready-time demand frontier is incorporated.

## D219 evidence

Source result:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d219_20261005/7b_confirmed_snapshot_cache_full_replay/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d219_7b_confirmed_snapshot_cache_full_replay.json`

Result SHA-256: `e2a83118a59019854e1110f40c3da333e7a5be79062552bed6ab2b87e0bbdfff`.

- Three physical scale-up events were triggered at visible request count 4;
  recorded activation times were 44.91 s, 46.57 s, and 45.69 s.
- The first selected requests on the three new replicas were `req_00022`,
  `req_00028`, and `req_00027`; their confirmed pre-dispatch tiers were NVMe,
  Remote, and NVMe respectively.
- All 3,003 requests served by the three scale-up replicas have
  `confirmed_dispatch_snapshot=true`.
- For the first 20 requests observed on each new replica (60 requests total),
  the confirmed tiers were GPU 10, HOST 23, NVMe 12, and Remote 15. Only
  10/60 (16.7%) were GPU-ready before dispatch.
- The first 20 per-replica requests have mean TTFT 1,889.6 ms; the three
  earliest requests include TTFTs of 11,288.8 ms, 1,591.8 ms, and 623.9 ms.
- Across all requests on these replicas, the tier counts were GPU 807, HOST
  1,340, NVMe 754, and Remote 102. This is descriptive evidence only; the
  result is not G1/G2-qualified because it lacks physical-capacity admission
  qualification and a common warm reference.

The D219 request record for `req_00022` is especially diagnostic: its
`routing_snapshot` selected the new replica with a confirmed NVMe service
class, while the incumbent had no available slot. Its
`source_admission.physical_capacity_qualified` field is false, and the
request performed demand loading from NVMe. The snapshot is genuine, but it
does not claim GPU readiness.

## Causal interpretation

The evidence supports two separate statements:

* **Metric statement:** zero `scaleup_first_service_*` is a schema gap for the
  native confirmed path and must not be used as a readiness result.
* **System statement:** the current native activation can publish a new
  runtime before a future demand adapter is GPU-resident. When the incumbent
  has no admission lane, Eq. (3)'s measured service-class selection can choose
  the new runtime's NVMe/HOST/Remote path. This is a genuine source of the
  observed scale-out tail.

No code change is justified solely by the legacy zero counters. A valid next
  optimization must preserve confirmed-owner semantics and either:

1. create a native, post-startup preparation epoch from the currently visible
   ready-time demand frontier before publishing GPU-ready claims; or
2. change admission/routing only through a measured, frozen service-class
   objective that accounts for the activation-time source state, then verify
   the change with the same fixed-output trace and physical-owner evidence.

Reusing the legacy handoff plan, treating a prediction as confirmed GPU state,
or labeling a non-GPU source as GPU-ready would violate the frozen protocol and
is explicitly disallowed.

## Publication status

This audit is diagnostic and does not authorize a formal claim, a baseline
comparison, or a new full replay. The next experiment remains blocked on the
existing host-memory gate and on the numeric PEFT reference/admission gates.
