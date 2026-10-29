# D945: initial-NVMe preload capacity rejection

The first PrimeLoRA initial-NVMe-preload candidate used a fresh execution key:

`d943_20261030/7b_initial_nvme_preload_w2`

It was rejected before the request replay began.  The fixed 500-adapter
manifest is approximately 12.09 GiB after materialization, while the native
file-owner guard must retain the published copy and the in-flight staging copy.
With a configured 16 GiB NVMe tier, the guard correctly raised:

`local file capacity conflict: retained copies plus transfer exceed tier budget`

No request result, GPU lifecycle metric, or comparison table is derived from
D943.  Its launcher, replay, watchdog, and service logs remain as failure
evidence.  The follow-up candidate uses a new execution key and a 24 GiB NVMe
capacity so that the existing 1.5x staging/retained-copy invariant is satisfied;
the GPU count, model, request trace, adapter map, generation contract, and
control policy remain unchanged.
