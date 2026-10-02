# D180 dependency-study table

CPU/saved-observation study; no serving latency measurement, no CI, no model acceptance.

| Snapshot | Adapters | Full tensor checks | Target checks | Reduction | GPU/HOST targets |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 2048 | 256 | 87.500% | 8/0 |
| 1 | 12 | 3072 | 256 | 91.667% | 8/4 |
| 2 | 20 | 5120 | 256 | 95.000% | 8/12 |
| 3 | 24 | 6144 | 256 | 95.833% | 8/16 |

64 target footprints and 192 service classes match; all identity fields preserved.
Shared-storage counterexample: reachable=512 bytes, globally reclaimable=0 bytes.
Unrelated malformed tensors are not checked by target observation; global safety equivalence is NOT claimed.
A scoped production schema, freshness/sharing tests, and complete 7B replay remain pending.
