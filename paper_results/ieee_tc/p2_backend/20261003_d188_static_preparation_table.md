# D188 component table

Three alternating component calls, not independent serving runs. No CI or TTFT extrapolation.

| Case | Current ms | Static prototype ms | Reduction | Compile once ms | Inventory calls | Digest calls |
|---|---:|---:|---:|---:|---:|---:|
| existing_fixture_4 | 3.338 | 3.127 | 6.298% | 0.279 | 1 -> 1 | 4 -> 0 |
| existing_fixture_500 | 36.685 | 13.320 | 63.690% | 23.609 | 1 -> 1 | 500 -> 0 |
| existing_7b_index_empty_owner | 63.946 | 11.300 | 82.329% | 53.558 | 1 -> 1 | 500 -> 0 |
