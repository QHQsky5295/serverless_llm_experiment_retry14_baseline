# D202 physical HOST representation qualification

CPU qualification and saved-state copy component only; no inference gain or independent-run CI.

| Check | Tests | Result | Body seconds | Peak memory bytes |
|---|---:|---|---:|---:|
| tests1 | 79 | FAILED_fixture_corrected | 0.051 | 645042176 |
| tests2 | 79 | PASS | 0.047 | 645144576 |
| regression1 | 1122 | PASS | 156.731 | 2052943872 |
| probe1 | None | FAILED_fixture_namespace_corrected | None | 670326784 |
| probe2 | None | PASS | None | 790769664 |

| Snapshot | Adapters | Views omitted per HOST table | Full JSON bytes before/after | HOST deepcopy ms before/after |
|---|---:|---:|---:|---:|
| 0 | 8 | 2048 | 1388807/550023 | 26.140382/9.928758 |
| 1 | 12 | 3072 | 2025691/766427 | 75.266975/14.411368 |
| 2 | 20 | 5120 | 3301339/1199835 | 96.688978/60.121510 |
| 3 | 24 | 6144 | 3936172/1415212 | 111.970420/30.927552 |

Copy values are medians of three alternating-order blocks, ten copies per block;
not CUDA inventory, RPC latency, end-to-end speedup or three workload repetitions.
