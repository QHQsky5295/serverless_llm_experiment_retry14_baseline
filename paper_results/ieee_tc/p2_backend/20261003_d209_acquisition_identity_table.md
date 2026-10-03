# D209 acquisition identity qualification

CPU qualification and exact-parent component checks only; no GPU inference gain or CI.

| Check | Tests | Result | Body seconds | Peak memory bytes |
|---|---:|---|---:|---:|
| tests1 | 222 | FAILED_old_fixture_field | 2.997 | 671211520 |
| tests2 | 222 | FAILED_old_fixture_field | 3.717 | 670810112 |
| tests3 | 222 | PASS | 2.862 | 670654464 |
| regression1 | 1144 | PASS | 157.105 | 2050699264 |
| probe1 | None | PASS | None | 725016576 |

| Source | Full graphs parent/candidate | Target graphs parent/candidate | Identity reads parent/candidate | Fresh reads parent/candidate |
|---|---:|---:|---:|---:|
| gpu | 0/0 | 1/1 | 0/0 | 1/1 |
| host | 0/0 | 1/1 | 0/0 | 1/1 |
| host_file | 1/0 | 1/1 | 1/2 | 3/3 |
| nvme | 1/0 | 1/1 | 1/2 | 3/3 |
| remote | 1/0 | 1/1 | 1/2 | 3/3 |

| D203 selected source | Native | Requests |
|---|---|---:|
| gpu | True | 1189 |
| host | False | 36 |
| host | True | 1766 |
| nvme | False | 873 |
| remote | False | 136 |

Recorded selected-source populations, not measured acquisition RPC counts or latency causality.
