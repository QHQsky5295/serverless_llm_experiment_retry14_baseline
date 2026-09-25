# Safety and preservation evidence — 2026-09-25

This is a prerequisite status table, not a performance experiment or a claim
that the production campaign is ready.

| Check | Observation | Interpretation |
|---|---|---|
| Plan identity | Source and committed snapshot SHA agree | Approved scope preserved |
| Historical files | Protected SHA seal and verification agree | Old results and selected dirty inputs unchanged |
| Physical admission | Initial available memory passed 102 GiB and disk passed 150 GiB floor | A point-in-time check, not a reservation |
| First tiny OOM witness | 1,737 high events, zero max/oom events; allocation timed out | Test did not isolate hard limit; failure retained |
| Revised tiny hard-limit witness | 128 MiB high=max scope; local OOM kill observed | Hard-limit mechanism works at tiny scale |
| Inheritance and cleanup | Child affinity/cgroup inherited; owned test processes cleaned | Basic process primitives pass |
| CPU controller | No delegated cpuset; task affinity is inherited | Do not claim cpuset enforcement |
| Production readiness | NOT authorized | Actual Ray/container workers, external watchdog, GPU release, quota checks remain |

The revised test changes only the tiny test envelope, not the 72/80 GiB
production limits. Linux distinguishes [high-limit throttling from max-limit
OOM](https://docs.kernel.org/admin-guide/cgroup-v2.html). No performance conclusion
is based on either tiny witness. No GPU model was loaded for these tests.

Machine-readable records retain limits, raw events, ownership and hashes.
`20260925_small_scope_attempt1_observed.json` is an explicitly labelled record
of the failed attempt's observed output, not a substitute raw successful run.
