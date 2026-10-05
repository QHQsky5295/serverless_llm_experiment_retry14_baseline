# D224 diagnostic table (not a Resident result)

Source: baseline raw replay and lifecycle receipt; source hashes are recorded in
`EXPERIMENT_AUDIT.json`.  Values below are descriptive only.  The workload plan
was 4,000 requests; D224 deliberately stopped at a 100-request prefix.

| Field | Observed value | Interpretation |
|---|---:|---|
| Planned requests | 4,000 | Full-workload denominator; not completed |
| Completed / successful | 100 / 100 | Prefix only |
| Native fixed-output matches | 100 / 100 | Generation-token contract only |
| Mean TTFT | 490.448 ms | Descriptive prefix value |
| Type-1 P95 TTFT | 789.498 ms | Descriptive prefix value |
| Type-1 P99 TTFT | 1,965.537 ms | Descriptive prefix value |
| Mean service TTFT | 363.880 ms | Excludes upstream wait by the runner definition |
| Mean E2E | 4,849.841 ms | Descriptive prefix value |
| Type-1 P95 E2E | 7,364.052 ms | Descriptive prefix value |
| Mean TPOT | 25.335 ms | Native vLLM token-ID timing |
| Legacy TTFT-only SLO display | 100% | Not the frozen joint TTFT/TPOT SLO |
| Observed receipt interval | 545.431 s × 4 GPUs | Not an auditable physical lease; excluded |

No figure is generated: a qualification/status table is the appropriate
artifact for an incomplete and protocol-invalid attempt.  No row above may be
used to define `U_ref`, a budget, G1/G2, or a system ranking.
