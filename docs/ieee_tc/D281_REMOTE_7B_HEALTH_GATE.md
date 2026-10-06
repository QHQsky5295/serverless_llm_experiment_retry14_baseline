# D281 — 7B remote delivery health gate

**Date:** 2026-10-06
**Scope:** read-only remote health and manifest check; no inference GPU run.

The owned remote 7B artifact unit was started through the verified SSH alias
`primelora-artifact-174` and checked without changing the immutable published
cache.  The service is the existing deployment from the D78/D80 publication;
no archive was regenerated and no source pool was scanned or copied.

| Check | Observation |
|---|---|
| Endpoint | `http://192.168.4.174:18081` (direct NIC path, proxy bypassed) |
| Unit | `primelora-artifact-tc-7b.service` |
| Unit state | active/running, one task |
| Delivery | `prepublished_gzip_v1` |
| Timing contract | `artifact_timing_v2` |
| Published object count | 500 |
| Manifest response SHA-256 | `e1ab04f023e1fe798ef47bd99271db336093f0a606184bcdda4176a53c10ac29` |
| Remote process clock | `remote-process-monotonic:bc6cbf52518e422495771aff1769c2fa` |
| Service `MemoryHigh` / `MemoryMax` | 2 GiB / 4 GiB |
| Service `TasksMax` | 128 |

The authenticated `/health` response reported `ok=true` and the expected
prepublished delivery mode.  The first unauthenticated local probe and a
short startup-race probe were not treated as failures; they returned 401 or
occurred before the listener was ready.  The authenticated direct-NIC probe
and the manifest count/hash above are the accepted health evidence.

This gate authorizes only a protected Resident-vLLM pilot after the local
GPU/memory/disk/cleanup preflight.  It does not establish request success,
native token correctness, lifecycle GPU seconds, warm SLO, or a Resident
reference budget.  The remote unit remains owned and must be stopped after a
pilot or failed launch using the same identity-checked SSH path.
