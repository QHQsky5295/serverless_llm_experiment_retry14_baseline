# D233 — current remote boundary and artifact-disk recheck

Date: 2026-10-05 20:09+08 (Asia/Shanghai)

This is a read-only state and provenance check. It does not modify the remote
node, start either artifact service, generate traffic, repack files, or alter
the frozen metric protocol.

## Link state

| Endpoint | Interface | Observed negotiation | State |
|---|---|---:|---|
| Remote artifact node `192.168.4.174` | `eno1` | 1000 Mb/s, full duplex | up |
| Inference host | `eno1np0` | 1000 Mb/s, full duplex | up |

The current readback used the already authorized strict SSH alias
`primelora-artifact-174`; the remote boot ID was
`b53c1f16-1dd2-4741-9271-6b75e86582ec`. Both remote services were inactive and
no artifact listener was open. The readback is not a claim about achieved
application throughput: payload size, TCP behavior, disk I/O and concurrency
still determine transfer time.

The D77 record of remote 100 Mb/s/full is retained as historical evidence. At
that time the peer advertised only 10/100 modes. The exact physical cause was
not established, and no agent action forced renegotiation. D78's later
post-reboot readback and this check show 1000/full with gigabit peer
advertisement. Therefore:

1. historical 100-Mbps results cannot be silently relabelled as 1-Gbps results;
2. the current run contract is 1-Gbps/full, without claiming 10/100-Gbps or
   H100-scale hardware;
3. application-layer bandwidth sensitivity and `no-delay` are controls, not
   physical-link measurements.

## Immutable delivery cache

Remote manifests were read without changing them:

| Model | Manifest | Manifest SHA-256 | Complete | Artifacts | Archive bytes |
|---|---|---|---:|---:|---:|
| 3B | `tc/d78_20260927/published/3b/delivery_manifest.json` | `9fbd57b6216541d90f01e3efb7e63e849fec011c51a278259e9a54d8546d8173` | yes | 500 | 1,160,474,851 |
| 7B | `tc/d78_20260927/published/7b/delivery_manifest.json` | `9af659e8178ba0994452a048b92e414c93988b51698b8b247e05d04eee2cfff8` | yes | 500 | 420,740,410 |

Both manifests declare `published_gzip_delivery_v1`, gzip level 9 and
`preparation_outside_inference=true`. The archive total is 1,581,215,261 bytes
(about 1.47 GiB). No extracted second pool was created by this check.

## Independent remote disk gate

The remote filesystem readback reported 143,522,885,632 free bytes (133.67
GiB). The approved artifact-only rule is independent of the inference host:

\[
P_{pack}=\sum_i q_i g_i,\qquad
F_{required}=S+\lceil1.5(P_{pack}+G)\rceil.
\]

The sealed conservative qualification bounds use the existing whole-pool
coverage contract: maximum 502 attempts per model, 64 MiB log growth, and 16
GiB positive safety reserve. They require 97,407,250,432 bytes (90.72 GiB), so
the current remote filesystem passes this independent artifact gate. This does
not change the inference-host requirements of 150 GiB start, 100 GiB stop,
physical-memory gate, swap gate or OOM containment.

## Measurement boundary for the paper

The following are part of the request-triggered service path and remain in
PrimeLoRA and every baseline's user TTFT/E2E:

- remote object read and response, actual wire bytes, contention and failures;
- local receive/write/materialize/decompression needed by the request;
- cache miss, tier migration, eviction and re-fetch caused by the system;
- the resulting GPU lifecycle interval and SLO outcome.

The following are recorded separately but do not enter request E2E and are never
subtracted from it:

- one-time creation of the immutable gzip caches before deployment notice;
- remote administration, restart, full-pool scans, logging and cleanup;
- unrelated remote work and deliberate packing/compression stress tests.

This preserves PrimeLoRA's legitimate advantage: if residency avoids a request
miss, it legitimately reduces that request's remote transfer and contention.
Fetch counts and bytes are not equalized across systems. Conversely, no system
may claim a benefit by omitting a request-triggered transfer or by subtracting
remote packing time from an overlapping end-to-end interval.

The single calculation entry is
`docs/ieee_tc/METRIC_PROTOCOL_FINAL_V2.md`; this document only records the
current external-state evidence supporting its boundary.
