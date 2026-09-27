# Remote artifact node: execution record

## Verified 2026-09-25

- User-authorized dedicated Ed25519 key deployed, restricted against SSH
  forwarding/PTY features. No password/private key is stored in this repository.
- Host identity matched the pre-existing trusted key for
  `[192.168.4.174]:8122`, fingerprint
  `SHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40`.
- Management alias: `primelora-artifact-174`; account `lab14`, port `8122`.
- Local private key: `~/.ssh/primelora_artifact_174_ed25519_20260925`, mode 0600.
- `ssh primelora-artifact-174 'id -un'` succeeds with BatchMode/strict host
  checking and no password interaction.
- The management IP `10.199.227.174` did not answer the short SSH probe.
  The already-trusted experiment LAN address `192.168.4.174` works and refers
  to the same approved node. HTTP data remains direct on the experiment LAN.
- User systemd manager reports running; no artifact listener found on 18080/18081.
- Original server exists at the approved path; model pools are reused in place.

## Historical gate (superseded by user authorization on 2026-09-27)

Remote available disk measured 149,137,707,008 bytes (138.9 GiB), below the
approved 150 GiB floor. No disposable artifact temp directories were found.
Existing full-pool archives total under 2 GiB; deleting them cannot resolve
the shortfall. Other projects and unique 13B assets were not touched.

A user question is pending about retaining the common floor versus approving
a separate measured-peak rule for the lightweight artifact node. Meanwhile,
local audits continue. No remote artifact service was started, and full
start/stop/restart/download qualification is NOT yet complete.

## Rechecked 2026-09-27 (D53, read-only)

Strict BatchMode key access still works. Available disk is148735832064 bytes
(138.52GiB). The7B/3B upload archives contain1967102690 bytes in total (1.83GiB);
removing these cannot reach the150GiB floor. No listener is present on18080/18081.
No files were removed and no service was started. The user was asked to retain
the floor and free space, or authorize a separate measured-peak artifact-node
rule. That rule is not yet approved, derived or installed; inference-host
  limits and the existing approved plan remain unchanged.

## D74 clarification: policy floor is not demonstrated storage exhaustion

The user asked why an artifact-only machine could block local inference.
Read-only SSH recheck on2026-09-27 finds148617461760B available (~138.42GiB),
71% filesystem use; no listeners18080/18081. This is NOT a disk-full error or
evidence that existing LoRA weights do not fit. The unresolved gate is the
plan's blanket150GiB minimum, not a measured artifact-server requirement.

Actual remote server.py lines120–155 create a temporary directory, write one
adapter tar.gz completely, stream that file to HTTP, then remove it in finally.
Therefore this implementation DOES perform remote temporary writes; it is not
read-only file streaming. Concurrent downloads can coexist as multiple archives;
CPU compression, temporary space, logs and failure cleanup matter. It does not
need inference-host GPU/model memory. The current138GiB free does not establish
shortage, and150GiB must not be described as an empirically necessary minimum.

A separate artifact-node rule based on bounded concurrent archive peak, logs
and a safety reserve is recommended, leaving inference-host rules unchanged.
Explicit user choice requested during D74; not approved or installed at the
time of this record. No remote service start, heavy profiling, deletion, upload
or configuration change was performed for this read-only clarification.

## D74 approved revision and interference boundaries

User subsequently APPROVED replacing only the artifact-node150GiB floor, and
directed a return to Prime Full after the current Serverless run. See the source
plan's2026-09-27 revision/section2.9 and approved snapshot20260927. The historical
paragraphs above describe the earlier gate, not a still-pending user choice.

For each real temporary/log filesystem, require free bytes >= positive frozen
safety reserve + ceil(1.5*(sum(all-service concurrent packs * remaining archive
growth bound) + remaining log growth)). The existing preflight module exposes
artifact_disk_required for this arithmetic, with invalid/unknown bounds rejected.
The inference-host disk_required,150/100GiB floors and all memory guards are
UNCHANGED. Packing concurrency is observed/bounded from the common protocol,
not secretly reduced to make a particular system pass. Quota/inodes and the
evidence behind bounds require separate readback before remote qualification.
No measured remote profile is yet claimed and neither service is yet qualified.

The latest user explicitly retains legitimate Prime advantages: NVMe/HOST/GPU
residency may reduce remote fetch count/bytes and contention. Do NOT equalize
misses, discard caching or subtract those effects. Unrelated remote jobs,
service restart/configuration, bulk hashing, packing stress and cleanup are
excluded from measured runs. Freeze the shared artifact configuration first.

Dynamic per-request tar packaging is a delivery-implementation confound, not
GPU inference time or pure network time. Record correlated remote pack/wait/send
and local receive/extract spans, bytes and call counts; preserve complete E2E.
Removing unnecessary request-path packaging requires a common validated delivery
contract, preferably reusing existing immutable artifacts; it is not permission
to regenerate weights/traces, copy the full pool or silently change one baseline.
Do not subtract aggregate pack CPU/wall time from overlapping request E2E.
Uncalibrated cross-host timestamps are not directly subtractable. W3C
[Server Timing](https://www.w3.org/TR/server-timing/) provides a duration-reporting
interface, not clock synchronization or a causal latency-subtraction rule.

No remote start, restart, deletion, stress download or configuration change was
performed during D74's local inference measurement. All new remote qualification
work occurs between runs and is a prerequisite for Prime's real-remote Full.

## D75 preparation and timing contract (not yet service qualification)

Reuse the existing server and HTTP client. The `artifact_timing_v1` observations
correlate HTTP attempt UUIDs with remote packing/send/cleanup spans. Keep the
physical file-reservation ID separately. Remote durations are not clock-synced
timestamps; packaging is nested in the client's request-to-header wait, not an
extra additive E2E stage. Client reserve/receive/write includes reservation work
and local I/O; it is not pure network time. Missing legacy timing stays unknown.

Two complete remote pools have500 directories each and no symlinks/special
members. A metadata-only PAX-header calculation and conservative zlib DEFLATE
bound give maximum allocated archives63725568B(3B),42684416B(7B), including gzip
wrapper/name and4KiB block rounding. No weight bytes were read for this bound.
Use the maximum of fixed/stored bounds in the installed zlib1.3
[source](https://github.com/madler/zlib/blob/v1.3/deflate.c), not average observed
zero-weight compression. Filesystem is ext4, no quota mount option,30896617 free
inodes and148635492352B available at this check.

For the bounded SERIAL functional qualification, allow at most502 attempts per
model (500 coverage, one sample, one cancellation); even if every temporary
archive remained until the end, their upper bound is53417811968B. Add64MiB log
growth allowance,16GiB safety reserve for OS/unrelated-user recovery and1.5 growth
margin: required97407250432B. This passes available space without deletion,
inference-node rule changes or backend concurrency throttling. This count bound
is only for qualification, NOT the eventual Full all-worker concurrency rule.

The existing client `verify-pool` performs authenticated manifest equality and
SHA-verified downloads serially into one owned temporary directory at a time.
It retains an exclusive JSONL journal, not another full dataset. No successful
sample alone establishes whole-pool, model or numerical-LoRA qualification.
Baseline work remains paused. No remote configuration work overlaps inference.

### D75 actual two-model samples and management (2026-09-27)

Source d8a0b49443e413947130a14a62c7aa5562bbee17 is pushed/remote-SHA verified.
Versioned remote script lives at
`/home/lab14/primelora_remote/tc/d8a0b49443e413947130a14a62c7aa5562bbee17/server.py`;
SHA942a8e52956a58c22e822d68011fbd47ffa3b64bd4306904168a46bad7059101.
Original server SHAa365072244512f4880432d7f4198cf3e45897ff19063a2b25eb141c8ca4e2a02
is unchanged. Local and remote token files are owner0600, outside repositories;
only their paths are recorded, never token material.

Local private token: `/home/qhq/.config/primelora-tc-d75/artifact.token`.
Remote private token: `/home/lab14/.config/primelora-tc-d75/artifact.token`.
Managed units: `primelora-artifact-tc-3b.service` and
`primelora-artifact-tc-7b.service`. Strict SSH sessions can execute user systemctl
status/start/stop/restart for these two owned units; no password interaction.
Both actually stopped (PID0, inactive, ports gone), started and restarted.
Post-restart invocation IDs3B5d1621ec44f44d5495e0fd2c86a182ba and
7B856cbbb9dfd94974ae19fffc2c8549c3. Resource readbacks and later health checks
must verify the current invocation, not reuse these IDs after another restart.
Each service1/2GiB high/max,swap0, affinity2–19/22–39 (18 physical cores),
PrivateTmp, no automatic restart. These are qualification settings, not yet the
common frozen Full performance delivery contract. No limit/event bypass.

| Functional sample, one each | 3B code_lora | 7B code_lora |
|---|---:|---:|
| Frozen content SHA verified | yes | yes |
| Logical payload bytes |56327468|20938675|
| Transferred archive bytes |2339543|780772|
| Client total through cleanup, ms |3004.945|1152.759|
| Request-to-headers, ms |2412.344|927.992|
| Remote pack, ms (nested in previous row) |2406.873|922.631|
| Client reserve/receive/write, ms |199.435|67.496|
| Client extract/verify, ms |391.039|155.381|
| Remote temporary removed |yes|yes|

This is NOT a representative performance/profile estimate or system comparison;
it verifies content and time-stage correlation. Never add the pack row to the
client rows. Source logs: D75sample3b.log/sample7b.log and matching server JSONL.
The uninstrumented original was not modified. Authentication without a token
returns401 on both ports. An initial curl invocation followed inherited proxy
configuration and timed out; explicit direct-LAN checks pass. The actual client
already disables environment proxies; do not call the proxy timeout a server
failure or silently use a proxy for experiments.

IMPORTANT actual link observation: route to inference host uses remote eno1;
sysfs reports100Mbps/full/MTU1500, while inference eno1np0 reports1000Mbps/full.
Do NOT label this measured route1Gbps or claim a configured1Gbps cap achieved it.
No link renegotiation, cable/NIC reset or remote tuning was attempted. Separate
hardware capability from current negotiated link speed and achieved throughput.
The full500/model content coverage and cancellation qualification remain open.

Sample extracted copies were removed after successful SHA verification, client
exit and ownership/reference check; original remote/local pools and raw timing
records remain. Complete-pool coverage is now in progress. Its1GiB per-service
memory.high triggers file-cache reclaim; max/OOM remain0 at the checkpoint.
This is an explicitly lightweight functional envelope, not a qualified Full
performance environment, and these durations will not be promoted into frozen
service/preparation profiles. Keep the active run unchanged; choose the common
remote resource contract from complete footprint/concurrency observations before
later performance profiling. No claim that a high event invalidates content.
### D76 complete 3B content coverage (2026-09-27)

The SAME D75 run completed; it was not restarted or replayed. Actual start
11:54:28+08, serial verification1410.208s. This is a functionality result only.

| Qualification evidence | 3B observed |
|---|---:|
| Complete frozen manifest / verified adapter IDs |500 /500|
| Content-SHA-verified regular files |4000|
| Logical payload bytes |19482573296|
| Actual HTTP archive bytes |1160493734|
| Distinct client/server UUID matches |500|
| Local / remote temporary cleanup confirmed |500 /500|
| Failed transfers |0|
| Local watchdog samples |1392|
| Local service peak, bytes |95641600|
| Local high / max / OOM / OOM-kill |0 /0 /0 /0|
| Remote high / max / OOM / OOM-kill |34982 /0 /0 /0|

All500 server records were copied AFTER completion, SHA-matched against the
remote journal, then reconciled individually with adapter identity, content
contract, payload and wire bytes, pack-duration rounding, ordered local/remote
spans and cleanup. Remote clocks were never subtracted from local timestamps.
The local service path is removed, its auxiliary scope is inactive, no owned
download directory remains, and no GPU model ran. Original pools are unchanged.

Curated source: `paper_results/ieee_tc/remote_qualification/20260927_d76_3b_coverage.json`.
It retains nine raw SHA references and the unchanged frozen input-index SHA.
Remote memory.high reclaim is retained as an observed condition; durations are
NOT production initialization profiles or an inference-performance conclusion.
7B complete coverage and real cancellation cleanup are still pending at this
checkpoint. No new baseline, weights, full trace or retained pool copy.
