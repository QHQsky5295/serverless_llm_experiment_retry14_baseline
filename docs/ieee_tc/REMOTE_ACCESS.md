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
