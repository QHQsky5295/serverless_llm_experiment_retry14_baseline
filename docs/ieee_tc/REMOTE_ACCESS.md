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

### D77 complete 7B content coverage (2026-09-27)

The prepared D75 run completed without restart: 12:21+08 start, serial loop
645.243s. Both model pools now pass full content coverage, not inference or
performance qualification. No serving configuration changed during either run.

| Qualification evidence | 7B observed |
|---|---:|
| Frozen manifest / verified IDs |500 /500|
| SHA-verified regular files |5000|
| Logical payload / HTTP archive bytes |12985984450 /420756591|
| Distinct client/server UUID matches |500|
| Local / remote temporary cleanup |500 /500|
| Failed transfers |0|
| Local watchdog samples / peak bytes |638 /74416128|
| Local high / max / OOM / OOM-kill |0 /0 /0 /0|
| Remote high / max / OOM / OOM-kill |22352 /0 /0 /0|

All 500 HTTP attempts reconcile individually by identity, bytes, content
manifest, ordered spans and cleanup. Remote journal copy SHA matches the source.
Service path removed, auxiliary inactive, no owned temporary pool or GPU model.
Curated evidence: `paper_results/ieee_tc/remote_qualification/20260927_d77_7b_coverage.json`.
Actual remote1/2GiB qualification limits cause reclaim; these timings must NOT
initialize Full profiles. Cancellation cleanup and the shared performance
resource contract remain separate gates before representative Full profiling.

### D77 post-header cancellation qualification

One existing `code_lora` request per model, no new data, no injected sleep or
socket pacing. Cancellation is triggered by actual received headers, before
the first application body read. Downloader cleanup is checked BEFORE the
outer temporary directory is removed. This is not a performance measurement.

| Model | Client published / body-read bytes | Downloader leftovers | Matched remote result | Remote temp removed | Local service/watchdog |
|---|---|---:|---|---|---|
|3B|no /0|0|ConnectionResetError, UUID92125202dde74e9d814889e37b117ef4|yes|0/0, released|
|7B|no /0|0|ConnectionResetError, UUID9d87fa4e53c64d9195247d30ae752efd|yes|0/0, released|

3B server bytes_written=0 counts completed socket write calls, NOT a guarantee
that zero bytes reached the network: a failing send can be partial. Packing
already completed before headers. Cancellation therefore does not erase that
work or imply remote packaging was cancelled. Local auxiliary is inactive.
Client/raw/remote receipts remain under D75 raw root with new cancel_* names;
the original500-row complete-coverage snapshots remain unchanged.

Both post-header checks pass. 3B/7B local watchdog3/2 samples, peak39182336/
38817792B, all high/max/OOM/OOM-kill/swap0; both service paths gone and auxiliary
scopes inactive. Curated `20260927_d77_cancellation.json` retains16 raw SHAs.
Owned vmstat monitor invocation181b9be672ca41b08512532c8046cfd0 was verified and
stopped after both checks; remote MainPID0/inactive, local capture exit0.
Final remote_vmstat.log SHA33b8b9601125dfca7c99560a31908af4b9b7ba114241617aa4abf29acc1c40de.
The artifact services themselves remain unchanged. No inference ran concurrently.

### D77 link-rate diagnosis and claim boundary (user question)

Read-only diagnosis, no speed/duplex/MTU/qdisc/interface or switch change:

| Observation | Actual evidence |
|---|---|
| Remote NIC | Broadcom BCM5720 Gigabit Ethernet, PCI14e4:165f, tg3 |
| Remote supported AND advertised modes | includes1000baseT/Full |
| Link partner advertised modes |10/100 only; no1000 |
| Negotiated mode |100Mb/s, full duplex, auto-negotiation on |
| Local interface |eno1np0, sysfs1000Mb/s/full |
| Data route |192.168.4.174↔192.168.4.178, directly through remoteeno1/localeno1np0 |
| Remote qdisc/filters inspected |mq/fq_codel; no rate shaper or ingress/egress filter shown |

Raw `d77_remote_link_diagnosis.json` SHA
30b1acd7f03257c5da9b47dfb237a627cf11da36e12b1fe901a3a432766dcb15.
One optional ethtool netlink query reports insufficient privileges, but the
driver's supported/advertised/peer modes and speed are returned. Local ethtool
is absent; local speed comes from sysfs. An unsupported remote diagnostic flag
was rejected without state change. No privileged write or installation occurred.

Inference: the GPU and NIC maximum capability are NOT the demonstrated limit.
The immediate peer is not advertising gigabit; a100M-only/configured switch
port, intervening equipment or physical-link/downshift issue requires peer-side
inspection/known-good port and cable to distinguish. This does NOT prove the
entire machine-room network is100M, nor rule out unobserved switch/path QoS.
Do not force link speed or renegotiate the active SSH/data interface.

The qualified500-transfer serial samples provide only a diagnostic hint:

| Model | Mean client total ms | Remote pack ms (nested) | Receive/reserve/write ms | Mean archive bytes |
|---|---:|---:|---:|---:|
|3B|2810.138|2331.979|199.532|2320987.468|
|7B|1255.401|959.539|74.296|841513.182|

Remote1/2GiB qualification limits reclaim cache; compression reflects existing
mostly-zero artifacts. These are NOT representative inference profiles. Receive
span is not pure wire time; packing is nested in header wait, not an additive
E2E term. Link negotiation alone does not establish the application's bottleneck.

For the paper: actual request-induced transfer remains in TTFT/E2E and resource
accounting. Deleting it also deletes legitimate caching benefits; summing remote
pack spans and subtracting them from E2E is invalid under overlap/queue feedback.
Report service-only timing separately under that name, not as corrected E2E.
Unrelated interference is handled by predeclared exclusion/rerun rules, never
post-hoc arithmetic subtraction selected by winner. Delivery-format overhead is
measured separately and the common delivery contract must be frozen beforehand.

100M-only data constrain external validity and may magnify remote-avoidance
benefits. Retain actual speed in methods; do not relabel0.25/0.5/1G caps as achieved
on this path. Prefer verifying/restoring the peer's intended gigabit capability
between runs, with physical administrator involvement if necessary. Without it,
higher-bandwidth local-sim is explicitly controlled/simulated supplementary
evidence, not measured faster Ethernet. Existing S1/S2/S3 and LastKnown controls
must establish HOST/GPU location/state/admission benefits beyond remote miss
avoidance; no new broad matrix or silently changed protocol is authorized here.

Primary references checked: Linux ethtool link-mode semantics
(https://www.kernel.org/doc/html/v6.12/networking/ethtool-netlink.html), Intel's
gigabit troubleshooting guidance
(https://www.intel.com/content/www/us/en/support/articles/000035045/ethernet-products.html).
ServerlessLLM §7.1 uses1Gbps to MinIO in one testbed and10Gbps cluster links in
another (https://www.usenix.org/system/files/osdi24-fu.pdf); HydraServe §2/§7
treats fetching/contention as part of cold start and uses16/64Gbps GPU-server
links with sufficient remote-storage capacity
(https://www.usenix.org/system/files/nsdi26-lou.pdf). These support evaluating
storage/network costs, NOT claiming our100M path matches their environments.

### Latest D77 user clarification: already-published artifacts, no dynamic pack

The formal scenario assumes ready-to-transfer remote artifacts. Our on-demand
tar/gzip is extra delivery preparation, not a Prime research contribution. It
will be removed from the ACTUAL measured path, not subtracted afterward. This
supersedes earlier language treating pack avoidance as part of the formal gain.
Necessary transfer/read/response/contention and inference-local tier work remain
observed service costs. Full E2E is measured on this revised common contract.
No dynamic-delivery functional timings become initialization profiles.

Two implementation choices, neither deployed yet:

- Once-only immutable compressed transport cache, from existing exact files,
  generated before any deployment notice and used by ALL systems. Measured total
  archive bytes across two500 pools =1581250325B (about1.47GiB). This is a small
  derived delivery representation, not new weights or another extracted pool,
  but requires explicit permission under the no-pool-copy rule. Keeps current
  compression semantics and avoids per-request compression/formatting.
- Direct frozen file GET, reusing existing files without new retained objects.
  Hugging Face's official file/snapshot download model supports this design
  (https://huggingface.co/docs/huggingface_hub/guides/download), and the pinned
  vLLM0.30 resolver was inspected. HOWEVER it changes wire volume substantially:
  existing uncompressed logical total32468557746B versus historical compressed
  archives1581250325B. Do not switch silently then attribute the larger network
  exposure to Prime's mechanism. All systems/profile identities must match.

An asynchronous choice was sent to the user; no derived cache or direct-file
implementation has been created. Prefer once-only compressed cache if permitted;
otherwise qualify direct existing files with explicit representation/byte audit.
Do not download/generate new weights, inspect future trace to choose objects,
prewarm inference-local caches for free, or alter native caching asymmetrically.
Source plan updated explicitly and snapshotted as
PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md, SHA
7d91a34791f5132b50e983c7552105a297f1ded2dcd1cbdd4108428e2b2dfb05.
Earlier plan snapshots and all D75–D77 raw identities remain intact.

### D78 explicit approval and implementation

User APPROVED the once-only compressed cache. The direct-file alternative is
not selected. Current plan/snapshot SHA
fa1027d968d906982ca183127f2e615b01c554613efe54c962db7e672c1d1577.
Existing server gains offline publication and immutable read-only serving;
client rejects delivery-mode mismatch and verifies compressed AND original
content SHA. No request-time preparation fallback. gzip level9 is the same
Python tarfile default used before; compressed object identity is newly recorded,
not claimed byte-identical to previous timestamped gzip responses. Logical
frozen files and weights remain exact.

Publication must precede deployment notice and all inference measurements;
all500 static IDs/model are prepared independently of demand. Only complete,
startup-hash-verified caches may serve; failed exclusive directories retained
as failed attempts. Requests do not reread/hash the source pool or mutate caches.
Client file verification/loading remains inside actual preparation/E2E.

Official Python3.12 tarfile/gzip formats and exclusive-create behavior checked:
https://docs.python.org/3.12/library/tarfile.html,
https://docs.python.org/3.12/library/gzip.html.
Full guard and independent numerical correctness limits unchanged. Actual
remote publication/deployment and common performance envelope still pending.

### D78 post-reboot current-state correction (16:25+08)

Both hosts restarted outside this agent's actions. Remote boot
b53c1f16-1dd2-4741-9271-6b75e86582ec; local
aef67ed2-79d1-46bc-82e4-37401a5b45bf. Old artifact units are inactive, no listeners
on18080/18081. Never act on old PID/invocation identities.

Remote ethtool now reports1000Mb/s/full and partner1000baseT/Full advertisement;
local sysfs reports1000/full. Direct route and inspected qdisc unchanged. No
agent network modification; change's physical cause unknown. Negotiated rate
is not achieved application throughput. D77's100M diagnosis is retained as
historical evidence, not overwritten or used for new transport profiles.
Raw d78_20260927/post_reboot_link.json
SHA68acaf331f2f7a04490198805154455e5420026c3cc5aa496bca0e9679ecbde4.
Plan/snapshot currentSHAfe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c.

| D78 prerequisite | Result | Scope |
|---|---|---|
| Client/content unit checks |40pass,3.680s|No real remote performance claim|
| Basic smoke |288pass,68.316s|Bounded CPU unit, no model|
| Protected historical entries |147unchanged|No old results overwritten|
| Current remote disk/memory |149560934400B /107319648KiB available|Recheck at launch|
| Remote publication |Not started|3B then table,7B then table|

### D78 live offline publication (supersedes preceding not-started row)

Sourcee00e3f7c898cb90022361b0afc7381d9ba947d40 was tested/pushed and independently
read back from the Git remote before deployment. Versioned serverSHA
cdcdbe1f4a90db9a5546e77d386075b62baf88b5ee1fefdd678d6b420fb346bc and both index
SHAs matched on174; original source and source pools were not overwritten.

3B started16:29:46+08 in primelora-publish-tc-3b-d78.service,
invocationefce4fd2c63f4b83822de2ccdb88ff45, separately supervised by
primelora-publish-monitor-3b-d78.scope/a4d98a1482c44fc29e2e11fe3b9c8563.
Publication high1G/max2G/swap0,2SMT siblings; monitor high128M/max256M/swap0,
separate physical core. This is offline preparation, NOT frozen serving limits.
Conservative whole-two-pool disk bound97407250432B passed, no150GiB exemption
extended to inference. Explicit single-model script at rawd78/publish_one_remote.sh
SHA8bc12cb37ee7bde595ddc6287b45282dca465037259e035bdda6e04373e07497.

| Publication | Snapshot status | Interpretation |
|---|---|---|
|3B|63/500 exact-source-verified objects, ongoing|Not complete; no usable manifest yet|
|7B|Not started|Wait for3B completion/cleanup/table|
|Host safety|~102GiB available; no max/OOM/swap event|Offline high events retained|
|Inference / new HTTP qualification|Not started|No model or comparative result claimed|

### D79 observed transfer stages and integration qualification

Published serving now records object-open duration, accumulated object-read
duration and accumulated blocking socket-write duration, plus bytes read and
successfully written. Read duration can include filesystem/page-cache/storage
wait. Socket writes include backpressure and do not measure isolated wire time;
failed writes retain elapsed time without inventing partial byte counts.
All use one remote monotonic clock; client receive/verify spans remain separate.
No cross-host subtraction or sum of overlapping spans is treated as E2E.
Python3.12 primary contracts: https://docs.python.org/3.12/library/socket.html#socket.socket.sendall
and https://docs.python.org/3.12/library/time.html#time.monotonic_ns.

| Qualification | Observed result | Claim boundary |
|---|---|---|
|HTTP + real file-owner/router path|Cold fetch verifies archive/content; following GPU hit makes no second transfer|CPU inference fixture only|
|HTTP/lifecycle/preparation suite|182pass,6.310s|Source checks, not Full performance|
|Offline basic smoke|288pass,21.870s|Bounded CPU, terminal pass; verbose capture truncated|
|Old-result protection|147unchanged|No overwritten legacy results|
|Live offline3B publication at16:45|381/500, same invocation|Still incomplete, no HTTP/profile claim|

Raw d78_20260927/published_integration_suite.json SHA
18c77c24c096c2b0dae82a4b9b1d2f056f40080c2cf08b5d500d914e25f554be;
d79_basic_smoke_offline_terminal.txt SHA
08c6bc1f71f1cb0b53d687e48ae28c975c0bdd3da4fe51d4cc2cee1d9d9b99e7.
Failed launcher/test-development attempts and the explicitly TERM-stopped online
smoke remain raw evidence; they are not passes or inference-performance failures.
No D79 serving code is substituted underneath live D78 publication.
