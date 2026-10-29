# D946: initial-NVMe preload timing failure

The 24 GiB follow-up candidate used a fresh key:

`d946_20261030/7b_initial_nvme_preload_w2`

The capacity guard passed and all 500 fixed adapters were materialized into
NVMe.  The run was stopped before completion because the preload was executed
after the external replay's deployment notice.  Requests therefore arrived
while the one-time transfer was still in progress.  The live service evidence
showed dispatch wait above 100 seconds and TTFT/E2E above 100 seconds for the
first admitted requests.  This is a protocol violation for a deployment-only
operation, not a serving result.

D946 is excluded from every comparison.  Its request ingress, watchdog,
service log, launch receipt, and a lossless archive of the 13 GiB materialized
NVMe tree are retained.  Archive:

`nvme_initial_preload_cache.tar.zst`

SHA-256: `fdd7748c74cd2c9f3a0f1396b5ae304395c8197930644af39cc1ee57f0c74552`

The archive is a storage-preserving copy of the unique cache evidence; the
unpacked duplicate was removed after archive creation and verification.

The next design must place the one-time materialization before the replay
deployment notice, or move the notice/arrival clock only after the materialized
cache is fully published.  It must also record deployment wall time separately
and prove that request ingress did not start during the transfer.
