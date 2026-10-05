# D218 full replay failure record

`d218_20261005/7b_confirmed_snapshot_cache_full_replay` is retained as a
failed run and is excluded from every performance summary. The guarded launch
started the service and the frozen 4,000-request replay publisher, but the
remote artifact endpoint was not available when the first requests were
served. The first request recorded
`RemoteArtifactError: download failed for finance_lora: connection refused`.
The subsequent residency task then raised
`RuntimeError: failed movement requires explicit owner recovery, not an
automatic retry`; this is the queue's fail-closed recovery contract, not a
valid latency observation.

Observed population: 4,000 planned, 4 terminal requests, 0 native output
contracts complete, `measurement_complete=false`, and no performance metrics
are publishable. GPU context release and service-path removal were confirmed.

The remote 7B service was subsequently started through the existing user
service and authenticated `/health` plus artifact requests were verified from
the experiment host. D219 is a fresh run-key with the same trace, adapter
subset, candidate gate, and backend; it is the only retry eligible for
analysis. D218 is not deleted or overwritten.

Raw evidence:

- `launch.json` SHA-256: `0ab8ce333adfe4019ccf14567c1846a25ecbc3f16b9db0cc64ddec615f5b4644`
- `physical_deployment/main_outcome.json` SHA-256: `f23308ff7a5b3ee95d7c543676541b7f7069bf08ada790a59a589c474cce2fab`
- `physical_deployment/summary.json` SHA-256: `75454d86e5693e49b8f7497214b97ce1fc56b81cc99326a199047d781a2cd38c`
- `service.log` SHA-256: `8ab72851fc784b2fc60214f37c8e67b8b4dcc082bb256d2a7a950d193cfed575`
- `replay.jsonl` SHA-256: `630c5ae1354c11d37b60a34defc67e08ef02e324158afe6e923eef332a4fd4e1`
