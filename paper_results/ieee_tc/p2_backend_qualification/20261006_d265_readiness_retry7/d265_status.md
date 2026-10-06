# D265 7B readiness/dispatch diagnostic

Campaign: `d265_7b_readiness_8phase_full_retry7`  
Model: Llama-2-7B; 4,000 requests; 500-adapter frozen W0 trace; eight
500-request phases; `fixed_length_greedy_v1`; no-delay local-sim delivery with
the approved read-only `prepublished_gzip_v1` artifact cache.

## Gate evidence

| Check | Observation | Decision |
|---|---:|---|
| Arrived / terminal / success | 4,000 / 4,000 / 4,000 | pass |
| Terminal failures | 0 | pass |
| Native fixed-output/token-ID contract | 4,000 / 4,000 | pass |
| Dispatch-tier coverage | 4,000 / 4,000 (100%) | pass |
| Tier invariant conflicts | 0 | pass |
| Eight-cycle phase coverage | 8 / 8 | pass |
| Natural scale-up events | 10 (minimum 8) | pass |
| First-service samples | 10 (minimum 20) | **open** |
| Physical owner-union release | 15,883.368 GPU-s; no open lease | pass |
| Service/replay/watchdog/cleanup | all return 0; GPU context and path released | pass |

The dispatch-time readiness partition is GPU 1,190 (29.75%), HOST 1,684
(42.10%), NVMe 994 (24.85%), and Remote 132 (3.30%). The readiness booleans
agree with the recorded tier for all 4,000 rows. The ten first-service rows
are all Remote/cold and have no planned-adapter match; this is a coverage
observation, not evidence that the planner is incorrect.

## Interpretation boundary

The diagnostic is operationally complete but not a formal comparison point.
It is marked exploratory (`formal_run=false`, `n_correct=null`) and has no
frozen-settings/source-commit witness in its metadata. Its latency summaries
are retained for diagnosis only: overall TTFT is 2,159.8 ms mean and 7,581.6
ms P95; E2E is 6,510.8 ms mean and 14,457.7 ms P95. The physical lifecycle
quantity is the owner-union `physical_gpu_seconds_total`, not a legacy billing
estimate. The analyzer's first-service gate remains open because n=10; one
identically configured cold run is required and will be aggregated at the
run level before any readiness claim.

No request-time packing occurred. Request-triggered artifact transfer and
extraction remain in the application timeline; one-time remote publication,
administration and unrelated remote work are outside it. The run shows no
recurrence of the earlier controller exception, but it does not establish a
causal root cause for that exception.

Raw result SHA256:
`545da871f734a0a7b6dbf25fa97af66465a6cdc9db33ca67182d535c0ca1256a`  
Launch receipt SHA256:
`faf08552233d7869926e562e32f3ba2dc5b44860cbb7b73d9f56453100e8c834`  
Configuration SHA256:
`95717b4b1a6a53c0f0f1184f552672c909bcc91f6bbc5db40a8eec3dd84c754a`  
Launcher SHA256:
`d24565ae4e02d56fb56c0c6a0a18c040c729d5f9bb54aac5b4a73dcfd77d6027`
