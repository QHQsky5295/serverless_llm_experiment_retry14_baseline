# D395 7B Resident reference R1 audit

> Correction, 2026-10-10: the completeness judgment below is superseded by
> [D397](D397_RESIDENT_AUDIT_CORRECTION.md). Native output counts are valid,
> but token-time endpoints and service/resource attribution are not qualified.
> D395 must not define U_ref or enter a final ranking. Historical numbers and
> the original interpretation are retained below for provenance, not endorsed.

## Decision

D395 is a completed, audited candidate for the three-run 7B Resident reference
series. It is not yet the frozen common reference, final `U_ref`, or a formal
G1/G2 result. The raw result is retained outside the Git repository; this
document and the adjacent JSON are the small, reviewable evidence record.

## Experimental identity

The run used the pre-registered W0 trace and adapter subset without regenerating
weights, workload, or artifacts:

- model: `llama2_7b_main_v2_publicmix`;
- workload: `llama2_7b_auto500_formal4000_s8`, 4,000 requests and 500-adapter universe;
- seed: 42;
- trace SHA-256: `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`;
- adapter-subset SHA-256: `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`;
- vLLM: 0.30.0, DP=4, TP=1, `max_num_seqs=2`, `max_loras=4`;
- generation contract: `fixed_length_greedy_v1`.

## Request contract and timing audit

All 4,000 records completed with HTTP 200 and one attempt. All records matched
the fixed-output request map and used vLLM native token IDs; no record used a
trace-output fallback, had an invalid native token ID, or ended as an empty
success. Request IDs and prompt hashes were complete. The decomposition checks

\[
  T_{\mathrm{E2E}}=T_{\mathrm{dispatch}}+T_{\mathrm{service,E2E}},\qquad
  T_{\mathrm{TTFT}}=T_{\mathrm{dispatch}}+T_{\mathrm{service,TTFT}}
\]

had maximum absolute error 0 ms over the 4,000 records. TPOT was observed for
all requests.

Using the current-code D394 warm candidate only as a diagnostic threshold
(not as a frozen common reference), the joint TTFT/TPOT criterion accepted
3,846/4,000 requests (96.15%). The 1,039 requests with input length at most
616 tokens used 2,995.349 ms / 59.744 ms TTFT/TPOT thresholds; the remaining
2,961 requests used 4,633.345 ms / 85.247 ms. This result must not be promoted
to the final G1/G2 denominator until the three-run common-reference process is
closed.

The measured averages were 1,882.464 ms TTFT, 26.533 ms TPOT, 4,902.365 ms
E2E, and 1.009 requests/s. P95 TTFT and E2E were 3,148.881 ms and 7,739.531
ms, respectively. These are audit values, not a claim that Resident is the
target system's final comparison winner.

## Physical GPU lifecycle

The receipt records one contiguous allocation interval for each of four GPUs,
from monotonic time 1,105,911.922902393 to 1,109,948.629676429, or
4,036.706774036167 s per GPU. The physical allocation-union value is therefore

\[
U_{\mathrm{physical}}=4\times4{,}036.706774036167
 =16{,}146.827096144668\ \mathrm{GPU\text{-}s}.
\]

The release event occurred after worker exit, reported `final_nvml_empty=true`,
and the release inventory showed 15 MiB on each GPU. The legacy summary also
reports 17,806.483043 GPU-s because it includes the 300-s deployment idle tail;
that value is retained for reproducibility but is not silently substituted for
the physical receipt interval in the lifecycle audit.

## Remote delivery audit

The request path used the approved read-only `prepublished_gzip_v1` cache. No
request-time packing or temporary archive creation occurred. The remote journal
contains 132 successful transfers, one per client-reported remote miss, totaling
131,480,060 wire archive bytes; 3,868 requests were cache hits. The client’s
`remote_lora_bytes` total (3,300,789,780 bytes) is the raw-artifact accounting
field and is intentionally reported separately from the gzip wire-byte total.

The remote process was an independent artifact-node service. Its PID and full
command line were verified before stopping PID 1946848. After the controlled
stop, PID 1946848 was gone and port 18081 was closed. The transfer journal and
launch-log SHA-256 values are recorded in the adjacent JSON, so the stop did
not alter the evidence.

## Next gate

The next permitted action is D396, a separate Resident reference run with the
same frozen inputs and protocol. No 3B, baseline, ablation, sensitivity, or
figure result is opened by D395. After three audited Resident references, the
common threshold and `U_ref` freeze review must be completed before any G2
budget comparison or Prime/baseline ranking is accepted.
