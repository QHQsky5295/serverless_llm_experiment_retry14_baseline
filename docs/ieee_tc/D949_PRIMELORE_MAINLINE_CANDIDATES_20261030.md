# D949: PrimeLoRA mainline candidate selection

After the ServerlessLLM 7B formal run and the D943/D946/D947 initial-NVMe
experiments, the current PrimeLoRA candidates are selected by the frozen G1
ordering rather than by TTFT alone.

## 7B

Use D931 (`d931_20261029/7b_current_guard_off_w2`) as the current 7B mainline
candidate.  It completed 4,000/4,000 requests, with 98.1% TTFT-SLO attainment,
1,215.95 ms average TTFT, 2,997.74 ms P95 TTFT, and authoritative physical
GPU usage of 11,716.06 GPU-seconds (2.9290 GPU-seconds/request).  D947 has
lower TTFT (1,144.78/2,464.45 ms average/P95) and 98.5% SLO, but uses
13,296.43 physical GPU-seconds (3.3241/request), so it is not the G1 choice.

## 3B

Use D937 (`d937_20261030/3b_full_w2_nvme_extent`) as the current 3B mainline
candidate.  It completed 4,000/4,000 requests, with 99.975% TTFT-SLO
attainment, 775.71 ms average TTFT, 2,204.69 ms P95 TTFT, and 11,941.41
physical GPU-seconds (2.9854/request).  D935 is retained as a nearby current
control: its physical usage is 11,936.55 GPU-seconds, but its average/P95 TTFT
are 803.77/2,302.04 ms and SLO attainment is 99.875%.  The difference is too
small to claim a resource win for D937; the 3B choice is retained for its
slightly better latency/SLO while reporting both physical measurements.

## Correctness status

Both selected candidates have 4,000/4,000 native contract-complete requests,
fixed-length greedy output matching, and vLLM token IDs for every request.  No
independent numerical oracle was run against the full workload, so `n_correct`
remains null and neither candidate is called numerically correct beyond the
contract evidence.  This is a publication limitation, not a silently imputed
100% correctness value.
