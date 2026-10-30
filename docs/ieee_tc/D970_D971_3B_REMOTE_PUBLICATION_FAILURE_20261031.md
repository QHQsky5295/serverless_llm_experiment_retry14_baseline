# D970/D971: 3B remote-publication qualification failures

## Status

D970 and D971 are failed setup/qualification attempts, not PrimeLoRA
performance points. Both stopped after the first one to three arrivals and
produced no complete request result.

## D970: wrong publication endpoint

D970 initially reused the 7B endpoint `http://192.168.4.174:18081` for the 3B
run. The first remote transfers returned `not_published` for the 3B content
manifest `ac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e`.
The owner correctly refused an automatic retry and the run terminated with
`failed movement requires explicit owner recovery`.

## D971: historical 3B endpoint also unavailable

D971 used the endpoint in the historical 3B wrapper,
`http://192.168.4.174:18080`, with all other inputs unchanged. The same
manifest was still reported as `not_published`; no artifact bytes were
transferred. A direct SSH probe to `10.199.227.14` from the inference host
returned `No route to host` at the time of the attempt. This is therefore a
remote service/publication availability failure, not an algorithmic result.

## Consequence

No 3B latency, SLO, correctness, lifecycle GPU-s, or G1/G2 claim may use D970
or D971. The runner was not changed to retry, substitute a local artifact, or
silently switch to a different 3B content set. A new 3B run requires a
reachable remote service whose published manifest matches the frozen 3B content
index; until then the 3B branch remains pending while the completed 7B audit is
preserved.

## Evidence

- D970 outcome:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d970_20261031/3b_overlap_lifecycle_full/launch.launch/physical_deployment/main_outcome.json`
- D971 outcome:
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d971_20261031/3b_overlap_lifecycle_full/launch.launch/physical_deployment/main_outcome.json`

