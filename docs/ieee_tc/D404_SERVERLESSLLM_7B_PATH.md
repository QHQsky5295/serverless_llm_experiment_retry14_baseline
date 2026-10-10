# D404: repaired ServerlessLLM 7B allocation path qualifies

2026-10-11. Execution baseline3cc25a87f3895a408a16efe2ef9c75d0e8780ff3,
main abd53201f0ed5959d554541a4b86beae2cfef108. Single100-request affected-path
qualification; no full performance/configuration claim, no polling-pair repeat.

100/100 native response contracts,17,369tokens,0failed; four backend-ready
receipts and four served instance IDs. Actual storage-aware scheduler source
matches the D403 opt-in view; original placement/scaling/loading unchanged.
Readback:loop alive,no exception,empty pending queue,correct service containment.
All29published HTTP downloads join content/receipts/server;0online packs.
Physical ownership closes at1362.0847175046802GPU-s. GPU/remote services released,
loader-only overlay restored;noOOM/swap/high events or protected-result changes.

Mean TTFT82.940567s,router queue82.332853s,serviceTTFT433.976789ms. Residual
startup/capacity queueing remains;target2/max-seqs4 is not selected optimum.
Native correctness/source/timing contracts do not add independent numerical
LoRA correctness.29touched IDs do not close500-ID engine qualification.

Main report:docs/ieee_tc/D404_SERVERLESSLLM_QUEUE_QUALIFICATION.md.
Main curated audit:paper_results/ieee_tc/baseline_audit/d404_7b_queue/audit.json,
SHAde80d7f8adbe92e4aaec8a6a4950d905ffd3da263a5c7acdff31c4c514290f3e.
Evidence manifest binds125files;same100request contracts as D402 but startup
timing differs. Do not attribute all observed latency change to this repair.

Next:reusable500-ID native-engine evidence audit,missing affected functional
coverage,reasonable public configuration validation and1000 compatibility,
then matched4000W0. D72/D73 local failed prefixes remainR2;D80HTTP/content
coverage is not an engine execution test. No new datasets or PrimeLoRA retuning.
