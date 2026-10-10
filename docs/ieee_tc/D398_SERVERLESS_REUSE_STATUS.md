# D398 Serverless native diagnostic reuse

2026-10-10. No model/service launch, policy change, remote mutation or result
overwrite. The 7B polling pair D72/D73 is complete and must not be repeated.

The existing summarizer now accepts `--tc-reuse-summary` and
`--tc-reuse-evidence` with `--replay` and a new `--output`. It checks every raw
source SHA, reproduces the sealed aggregation, and reuses the existing native
HTTP validator to reconstruct metrics from token timestamps. A consistent
derived component sum alone is not sufficient evidence of correct TTFT.

D72: 24 raw references match; 997 native responses revalidate, maximum metric
error 0 ms, 3 failures retained. D73: 25 references match; 996 revalidate,
maximum error 0 ms, 4 failures retained. Existing summaries reproduce exactly.
R2 diagnostic reuse only: local artifacts, 1000-request development prefix,
old backend/configuration, incomplete workload, no physical lease ledger.

60 bounded CPU tests pass (18.890 s): dispatch/reuse audit, native RR behavior,
launch/loader, measurement/open-loop HTTP, old bandwidth summary, and native
SSE helper. No GPU performance inference follows from these tests.

Before a new matched 7B run, retain native fast loading and repair the missing
startup admission, service-owned published-artifact delivery, and actual
GPU acquisition/release accounting. Native store contexts count too. The
registration-time adapter preparation policy must not silently change when
moving local paths to remote delivery. Inspect and disclose any backend-specific
compatibility path. Do not shift business t0, conceal retries, or perform a
third Resident replay with the invalid first-payload/EOF timing path.

Full evidence and next action are in the main repository:
`docs/ieee_tc/D398_SERVERLESS_7B_REUSE_AUDIT.md` and
`paper_results/ieee_tc/serverless_audit/20261010_d398_*_reuse_audit.json`.
User-modified `replay_openai_trace.py` and relayserve script remain untouched.
