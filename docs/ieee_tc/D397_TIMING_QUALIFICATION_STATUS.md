# D397 — native token timing preparation, not baseline qualification

2026-10-10. Latest user direction is one performance run per execution key,
no repeated-run means/CIs, 7B comparison before 3B. See the main repository's
`METRIC_PROTOCOL_SINGLE_RUN_V2.md` and `D397_RESIDENT_AUDIT_CORRECTION.md`.

D395 and D396 completed 4,000 requests but are diagnostics only. The replay
currently uses first nonempty payload as TTFT and response end in its TPOT
fallback; the dynamic artifact loader also runs in the auxiliary domain.
The Resident receipt is not equivalent to the main qualified owner/lease
ledger. Do not start a third unchanged repeat or freeze U_ref from these runs.

`scripts/native_stream_timing.py` is a prepared SSE parser, with seven CPU
tests in `tests/test_native_stream_timing.py`. Native output IDs alone advance
first/last delivery clocks. Keepalive, usage, finish and completion tail do
not count as tokens; fragmented UTF-8/CRLF, multi-token events and invalid
IDs are tested. This is not yet integrated into `replay_openai_trace.py` and
does not by itself make any old or new baseline timing protocol-compliant.
Client delivery times must not be confused with backend kernel token times.

The pre-existing replay and RelayServe modifications remain unstaged and
unchanged. No baseline scheduling/loading semantics were modified in this
checkpoint. Serverless remains first in the next baseline audit.
