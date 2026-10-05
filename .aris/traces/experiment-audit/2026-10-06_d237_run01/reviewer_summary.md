# Fresh reviewer trace summary: D227 retry2

Reviewer: gpt-5.6-sol ultra, same-family provisional audit.  The reviewer and
its raw-result checker read the retry2 launch receipt, replay/ingress/terminal
journals, physical deployment summary, result JSON, config, and relevant
source paths.  No files, GPU services, or remote services were modified.

Key findings returned on 2026-10-06:

- All 22 request IDs match across replay, ingress, terminal, and result; source
  and canonical-prompt hashes, completion-token hashes, target/output lengths,
  and the seven launch evidence hashes recompute consistently.
- 22/22 native token-contract terminals completed successfully.  This is not
  semantic correctness: `eligible_correctness=false`,
  `correctness_qualification=not_inferred_from_token_contract`, and
  `n_correct=null`.
- Physical owner-union GPU time is 348.7213928719284 GPU-s with all leases
  released.  The 602.5108013 infrastructure billing field includes a 300-s
  idle tail and is not the lifecycle metric.
- The configured effective-capacity gate is true, and control events saw
  queue depths 22, 18, and 17, but native request evidence has zero GPU
  admission decisions and no physical KV reservation.  The ordinary native
  `demand_load_and_acquire` route bypasses the effective native E(t) admission
  transaction.  This is a gate-execution failure/root-cause diagnostic, not a
  no-pressure result and not a performance ranking.
- The raw result JSON is in a sibling result directory rather than under the
  launch tree, and its SHA is absent from `launch.json.evidence_sha256`.  The
  audit therefore marks chain-of-custody WARN and records an explicit
  path/SHA manifest for later sealing.
- The first launch was a separate `qualification_command_failure` caused by a
  missing published-remote delivery variable; it created no request terminals
  and is not a worker or performance failure.

Overall reviewer verdict: bounded contract/lifecycle integrity PASS; semantic
correctness unknown/not inferred; admission-gate qualification FAIL; scope
and chain-of-custody WARN; evaluation type `self_supervised_proxy`.  A targeted
native admission-stress test with a typed owner/epoch/decision/reservation
receipt is required before any full replay.
