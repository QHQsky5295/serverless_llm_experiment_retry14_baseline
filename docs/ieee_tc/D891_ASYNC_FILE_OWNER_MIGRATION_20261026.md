# D891 — async file-owner request-path migration

## Paper-level status

This checkpoint is a correctness and ownership qualification step, not a new
PrimeLoRA performance result. The latest 3B and 7B PrimeLoRA observations
remain below the frozen joint-SLO acceptance boundary, so no performance point
is frozen and no baseline experiment is started from this checkpoint.

The request and preparation paths can now use the same spawned file-state owner
without blocking the event loop on ordinary state transitions. The migration
covers demand witnesses, selected-copy acquisition, source snapshots,
preparation-plan registration/closure, native fallback leases, file-movement
interest, remote-publication verification, and host-budget configuration.
Byte copying and remote I/O remain in the existing owned I/O path; the child
process receives only state commands and receipts. No second controller ledger
or inferred release is introduced.

## Evidence

- `tests/test_ieee_tc_file_state_process.py`: 15 passed, including an actual
  async facade context whose entry and exit receipts are joined.
- The same module exercises the `ResidencyManager` process-mode path from
  verified NVMe publication through HOST materialization and quiescent close;
  both tiers are observed from the child-owned source snapshot.
- Request-path and demand suites: 14 additional tests passed.
- Python compilation and `git diff --check` passed.
- The full preflight file-state subset remains green. A broader preflight
  collection is environment-limited because this Python build does not expose
  `signal.pidfd_send_signal`; that unrelated platform test is not used as
  evidence for or against the file-owner migration.
- Backup commit: `05ba8409` (`D891`); rollback branch:
  `backup/d891-async-request-paths-20261026`.

## Boundary and remaining qualification

The rare retirement path now also joins process-owned native retirement and
runtime source release asynchronously; shutdown remains a separate lifecycle
operation, and manager shutdown now joins quiescent close off the event loop.
The opt-in process owner is still not enabled for a formal replay. The
no-GPU integration smoke now exercises verified publication, asynchronous
materialization, context closure, and manager shutdown. A later smoke must
also exercise demand cancellation and native retirement together before a
formal replay, so process-mode and legacy-owner identities cannot be mixed.
The production default and all historical results are unchanged. Only after
that smoke will one PrimeLoRA replay receive a new execution identity, subject
to ownership receipts, the native token contract, joint SLO, and physical GPU
accounting.
