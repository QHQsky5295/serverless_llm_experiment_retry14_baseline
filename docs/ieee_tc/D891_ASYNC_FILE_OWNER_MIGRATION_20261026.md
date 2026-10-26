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

- `tests/test_ieee_tc_file_state_process.py`: 14 passed, including an actual
  async facade context whose entry and exit receipts are joined.
- Request-path and demand suites: 14 additional tests passed.
- Python compilation and `git diff --check` passed.
- The full preflight file-state subset remains green. A broader preflight
  collection is environment-limited because this Python build does not expose
  `signal.pidfd_send_signal`; that unrelated platform test is not used as
  evidence for or against the file-owner migration.
- Backup commit: `05ba8409` (`D891`); rollback branch:
  `backup/d891-async-request-paths-20261026`.

## Boundary and remaining qualification

The opt-in process owner is still not enabled for a formal replay. Before that
step, the remaining synchronous lifecycle paths (notably rare retirement and
shutdown calls) must be audited so a process-mode run cannot silently block or
mix owners. The production default and all historical results are unchanged.
Only after that audit will a new no-GPU integration smoke be run, followed by
one PrimeLoRA replay under a new execution identity if the ownership receipts,
native token contract, joint SLO, and physical GPU accounting all remain
valid.
