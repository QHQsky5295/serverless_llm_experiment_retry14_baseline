# D273 source-plan supersession audit

Date: 2026-10-06  
Scope: CPU-only semantic audit; no model startup, GPU replay, remote mutation,
or performance result.

## Initial hypothesis

The D272 contract table raised a narrow concern: `_validate_source_binding`
has a same-content path-migration shortcut, and a direct call with an active
preparation plan appeared to accept a path different from that plan's row.
The falsifiable question was whether this represented an unsafe plan-binding
bug in the request path.

## Controlled check

The direct check used the existing `NativeDemandTransactions` fixture: it
loaded adapter 4 with a verified SHA, released and evicted the old native copy,
installed a plan whose selected source was `/existing/adapter-4`, and called the
validator with `/new-tier/adapter-4` and the same SHA. The direct call accepted
the source change.

That observation is not sufficient to classify a bug, because the call did not
carry a preparation-plan identity. The real request-side demand path is
deliberately allowed to introduce a newly confirmed copy; the enclosing
controller then supersedes the stale plan and advances the source epoch rather
than silently applying the old objective. The existing integration tests encode
this contract at staging, native-HOST, handoff, movement, and native-commit
boundaries.

## Validation

After reverting the exploratory source change, the complete transfer-pressure
suite passed:

```text
Ran 169 tests in 11.086s
OK
```

The focused GPU-reference suite also passed 135/135 tests. The seven relevant
automatic-replacement tests passed independently. No production source or
validator change was retained from the hypothesis check.

## Decision

The apparent bypass is a boundary distinction, not evidence of a confirmed
implementation defect: an unplanned request source is an input that can
supersede a stale plan, whereas an operation carrying an explicit plan ID is
checked against its frozen objective by the higher-level native preparation
path. D273 therefore makes no code change and does not reinterpret D266. A
future diagnostic should preserve both fields—whether the source operation was
plan-owned and whether a controller supersession was recorded—before treating a
same-content path transition as an identity conflict.
