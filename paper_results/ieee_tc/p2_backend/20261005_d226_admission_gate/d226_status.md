# D226 status

This is a read-only configuration audit, not a serving run.

| item | observation |
|---|---|
| D219 scenario label | `tc_ieee_full` |
| effective-capacity gate | `false` |
| admission decision/admit/defer/reject counts | `0/0/0/0` |
| root cause | YAML omits the key; code default is false |
| valid performance result | none added |
| next execution key | new 22-request gate=true diagnostic, only after resource and numerical gates |

See `docs/ieee_tc/D226_D219_ADMISSION_GATE_AUDIT.md` for the exact source locations
and acceptance rules. D219 remains a route/full diagnostic and is not promoted to
the formal Full or G1/G2 denominator.
