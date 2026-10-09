# D375 7B capacity-revalidation diagnostic: profile qualification failure

Date: 2026-10-10

D375 passed the unchanged host disk/memory/GPU/remote preflight and entered the
controlled service path, but it terminated during `ScenarioRunner` startup
before the first request. The native profile loader rejected the configuration
because the requested C=8 model included
`ieee_control_interleave=atomic_control_yield_v1` and
`ieee_file_demand_transaction=protected_file_owner_turn_v1`, while the bound
profiles were measured for C=14. The service log reports
`ValueError: service profile model/configuration/context mismatch`.

The run therefore has no request observation and is not a performance result.
The replay publisher emitted `N_arrived=0`, `N_submitted=0`, and
`replay_incomplete`; the launcher classification is
`qualification_command_failure`. GPU context release was confirmed, owned
services and auxiliary scope were removed, and the remote units were stopped.
The retained receipt is under
`results/ieee_tc/p2_backend_qualification/d375_20261010/7b_capacity_revalidation_100/`.

This is a provenance/qualification failure, not evidence against C=8 or
against PrimeLoRA. The existing D298/D321 profiles are C=8 but do not include
the two later opt-in policies. The D362 profiles include both policies but are
C=14. The next step must choose one of two explicit, reproducible paths: (1)
run the C=8 configuration with its already-qualified D298/D321 profile set,
which isolates the capacity envelope while retaining the current source-
identity code; or (2) perform a new measured C=8 source/profile qualification
for both policies before using them. No profile hash or consistency check may
be bypassed.
