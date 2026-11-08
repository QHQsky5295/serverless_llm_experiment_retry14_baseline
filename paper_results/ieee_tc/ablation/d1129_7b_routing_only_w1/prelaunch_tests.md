# D1129 targeted CPU checks

2026-11-08. `32 passed, 6 warnings in 2.26s` under pytest9.1.1/Python3.12.12.
No GPU runtime, new workload or performance replay started.

Unit: `primelora-tc-d1129-focused-tests-20261108.service`.
Invocation: `8f295c3b582345c7969ffd31f9d09ddf`.
MemoryHigh3GiB/Max4GiB, swap0, CPU2/3/26/27, CUDA hidden, explicit repo cwd.

Selected tests:

- tests/test_ieee_tc_launch.py::RoutingOnlyContract
- tests/test_ieee_tc_launch.py::ElasticOnlyContract
- tests/test_ieee_tc_launch.py::ElasticLocalityContract
- tests/test_ieee_tc_launch.py::IEEEIntegratedLaunchContract
- tests/test_ieee_tc_service_routing.py::CommittedRoutingContract
- tests/test_ieee_tc_service_routing.py::LoadOnlyRoutingContract

Checks cover complete non-target configuration equality, actual service-bin and
demand-file owners, explicit scenario, prior Full/ElasticOnly/ElasticLocality
contracts, feasibility, stable selection, and non-mutating shadow decisions.
They do not certify measured performance, native numerical equality or causal gain.

Warnings: one already-imported anyio assertion-rewrite warning, one unknown
asyncio_mode configuration warning, and four existing Pydantic class-config
deprecations. The CUDA-hidden test process also reported no CUDA runtime; this
is expected for these CPU fixtures, not a GPU qualification result.

Main exited0. Manager wrapper terminal status is recorded separately in
prelaunch.json; no test rerun merely for the previously observed stop timeout.
