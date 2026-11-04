# Pre-test launcher failure retained

2026-11-04: first CPU qualification scope was named
`d1063-source-auth-c283fb02643a4b9c88c423e1ac180202.scope`.
Existing `ieee_tc_preflight.scope_identity` rejected it with
`ValueError: not an owned TC service/test UUID scope` before suite loading.
No test JSON or result file was created. No model or GPU work started.

Used the existing `primelora-tc-test-<32 hex>.scope` convention for the next
launch; did not relax the resource or ownership checks. That next launch is
`source_tests1`, which retains its own separate test failures and JSON receipt.
