# D987 — corrected predictive-horizon preflight (2026-10-31)

D986 is retained as a failed pre-execution attempt. Its output root had not
been created before invoking the disk preflight, which intentionally refuses to
stat a nonexistent publication directory. No service, worker, request, or GPU
result was produced.

D987 is a new execution key. Before invoking the existing runner, the launcher
must create its unique output root, then use the unique NVMe and memory-backed
host cache roots from the configuration. All model, trace, adapter, endpoint,
and predictive-control semantics remain unchanged.
