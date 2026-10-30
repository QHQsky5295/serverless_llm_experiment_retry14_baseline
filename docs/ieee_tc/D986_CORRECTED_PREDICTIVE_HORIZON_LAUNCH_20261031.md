# D986 — corrected predictive-horizon launch (2026-10-31)

D985 is retained as a failed pre-execution attempt. Its copied configuration
still pointed at the D955 NVMe and `/dev/shm` workspace roots, so the runner
correctly stopped with `FileExistsError` before any request arrived. The
preflight receipt, service log, and cancelled replay publisher are preserved
under the D985 output directory; no GPU result was produced.

D986 is a new execution key with the same frozen model, W0 trace, adapter
subset, remote endpoint, and predictive controller. The only launch correction
is a fresh NVMe workspace and a fresh memory-backed host workspace. It must be
evaluated independently and never merged with D985.
