# D1068 bounded qualification record

The bounded CPU qualification ran inside the existing `run-experiment` resource
envelope (`MemoryHigh=2 GiB`, `MemoryMax=3 GiB`, `MemorySwapMax=0`, CPUs
2,3,26,27, `CUDA_VISIBLE_DEVICES=`). The final single-test spawned-worker run
completed successfully after the local Humming compilation path finished.

The first five-test attempt is retained as a negative execution record: four
tests passed, then the final test was interrupted while its child was blocked in
local extension compilation. It was not classified as a production failure and
was not silently discarded. The final test was rerun alone and passed.

No GPU process, remote service, shared k3s service, artifact pool, trace, or old
result was modified. This qualification does not establish end-to-end benefit,
adapter numerical correctness, G1/G2 eligibility, or the fraction of production
epochs that will trigger the proof.
