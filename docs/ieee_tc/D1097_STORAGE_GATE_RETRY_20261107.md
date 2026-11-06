# D1097 — storage-gate retry of the D1096 validation

D1096 did not launch a service or replay a request: the existing inference-node
disk gate observed 149.2 GiB free against the frozen 150 GiB start floor. Its
launch receipt and empty-result failure are retained as protocol evidence.

To recover the gate without deleting experiment evidence, the historical,
non-active D90 diagnostic directory was losslessly archived as:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d90_20260927.tar.zst`

Archive SHA-256:

`fcda9770f8092effd303b4154dfcebaab6ef28f1eed400647451ad1d96ba0f8a`

The archive was tested with `zstd -t` and its member listing was checked before
removing only the expanded copies. No unique LoRA artifact, frozen trace,
published result, or current execution directory was removed. The recovered
free space is 152 GiB, above the unchanged 150 GiB inference-node floor.

D1097 is therefore a new execution key for the same bounded D1096 guard
validation, with identical model, trace, adapter subset, generation contract,
remote delivery, and guard setting. It is not a second result selected by
performance; it exists solely because D1096 never entered the service phase.
