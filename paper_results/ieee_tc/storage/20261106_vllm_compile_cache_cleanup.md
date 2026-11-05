# 2026-11-06: regenerable vLLM compile-cache cleanup

## Scope and authorization

The inference node was below the frozen 150 GiB free-space start condition. A
read-only capacity audit found two derived vLLM caches as the only local
objects large enough to restore the margin without touching experiment inputs:

* `/home/qhq/.cache/vllm/torch_compile_cache`
* `/home/qhq/.cache/vllm/device_compile_cache`

Their measured directory usage before cleanup was 811,433,984 bytes and
464,216,064 bytes, respectively (1,275,650,048 bytes total). Neither path is
an adapter, base-model weight, trace, result, source file, or remote artifact.
Both are regenerable compiler products.

## Safety checks

The check found no running vLLM, PrimeLoRA, PyTorch, or experiment replay
process, and no open handle under either cache directory. No GPU process was
active. The cleanup therefore occurred outside a measurement window and cannot
alter a recorded run.

## Action and boundary

Only the two cache directories were removed. The original experiment result
trees, frozen adapter pools, model weights, remote backups, user-modified
manifest, and historical evidence were not touched. A later vLLM execution may
rebuild these caches; the cleanup is not a code or configuration change.
