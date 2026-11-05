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

The same audit also identified `/home/qhq/.cache/huggingface/hub/
datasets--anon8231489123--ShareGPT_Vicuna_unfiltered` (672,890,880 bytes) as
a downloaded dataset cache unrelated to the frozen IEEE trace or adapter pool.
The current campaign reuses the existing trace and does not regenerate it from
this dataset, so this cache is also regenerable and outside the measured input
set.

## Safety checks

The check found no running vLLM, PrimeLoRA, PyTorch, or experiment replay
process, and no open handle under either cache directory. No GPU process was
active. The cleanup therefore occurred outside a measurement window and cannot
alter a recorded run.

## Action and boundary

Only the two vLLM compile-cache directories and the unrelated ShareGPT dataset
cache were removed. The original experiment result
trees, frozen adapter pools, model weights, remote backups, user-modified
manifest, and historical evidence were not touched. A later vLLM execution may
rebuild these caches; the cleanup is not a code or configuration change.
