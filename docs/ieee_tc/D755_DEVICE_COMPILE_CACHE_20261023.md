# D755: preserve native compilation with per-device cache ownership

The 3B D754 native ServerlessLLM run completed 100 HTTP responses but only two
of four engines became ready. Two torch2.8/vLLM0.10.2 processes failed renaming
temporary files under one shared rank_0_0 compilation artifact directory.

Pinned torch CompiledArtifact.save(unpacked) removes the target directory before
unpacking. A bounded two-process test of this actual method reproduces removal
of the other writer's temporary file; disjoint roots avoid that failure.
Only the tiny artifact payload is mocked. It is not a model/performance test.

The source adapter now accepts an explicit device_compile_cache_root. Before
engine construction it sets the documented VLLM_CACHE_ROOT to a persistent
hostname/assigned-CUDA-device directory. Native config/code hashing remains;
no shared cache deletion, per-run cold cache, eager fallback, retry, or native
policy change. Scope: exclusive one TP1 native engine per assigned physical GPU.
Actual selection is included in the existing durable ready receipt.

17 affected source/config/receipt tests passed. Prior 114 native and 89 input
checks are reused only for unchanged paths. D756 is the necessary real-engine
qualification, not completed by these CPU tests. No RR/scaling/loading/batch
parameters or workload changes, no baseline performance search. D754 retained.

Sources: https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/standalone_compile.py
and https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/compilation/backends.py
and https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/envs.py.
