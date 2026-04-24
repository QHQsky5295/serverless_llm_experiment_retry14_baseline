# S-LoRA Environment

S-LoRA uses an isolated CUDA 11.8 environment and must not share dependencies
with FaaSLoRA, SGLang, vLLM, or ServerlessLLM.

## Current Environment

- Conda env: `/home/qhq/anaconda3/envs/slora_official_cu118`
- Python: environment-specific Python under the path above.
- CUDA toolkit: CUDA 11.8 from the `nvidia/label/cuda-11.8.0` channel.
- PyTorch/Triton family: aligned with the official S-LoRA CUDA extension path.
- Upstream repo: `/home/qhq/serverless_llm_baselines/repos/S-LoRA`
- Project entry: `/home/qhq/serverless_llm_baselines/S-LoRA_project`

## Current Rules

1. Use native `/generate_stream`, not the OpenAI chat endpoint.
2. Use the shared trace and shared adapter subset from the active round.
3. Use S-LoRA-native prompt budgeting with special tokens included.
4. Fail fast if token source falls back to raw trace expected tokens.
5. Do not patch S-LoRA algorithms to mimic PrimeLoRA mechanisms.
