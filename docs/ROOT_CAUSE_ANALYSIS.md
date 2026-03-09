# FaaSLoRA 实验「已杀死」根因分析报告

## 一、现象

- **weights_only 错误**：GPU warmup 阶段加载 LoRA 时报错  
  `Cannot use weights_only=True with files saved in the legacy .tar format`
- **进程被杀**：Phase 2 Serving 刚开始即出现「已杀死」（通常是 Linux OOM killer）

---

## 二、问题 1：weights_only 错误

### 2.1 调用链

```
run_all_experiments.py
  → ScenarioRunner._warmup_gpu()  [约 1387 行]
    → engine.generate("Hi", local_path, aid, ...)
      → InferenceEngine._generate_transformers()
        → _sync_generate_transformers()  [808-917 行]
          → PeftModel.from_pretrained() 或 load_adapter()
            → peft/utils/save_and_load.py: load_peft_weights()
              → torch_load(filename, map_location=...)  [695 行]
                → torch.load(..., weights_only=True)  [592-598 行]
```

### 2.2 根因 1：PEFT 显式传 `weights_only=True`

**文件**：`peft/utils/save_and_load.py`

```python
# 591-598 行
def torch_load(*args, weights_only=True, **kwargs):
    """Call torch.load and handle weights_only.
    Defaults to weights_only=True to anticipate upcoming switch on the PyTorch side.
    """
    return torch.load(*args, weights_only=weights_only, **kwargs)
```

- PEFT 默认传入 `weights_only=True`，与 PyTorch 2.6+ 新默认一致
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` 只在调用方**未显式传入** `weights_only` 时生效  
- PEFT 显式传入后，环境变量不会覆盖，因此 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 无效

### 2.3 根因 2：当前 patch 只覆盖了 `torch.serialization.load`

**文件**：`run_all_experiments.py` 88-100 行

```python
_ts.load = _faaslora_torch_load  # 只 patch 了 torch.serialization.load
```

- PEFT 使用 `torch.load()`，通常绑定到 `torch` 模块的 `load`
- Python 模块导入时，`torch.load` 与 `torch.serialization.load` 可能是同一对象，也可能不是
- 若 `torch.load` 在导入时就复制了 `serialization.load` 的引用，后续修改 `_ts.load` 不会改变 `torch.load`
- 因此需同时对 `torch.load` 做 patch

### 2.4 根因 3：适配器使用 legacy .bin 格式

**本地适配器**：`artifacts/remote/finance_lora/`

```
adapter_config.json
adapter_model.bin    # 4KB 或更大（legacy pickle/tar）
adapter_data.bin    # 填充文件
```

**生成逻辑**（`generate_synthetic_lora`，约 2237-2257 行）：

- 优先创建 `adapter_model.safetensors`（需 safetensors）
- 失败时创建占位 `adapter_model.bin`（4KB 零字节）或真实 `torch.save` 的 .bin
- 若只有 .bin，PEFT 会走 `torch_load()`，从而触发 `torch.load` + `weights_only=True`，在 legacy 格式下报错
- 若有 .safetensors，PEFT 会走 `safe_load_file()`，不会调用 `torch.load`，因此不会触发该错误

---

## 三、问题 2：Phase 2 进程被杀死

### 3.1 可能的 OOM 触发点

| 阶段 | 操作 | 显存/内存 |
|------|------|-----------|
| 基座加载 | Qwen2.5-3B fp16 | ~6GB GPU |
| GPU warmup | 尝试加载 1 个 LoRA（当前失败） | 若成功约 +30MB |
| Phase 2 首个 batch | 2 个并发请求，各需加载 LoRA | 每个约 +30-45MB |
| LoRA 切换 | delete 旧 + load 新 | 瞬时显存峰值 |
| KV cache | 2 并发 × (256 输入 + 64 输出) tokens | 约 1-2MB |

### 3.2 相关代码与配置

**并发与批次**（`run_all_experiments.py`）：

- `Semaphore(concurrency)` 限制并发为 2（1414 行附近）
- 每批 `scale_decision_interval=25` 个请求（1418 行）
- `asyncio.gather` 会同时提交一批协程，但实际并发受 semaphore 限制（1458-1462 行）

**transformers 推理**（`_sync_generate_transformers`，834-867 行）：

- 每次推理在 `torch.inference_mode()` 中调用 `model.generate()`
- 推理后显式 `del`、`gc.collect()`、`torch.cuda.empty_cache()`（912-916 行）
- 若 LoRA 加载失败，会抛出异常而非 OOM；因此「已杀死」更可能是内存/显存超限导致 OOM killer 杀进程

### 3.3 可能的杀进程原因

1. **显存 OOM**  
   - 基座 6GB + 多 LoRA + KV cache 在 24GB 3090 上接近上限  
   - LoRA 切换时 delete/load 的瞬时峰值叠加  
   - 若 warmup 失败，首个请求需在无预加载情况下加载 LoRA，瞬时占用更高  

2. **系统 RAM OOM**  
   - 数据集、traces、asyncio 队列等占用  
   - PyTorch 在 CPU 上的缓冲和临时分配  

3. **weights_only 失败后的异常处理**  
   - warmup 中 LoRA 加载失败被 `except` 捕获，继续执行（1401-1402 行）  
   - 请求阶段若同样失败，可能触发未完全处理的异常路径，间接导致内存/状态异常  
   - 但更直接的仍是 OOM

---

## 四、修复建议（按优先级）

### 4.1 修复 weights_only（必须）

1. **同时 patch `torch.load`**  
   - 在 `run_all_experiments.py` 的 patch 逻辑中增加：  
     `torch.load = _faaslora_torch_load`  
   - 确保通过 `torch.load` 的调用也走我们的 wrapper  

2. **或确保适配器为 safetensors 格式**  
   - 安装 `safetensors`：`pip install safetensors`  
   - 重新生成适配器，使 `generate_synthetic_lora` 成功创建 `adapter_model.safetensors`  
   - PEFT 会优先用 `safe_load_file`，不再调用 `torch.load`  

### 4.2 降低显存与内存压力（缓解 OOM）

- 将 `max_instances` 设为 1（已改）
- 将 `warm_pool_size` 设为 2（已改）
- 将 `gpu_warmup_hotness` 提高到 0.9，减少预热 LoRA 数量（已改）
- 将 `max_output_tokens_cap` 设为 64（已改）
- 将 `concurrency` 降到 2（已改）
- 如仍 OOM，可继续降到 `concurrency=1`、`warm_pool_size=1` 测试

### 4.3 子进程与 sitecustomize

- `run_cold_start_subprocess` 通过 `subprocess.run(..., env={**os.environ, ...})` 启动，会继承环境变量  
- `sitecustomize` 仅在 Python 启动时从 site-packages 等路径加载，不是通过 `PYTHONPATH` 中的普通目录  
- 子进程如需 patch，应确保 `PYTHONPATH` 包含 `REPO_ROOT`，并在子进程入口显式执行 patch（或 `import sitecustomize`），否则 `sitecustomize` 可能不生效

---

## 五、总结

| 问题 | 根因 | 修复方向 |
|------|------|----------|
| weights_only 报错 | 1. PEFT 显式传 `weights_only=True`，环境变量无法覆盖<br>2. 仅 patch 了 `torch.serialization.load`，未 patch `torch.load`<br>3. 适配器为 .bin 格式，走 `torch.load` | 同时 patch `torch.load`，或改用 safetensors |
| 进程被杀死 | OOM（显存或系统内存） | 降低并发、预热数量、输出长度等参数；先修好 weights_only，再观察是否仍 OOM |
