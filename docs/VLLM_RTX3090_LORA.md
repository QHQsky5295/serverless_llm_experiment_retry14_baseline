# vLLM + LoRA 在 RTX 3090 上 "EngineCore died unexpectedly" 的解决方案

## 当前环境检查（LLM）

在 **LLM** 环境中执行 `pip show vllm torch` 与 `python -c "import torch; print(torch.__version__, torch.version.cuda)"` 可得到：

| 组件   | 当前版本（示例） | 说明 |
|--------|------------------|------|
| Python | 3.12.x           | 满足 vLLM 0.10/0.16 要求 |
| PyTorch| 2.9.1+cu128      | 为 CUDA 12.8 构建，多用于新架构（如 Blackwell） |
| vLLM   | 0.16.0 或 0.10.2 | 项目推荐 **0.10.2**（见下方） |
| CUDA   | 随 PyTorch（cu128/cu126 等） | 3090 为 Ampere，cu121/cu124 更常见 |

**是否可用于完备实验（FaaSLoRA + vLLM LoRA）**  
- **vLLM 0.16 + PyTorch 2.9+cu128**：在 RTX 3090 上多次出现「首次 LoRA 请求后 EngineCore 子进程退出」，**不建议**用于当前完备实验。  
- **vLLM 0.10.2 + PyTorch 2.4~2.5（cu121/cu124）**：社区反馈 0.10.x 在消费级 GPU 上 LoRA 更稳定，**推荐**用于 3090 完备实验。  
- 若暂时无法稳定 vLLM LoRA，可在 `configs/experiments.yaml` 中使用 `backend: "transformers"` 跑通实验。

**推荐组合（RTX 3090 + LoRA）**  
- vLLM **0.10.2**（`pyproject.toml` 已固定；0.10.0 需 Python &lt;3.13）  
- PyTorch **2.4.x**，CUDA **12.4**（cu124），如：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`  
- 安装 vLLM：`pip install vllm==0.10.2`  

**一键重装**：使用脚本 `bash scripts/reinstall_env.sh` 可删除并重建 **LLM** 环境，按上述组合安装。详见 [docs/ENVIRONMENT.md](ENVIRONMENT.md)。

---

本实验使用 vLLM 时，若在 **首次带 LoRA 的请求** 后出现：

```text
ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
EngineDeadError: EngineCore encountered an issue.
```

可依次尝试以下**社区验证过的做法**（均来自 vLLM 官方 Issue / 论坛）：

---

## 1. 增大 /dev/shm（共享内存）【优先尝试】

**原因**：EngineCore 子进程与主进程通过共享内存通信，`/dev/shm` 过小会导致崩溃。

**做法**：

- **本机**（需 root）：
  ```bash
  sudo mount -o remount,size=1G /dev/shm
  df -h /dev/shm   # 确认 >= 512MB，建议 1GB
  ```
- **Docker**：
  ```bash
  docker run --shm-size=1g ...
  ```
- 若为 Kubernetes，请调大 Pod 的 `emptyDir` 或对应 volume 的 size。

**参考**： [vLLM Issue #27557](https://github.com/vllm-project/vllm/issues/27557) — 有用户将 `/dev/shm` 从 64MB 增至 200MB+ 后问题消失。

---

## 2. 降级 vLLM 到 0.10.0

**原因**：部分用户反馈 vLLM > 0.10.0 后出现 EngineCore 异常退出，0.10.0 可正常跑 LoRA。

**做法**：

```bash
pip install vllm==0.10.0
```

然后重新运行实验。注意 0.10.0 的 API/参数可能与 0.16 略有差异，若脚本报错再针对性改。

**参考**： [vLLM Issue #23517](https://github.com/vllm-project/vllm/issues/23517) — "it works with 0.10.0"。

---

## 3. 确保引擎在 main 入口内创建

**原因**：若在「模块加载时」就创建 vLLM 引擎，进程退出顺序可能导致 EngineCore 异常退出；在 `if __name__ == "__main__":` 调用的 `main()` 里创建则更稳定。

**做法**：本仓库的 `run_all_experiments.py` 已满足该条件（引擎在 `main_async()` 中创建，并由 `asyncio.run(main_async(...))` 在 main 中调用）。若自行写脚本，请将 `LLM(...)` / 引擎初始化放在 `def main():` 内，并在 `if __name__ == "__main__": main()` 中调用。

**参考**： [vLLM Issue #23517](https://github.com/vllm-project/vllm/issues/23517#issuecomment-3261567306)、[#27557](https://github.com/vllm-project/vllm/issues/27557)。

---

## 4. 已在本项目中启用的 vLLM 相关设置

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`（避免 KV cache 等多进程初始化崩溃）
- `VLLM_DISABLE_LORA_STREAM=1`（Punica 单流，减少 3090 上 LoRA 崩溃）
- 引擎创建失败时自动重试：先关闭 chunked prefill / prefix caching，再降低显存占用
- 启动时检查 `/dev/shm` 可用空间，不足会打印警告

若仍崩溃，请先做 **1（增大 /dev/shm）** 和 **2（降级 0.10.0）**；仍不行再考虑在 `configs/experiments.yaml` 中临时改用 `backend: "transformers"` 跑通实验。
