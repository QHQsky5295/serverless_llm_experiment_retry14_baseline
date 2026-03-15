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

## 0. 当前仓库在 14B TP=2 上的新增结论（2026-03-13）

这部分不是泛泛而谈的社区经验，而是本仓库在当前机器上已经实际复现并确认过的结论：

- 如果你看到的不是经典的 `EngineCore died unexpectedly`，而是：
  - `ray_distributed_executor`
  - `The current node timed out during startup`
  - `Failed to get node info`
  这时优先怀疑的根因不是模型权重缺失，而是 **TP=2 场景下可见 GPU 被错误缩成了单卡**。

- 本仓库已经修复：
  - `scripts/run_all_experiments.py`
    - `TP>1` 时优先使用 `model.visible_device_ids`
    - 本机双卡可见时显式优先 `distributed_executor_backend="mp"`
  - 否则 vLLM 会误判“本机可见 GPU 数 < TP world size”，自动切到 Ray，再在单节点 Ray bring-up 阶段超时。

- 对当前 `Qwen2.5-14B-Instruct + TP=2 + 100 adapters` 来说：
  - `gpu_memory_utilization=0.90` 会让 serving 阶段长期处在约 `22.3/24.0 GB (93%)`
  - ResidencyManager 会周期性报 `Memory pressure detected`
  - GPU resident LoRA 容易被反复驱逐

- 当前主线默认已经收敛到：
  - `distributed_executor_backend=mp`
  - `gpu_memory_utilization=0.85`

- 在这组新默认下，quick 复测中：
  - GPU 常驻约 `19.8/24.0 GB (83%)`
  - 周期性 memory pressure 告警消失
  - GPU resident adapter 数可以逐步升到 `3`
  - 14B 已能稳定进入真实 serving，而不再卡死在 bring-up

- 在 `gpu_memory_utilization=0.80` 下，完整 `representative 1000 requests` 已跑通：
  - `1000/1000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 603 / 1006 / 1174 ms`
  - `E2E avg/p95/p99 = 13.10 / 14.91 / 15.22 s`
  - `TPOT avg = 99.3 ms`
  - `cache hit rate = 85.2%`
  - `warm_pool_hits = 806`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 运行期间未再出现 `Memory pressure detected`、Ray startup timeout 或 EngineCore 异常退出

- 在 `gpu_memory_utilization=0.85` 下，完整 `representative 1000 requests` A/B 也已跑通：
  - `1000/1000` 完成，`fail=0`
  - GPU 常驻约 `21.1/24.0 GB (88%)`
  - 仍未出现 `Memory pressure detected`、Ray startup timeout 或 EngineCore 异常退出
  - 相比 `0.80`：
    - `TTFT avg` 略降约 `0.7%`
    - `TTFT p99` 略降约 `1.1%`
    - `E2E avg` 降约 `1.5%`
    - `E2E p95/p99` 降约 `8.5%`
    - `TPOT avg` 降约 `1.7%`
    - `RPS` 升约 `1.5%`
    - 仅 `TTFT p95` 小幅上升约 `1.5%`
  - 这说明 `0.85` 是当前机器上更优的稳定参数组合，后续可以直接用它进入 `representative 4000 requests`

- 在 `gpu_memory_utilization=0.85` 下，后续的 `representative 4000 requests` 也已跑通：
  - `4000/4000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 588 / 1013 / 1133 ms`
  - `E2E avg/p95/p99 = 13.15 / 14.90 / 15.16 s`
  - `TPOT avg = 99.6 ms`
  - `RPS = 0.1475`
  - `cache hit rate = 86.05%`
  - `warm_pool_hits = 3341`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 因此 `0.85` 现在可以视为当前 14B 路径的冻结默认参数

- 补做的 `Qwen2.5-7B-Instruct TP=2` 单实例双卡对照也已确认稳定：
  - `1000/1000` 完成，`fail=0`
  - 相比 `7B TP=1`：
    - `RPS` 约 `+49.2%`
    - `E2E avg/p95/p99` 改善约 `30.8% / 38.1% / 14.6%`
    - `TPOT avg` 改善约 `43.2%`
    - 但 `TTFT avg` 变差约 `37.7%`
    - `cache hit rate` 与 `warm_pool_hits` 均下降
  - 因此 `7B TP=2` 更适合作为吞吐导向对照，不建议替换当前 `7B TP=1` 默认模式

- 当前机器最近多次出现登录会话中途转成 `closing`，所以 14B 长实验应优先通过：
  - `scripts/run_all_experiments_user_scope.sh`
  启动，不要直接把长实验绑在当前 SSH/TTY 的 session scope 里。

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
