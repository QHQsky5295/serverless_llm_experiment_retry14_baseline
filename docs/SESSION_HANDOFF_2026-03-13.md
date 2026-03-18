# FaaSLoRA 会话交接文档（2026-03-13）

## 1. 文档用途

这是一份给“新会话”直接续接使用的项目交接文档。目标不是替代 README 或 PROJECT_PROGRESS，而是把当前这轮协作里已经形成的：

- 主线状态
- 已验证结论
- 当前默认配置
- 未提交改动
- 运行风险
- 交互习惯

一次性写清楚，避免新会话重新摸索。

---

## 2. 项目一句话定位

项目名称固定为：

**FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知 Serverless 系统**

当前项目已经不是“能不能跑”的阶段，而是：

- 真实模型
- 真实 LoRA
- 真实 Azure trace 驱动
- 单节点、双 RTX 3090 24GB
- 面向系统实验与交付整理

的研究原型。

---

## 3. 当前硬件与稳定环境

### 硬件

- 单节点
- 2 × RTX 3090 24GB

### 当前稳定环境

- Conda 环境：`LLM_vllm0102`
- Python：`3.12`
- Torch：`2.8.0+cu128`
- vLLM：`0.10.2`
- Transformers：`4.57.6`

### 环境补充

已安装并接入：

- `FlashInfer`
- `torch-c-dlpack-ext`

当前主线实验应统一使用：

- `/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`

不要把 `LLM` 和 `LLM_vllm0102` 混作同一套主线环境。

---

## 4. 当前主线规划

### 4.1 模型家族主线

当前扩展顺序是：

1. 先完成 Qwen 家族
2. 当前正在推进：`Qwen2.5-14B-Instruct`
3. Qwen 之后的下一个家族固定为 **Mistral**
4. `OPT` 已确认不支持当前本机 `vLLM 0.10.2 + LoRA`；当前小档位固定为 `mistralai/Mistral-7B-Instruct-v0.3`，大档位固定为 `mistralai/Mistral-Nemo-Instruct-2407`
5. Gemma 暂不进入当前配置与实验轮次，但继续保留在计划列表中

### 4.2 数据集主线

当前先不接新数据集。

顺序是：

1. 先做 `ShareGPT prompt pool + Azure representative trace`
2. 当前优先把 `Qwen2.5-14B-Instruct` 跑通
3. 新数据集（例如 `GSM8K`）放到 `14B + ShareGPT + representative 4000` 之后再接

### 4.3 保留但不作为当前主线的接口

以下内容实现和接口都保留，但**不作为当前主线推进对象**：

- `shared`
- `dedicated`
- `28185 full trace`

`effective_capacity_admission_enabled`（P2.5）接口仍保留用于 on/off 对照；当前仓库主线默认配置已切到开启状态。不同 backbone 的差异主要体现在收益幅度与最终冻结结论上，而不是默认开关口径本身。

---

## 5. 已完成的重要结论

### 5.1 Qwen 3B

已验证并形成稳定结论：

- `Qwen2.5-3B-Instruct`
- `auto`
- `500 LoRA`
- `representative 1000 requests`
- 稳定 serving 参数：
  - `concurrency=8`
  - `runtime_concurrency_cap=8`
  - `max_num_seqs=8`
  - `max_loras=8`
  - `max_num_batched_tokens=4096`
  - `max_model_len=2048`

关于 P2.5：

- 3B 上开启 P2.5 **没有显著收益**
- 但也没有严重回归
- 因此 3B 这条线里，P2.5 更像“统一口径项”，不是主要性能来源

### 5.2 Qwen 7B

已验证并形成稳定结论：

- `Qwen2.5-7B-Instruct`
- `auto`
- `100 adapters`
- `1000 requests`
- `P2.5 on`

7B 上的明确结论：

- P2.5 是**有效的**
- 修复真实观测口径并补上 FlashInfer 后，P2.5 显著降低了 `contention` 和 `defer`
- 因此 7B 当前确认版应视为 **P2.5 on**

### 5.3 Qwen 14B（bring-up 最新状态）

当前已经形成的最新结论是：

- `Qwen2.5-14B-Instruct`
- `tensor_parallel_size=2`
- `100 adapters`
- 当前默认 profile 已改为：
  - `distributed_executor_backend=mp`
  - `gpu_memory_utilization=0.85`

这轮 bring-up 中确认过两个关键事实：

- 最初 14B 启动失败的根因**不是**模型缺文件，也**不是**直接 OOM
- 真正根因是旧代码在 `TP=2` 路径里把 `CUDA_VISIBLE_DEVICES` 临时缩成了单卡 `0`
- 这会让 vLLM 误判“本机可见 GPU 数小于 TP world size”，自动切到 `ray_distributed_executor`
- 随后在单节点本机 Ray 初始化阶段报：
  - `The current node timed out during startup`

现在已经修复：

- `scripts/run_all_experiments.py`
  - `TP>1` 时优先使用 `visible_device_ids`
  - 本机双卡足够时显式优先 `distributed_executor_backend=mp`

修复后的实际复测结论：

- 通过 `run_all_experiments_user_scope.sh` 启动的 14B quick 已确认可以稳定完成：
  - vLLM engine bring-up
  - TP=2 worker 初始化
  - 权重加载
  - KV cache 初始化
  - 真实 serving 进入 Phase 2
- `gpu_memory_utilization=0.90` 时，GPU 常驻约 `22.3/24.0 GB (93%)`
  - ResidencyManager 会持续报 `Memory pressure detected`
  - LoRA 在 GPU tier 中容易被反复驱逐
- 把 `gpu_memory_utilization` 调到 `0.80` 后：
  - quick 复测时 GPU 常驻约 `19.8/24.0 GB (83%)`
  - 周期性 memory pressure 告警消失
  - GPU resident adapter 数能逐步升到 `3`
  - quick 前 6 个请求的推进速度相较 `0.90` 有可见改善
- 随后的完整 `representative 1000 requests` 长跑已经完成：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_p25_on.json`
  - `1000/1000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 603 / 1006 / 1174 ms`
  - `E2E avg/p95/p99 = 13.10 / 14.91 / 15.22 s`
  - `TPOT avg = 99.3 ms`
  - `RPS = 0.148`
  - `cache hit rate = 85.2%`
  - `warm_pool_hits = 806`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 运行期间未再出现 `Memory pressure detected`、Ray startup timeout 或 EngineCore 异常退出
- 在此基础上继续补做了 `gpu_memory_utilization=0.85` 的完整 `r1000` A/B：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_u085_p25_on.json`
  - `1000/1000` 完成，`fail=0`
  - GPU 常驻约 `21.1/24.0 GB (88%)`
  - 仍未出现 `Memory pressure detected`、Ray timeout 或 EngineCore 异常退出
  - 相比 `0.80`：
    - `TTFT avg` 略降约 `0.7%`
    - `TTFT p99` 略降约 `1.1%`
    - `E2E avg` 降约 `1.5%`
    - `E2E p95/p99` 降约 `8.5%`
    - `TPOT avg` 降约 `1.7%`
    - `RPS` 升约 `1.5%`
    - 仅 `TTFT p95` 小幅上升约 `1.5%`
  - 综合判断 `0.85` 是当前 `14B` 更优的稳定参数组合

当前判断：

- **14B 已经从“启动阻塞”进入“最优稳定参数已收敛完成”阶段**
- 当前 `gpu_memory_utilization=0.85` 可作为 `14B` 的新默认稳定参数
- 后续的 `representative 4000 requests` 也已经完成：
  - 结果文件：
    - `results/experiment_results_full_vllm_auto_a100_r4000_c2_faaslora_full_qwen14b_tp2_r4000_u085_p25_on.json`
  - `4000/4000` 完成，`fail=0`
  - `TTFT avg/p95/p99 = 588 / 1013 / 1133 ms`
  - `E2E avg/p95/p99 = 13.15 / 14.90 / 15.16 s`
  - `TPOT avg = 99.6 ms`
  - `RPS = 0.1475`
  - `cache hit rate = 86.05%`
  - `warm_pool_hits = 3341`
  - `contention_events = 0`
  - `avg_defer_ms = 0`
  - 运行期间未再出现 `Memory pressure`、Ray timeout、OOM 或 Traceback
- 因此 `0.85` 现在可以视为 **14B 的冻结默认参数**

### 5.4 Qwen 7B TP=2 对照最新状态

在 `Qwen2.5-7B-Instruct` 上，新增的单实例双卡对照 profile 已经完成：

- model profile：`qwen_7b_tp2_compare`
- workload profile：`qwen_7b_tp2_compare_main`
- 参数形态：
  - `tensor_parallel_size=2`
  - `distributed_executor_backend=mp`
  - `max_instances=1`
  - `representative 1000 requests`

结果文件：

- TP=1 基线：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_auto_r1000_p25_on.json`
- TP=2 对照：
  - `results/experiment_results_full_vllm_auto_a100_r1000_c4_faaslora_full_qwen7b_tp2_compare_r1000_p25_on.json`

对比结论：

- 两轮都完成 `1000/1000`，且 `fail=0`
- TP=2 相比 TP=1：
  - `RPS` 约 `+49.2%`
  - `E2E avg/p95/p99` 分别改善约 `30.8% / 38.1% / 14.6%`
  - `TPOT avg` 改善约 `43.2%`
  - `P95 TTFT` 改善约 `36.4%`
  - 但 `TTFT avg` 变差约 `37.7%`
  - `P99 TTFT` 变差约 `5.9%`
  - `cache hit rate` 从 `94.6%` 降到 `85.2%`
  - `warm_pool_hits` 从 `896` 降到 `801`
- 两轮都没有 `contention` / `defer` / `Memory pressure` 回归

当前判断：

- 若目标是吞吐、平均 E2E 与 TPOT，`7B TP=2` 明显更强
- 若目标是平均 TTFT、cache hit 和更贴近单卡副本扩缩容的现实路径，`7B TP=1` 仍更适合作为默认主线
- 因此当前建议是：
  - 保持 `7B TP=1` 为默认模式
  - 保留 `7B TP=2` 作为正式对照 profile，不替换默认模式

---

## 6. 这轮代码与配置已做的关键改动

### 6.1 数据源选择已接入 YAML

之前的问题：

- `model.name` 和 `workload.total_requests` 本来就在 YAML
- 但数据源选择还被代码部分写死

现在已经接好：

- `datasets.arrival_source`
- `datasets.token_source`
- `datasets.prompt_source`
- `datasets.azure_max_records`
- `datasets.sharegpt_max_records`

代码位置：

- `configs/experiments.yaml`
- `scripts/run_all_experiments.py`
- `faaslora/datasets/dataset_loader.py`

### 6.2 主线切换已收敛到 profile_selection

现在 `experiments.yaml` 已支持：

- `profile_selection.model`
- `profile_selection.dataset`
- `profile_selection.workload`

并新增：

- `model_profiles`
- `dataset_profiles`
- `workload_profiles`

目的：

- 不再通过手改一堆散落字段切换实验
- 可以更稳定地在 `3B / 7B / 14B` 与 `1000 / 4000` 之间切换

### 6.3 新增的关键配置开关

- `lora_adapters.apply_scale_preset`

作用：

- 控制是否按 adapter 数量自动套用旧的 `scale_presets`
- 对 `Qwen2.5-14B-Instruct` 这类新模型很重要
- 否则旧的 `100/500` preset 会把 3B/7B 的参数错误套到 14B 上

### 6.4 14B TP=2 bring-up 修复

这轮还新增了一个关键修复：

- `scripts/run_all_experiments.py`
  - 修复了 `TP>1` 时错误把 `CUDA_VISIBLE_DEVICES` 缩成单卡的问题
  - 增加了对 `model.visible_device_ids` 和 `model.distributed_executor_backend` 的显式处理
  - 当前单节点双卡 14B 主线会优先使用 `mp`，不再误走 Ray

- `scripts/run_all_experiments_user_scope.sh`
  - 现在会显式透传 `VLLM_*` 和 `PYTHONUNBUFFERED`
  - 与交接文档里的 user-scope 启动命令保持一致

### 6.5 Mistral 主线准备

为了让下一步切到 Mistral 时不再临时拼配置，这轮又补了两项准备：

- `configs/experiments.yaml`
  - 已新增 `mistral_7b_main / mistral_nemo_12b_tp2` model profile
  - 已新增 `mistral_7b_auto100_main / mistral_7b_auto500_main / mistral_nemo_12b_tp2_bringup100_main / mistral_nemo_12b_tp2_main` workload profile
- `scripts/download_model.py`
  - 已不再尝试按旧版单模型 YAML 结构自动改写 `experiments.yaml`
  - 当前行为改为：下载完成后打印推荐的 `FAASLORA_PROFILE_MODEL / WORKLOAD` 与本地模型路径
  - 否则旧逻辑会误改到 `experiment.name` 这类错误位置

### 6.6 论文主线 LoRA 工件默认已切到 PEFT+finetune

当前主线口径已更新为：

- `configs/experiments.yaml -> lora_adapters.generation_mode = peft_finetune`
- `generate_synthetic = false`
- `scripts/generate_lora_adapters.py` 默认跟随 YAML，直接生成 `PEFT+finetune` 工件
- 其默认值现在会跟随当前激活的 `profile_selection + model_profiles + workload_profiles` 解析，不再只读取顶层 `model.name`
- `PEFT+finetune` 生成路径已改为单次加载 base model 后循环生成多个 adapters
- `scripts/run_all_experiments.py` 现在支持由 `lora_adapters.preparation_mode` 控制工件准备流程：
  - `one_shot`：正式实验前自动补齐缺失/不兼容工件
  - `two_phase`：强制先手动执行 `scripts/generate_lora_adapters.py`，再启动正式实验
- `synthetic` 仍保留，但仅作为 quick/debug 回退路径，不再作为论文主线默认
- 当前 Mistral 7B 主线统一按 `PEFT+finetune + 500 adapters + representative r1000` 推进；`100 adapters` 仅保留给早期 bring-up / 快速验证口径
- 当前 Mistral Nemo 主线也统一按 `TP=2 + PEFT+finetune + 500 adapters + representative r1000` 推进；`mistral_nemo_12b_tp2_bringup100_main` 仅保留给显式 bring-up / 快速排障

补充说明：

- 当前正在运行中的旧 `Mistral-7B` 生成进程仍按旧逻辑继续，不应中途打断
- 下一次重新启动生成器时，会自动获得“单次加载 base model”的新实现

---

## 7. 当前 experiments.yaml 的默认状态

当前默认已经切到：

```yaml
profile_selection:
  model: "qwen_14b_tp2"
  dataset: "azure_sharegpt_rep1000"
  workload: "qwen_14b_tp2_main"
```

这对应的实际组合是：

- 模型：`Qwen2.5-14B-Instruct`
- 张量并行：`tensor_parallel_size=2`
- distributed executor：`mp`
- 数据集：`Azure representative 1000 requests + ShareGPT prompt pool`
- adapter 数：`100`
- `concurrency=2`
- `max_instances=1`
- `P2.5 on`
- `gpu_memory_utilization=0.85`
- `apply_scale_preset=false`

### 为什么 14B 是 max_instances=1

因为当前 14B 配置是：

- `tensor_parallel_size=2`

这意味着：

- 一个 runtime 就会占满两张 3090
- 当前机器也只有两张 3090

因此：

- 物理上不可能再扩出第二个实例

所以当前 14B 这一步的定位是：

- **单实例 TP=2 的可运行性与性能验证**

不是：

- 双实例 auto 扩容验证

---

## 8. 当前 14B 模型下载状态

本地模型目录已经完整：

- `/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-14B-Instruct`

已确认存在：

- `config.json`
- `tokenizer_config.json`
- `generation_config.json`
- `8` 个 `*.safetensors` 分片

所以从“模型文件是否齐全”这个角度看，**14B 可以直接开始实验**。

---

## 9. 当前未提交工作区状态

当前 `HEAD` 为：

- `96a393f5b4c7e2b09ac2c41190a0310d1bda5990`

当前本地有未提交改动：

- `EXPERIMENT_GUIDE.md`
- `README.md`
- `configs/experiments.yaml`
- `configs/generated/lora_manifest_1000.json`
- `docs/PROJECT_PROGRESS.md`
- `docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`
- `faaslora/datasets/dataset_loader.py`
- `scripts/run_all_experiments.py`
- `tests/test_basic_smoke.py`

其中：

- `configs/generated/lora_manifest_1000.json` 是运行期文件，不应默认提交
- 其他文件是这轮本地代码 / 文档改动，**尚未同步到 GitHub**

---

## 10. 已做的验证

这轮本地改动已经通过：

- YAML 解析检查
- `pyflakes`
- `python3 -m compileall`
- `tests.test_basic_smoke`

最近一次 smoke 结果是：

- `25 tests, OK`

---

## 11. 我们已经形成的交互习惯

这一部分非常重要，新会话应尽量延续。

### 11.1 语言与风格

- 默认使用中文
- 回答要清楚、直接、严谨
- 不要把没验证的东西当成已经验证的结论

### 11.2 “继续”的含义

如果用户说：

- `继续`

含义是：

- 不要重新问“接下来做什么”
- 直接沿着上一条回复最后给出的 **之后步骤** 往下推进

### 11.3 每次执行完任务后的回复格式

每次完成任何一步后，回复里都要带：

- `当前步骤位置：...`
- `之后步骤：...`

这样用户可以随时知道我们在整个 TODO 中走到哪一步了。

### 11.4 文档更新习惯

用户对文档有明确要求：

- 要在已有文档基础上**迭代修改**
- 不要无缘无故整篇重写
- 只有像本文件这种“专门交接文档”才适合新建

### 11.5 GitHub 同步习惯

如果之后要同步 GitHub，必须：

- 同步代码
- 同步 README / GUIDE / docs
- 保证文档与当前实现一致
- 不要只推代码不推文档

### 11.6 主线管理习惯

- 当前主线只推进一个主要目标
- 不要在没有新证据前来回切换主线
- `shared / dedicated / full-trace` 接口保留，但不再拉回主线

### 11.7 解释参数时的习惯

对于：

- `concurrency`
- `runtime_concurrency_cap`
- `max_num_seqs`
- `max_num_batched_tokens`

必须区分层次解释，不能把它们说成同一个“并发”。

---

## 12. 当前会话层面的已知问题：closing session

最近的主要启动阻塞不是代码错误，而是：

- 当前某些 Remote-SSH / 终端会话会进入 `State=closing`

已经确认过：

- 存在旧的 `Remote=yes, Type=tty, State=closing` 会话
- 也存在新的 `State=active` 会话
- 当前用户：
  - `Linger=no`

### 12.1 这意味着什么

`State=closing` 不是异常崩溃，而是：

- 一个旧登录会话正在退出

脚本会主动拦下这种状态，避免长实验跑到一半被 systemd 杀掉。

### 12.2 不能做什么

- 不能把一个已经 `closing` 的会话恢复成 `active`

### 12.3 正确做法

1. 新开一个 Remote-SSH 窗口或新的 TTY
2. 先检查：

```bash
echo $XDG_SESSION_ID
loginctl show-session "$XDG_SESSION_ID" -p State -p Active
```

3. 只在 `State=active` 的会话里跑长实验

### 12.4 更稳的做法

优先用：

```bash
scripts/run_all_experiments_user_scope.sh
```

把实验挪到新的 user scope。

### 12.5 可进一步优化的做法

建议开启：

```bash
sudo loginctl enable-linger qhq
```

这样 user manager 会更稳定，user-scope 任务也更不容易跟着登录会话一起消失。

---

## 13. 当前应该怎么继续

### 13.1 推荐启动命令（Qwen 14B，1000 requests，优先 user-scope）

当前 YAML 已经切到 14B 默认 profile，且 `TP=2` 的 `visible_device_ids + mp` 修复已经落地。

这台机器最近多次出现登录会话在实验中途转成 `closing`，所以当前更推荐直接用 user-scope 启动：

```bash
cd /home/qhq/serverless_llm_experiment
FAASLORA_PYTHON=/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python \
PYTHONUNBUFFERED=1 \
VLLM_NO_USAGE_STATS=1 \
FAASLORA_LOG_TAG=qwen14b_tp2_r1000_p25_on \
FAASLORA_RESULTS_TAG=qwen14b_tp2_r1000_p25_on \
bash scripts/run_all_experiments_user_scope.sh \
  --config configs/experiments.yaml \
  --scenario faaslora_full 2>&1 | tee /tmp/qwen14b_tp2_r1000_p25_on.launch.log
```

### 13.2 如果你已经明确确认当前窗口仍是 active

只在你已经确认当前会话仍是 `State=active` 时，才直接跑：

```bash
cd /home/qhq/serverless_llm_experiment
PYTHONUNBUFFERED=1 \
VLLM_NO_USAGE_STATS=1 \
FAASLORA_LOG_TAG=qwen14b_tp2_r1000_p25_on \
FAASLORA_RESULTS_TAG=qwen14b_tp2_r1000_p25_on \
/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python \
scripts/run_all_experiments.py --config configs/experiments.yaml --scenario faaslora_full 2>&1 | tee /tmp/qwen14b_tp2_r1000_p25_on.log
```

### 13.3 继续当前主线时的实际建议

当前推荐顺序是：

1. 当前 `gpu_memory_utilization=0.85` 已确定为 `Qwen2.5-14B-Instruct` 的更优稳定参数
2. 先读取并分析：
   - `/tmp/qwen14b_tp2_r1000_p25_on.launch.log`
   - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_p25_on.json`
   - `/tmp/qwen14b_tp2_r1000_u085_p25_on.launch.log`
   - `results/experiment_results_full_vllm_auto_a100_r1000_c2_faaslora_full_qwen14b_tp2_r1000_u085_p25_on.json`
3. 以 `0.80` 为稳定基线，以 `0.85` 为当前更优稳定组合
4. 当前 `14B r4000 @ 0.85` 已完成，`0.85` 可以视为 14B 的正式冻结参数
5. 当前 `Qwen2.5-7B-Instruct TP=2` 对照也已完成，并已确定“保留 TP=1 为默认、TP=2 为对照”的结论
6. 已确认 `facebook/opt-6.7b` 在当前 `vLLM 0.10.2 + LoRA` 环境下不可用，应停止 OPT 路线
7. 已完成：`mistralai/Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000`
8. 论文正式对比现在改为“模型专属冻结工件目录 + two_phase 预生成”工作流：同一 base model 先单独建好冻结工件池，后续严格复用，避免不同系统/不同超参数实验使用了不同 LoRA。
9. `V2` 路线已经改口径：不再直接沿用旧 `realistic_v2` 目录，而是采用 `publicmix` 建库方案：
   - `Qwen 7B / Mistral 7B`：尽量下载公开 adapter，先做本地兼容性验证后再纳入正式工件池
   - `Qwen 14B / Mistral-Nemo`：先下载现有公开 adapter，再按统一规则补齐到 `500`
10. `V1` 冻结工件保留不动；旧 `V2` 目录已删除，等待按新的 `publicmix` 规则重建。
11. 当前已经新增 `scripts/prepare_publicmix_pool.py`，可先对本地下载的公开 adapter 做兼容性验收（`validate`），再生成 formal `V2 publicmix` 清单（`plan`）；后续建库请优先使用该脚本，而不是手工挑目录。
12. 当前 `V2 publicmix` 第一阶段已完成：`Qwen 7B / Qwen 14B / Mistral 7B / Mistral-Nemo` 的 validation report 与 manifest 均已生成；当前 accepted public 数分别为 `5 / 4 / 5 / 4`。
13. `scripts/prepare_publicmix_pool.py` 已新增 `build` 子命令，可按 manifest 将公开 adapter 复制进冻结目录，并仅对 `generated_fill` 缺口按 `topup_profile + seed` 调用生成器补齐。后续正式 `V2` 建库请走 `validate -> plan -> build`，不要手工混拷目录。

---

## 14. 新会话里建议的第一句话

建议在新会话直接贴这份文档，或至少贴出下面这段摘要：

> 继续当前主线。`Qwen2.5-14B-Instruct` 的 `r1000@0.80`、`r1000@0.85` 与 `r4000@0.85` 都已完成，`0.85` 已可视为 14B 的冻结默认参数；`Qwen2.5-7B-Instruct TP=2` 对照也已完成，结论是保留 `TP=1` 为默认、`TP=2` 作为吞吐导向对照。`OPT` 已确认不支持当前本机 `vLLM 0.10.2 + LoRA`；`mistralai/Mistral-7B-Instruct-v0.3` 的论文主线 `PEFT+finetune + 500 adapters + representative r1000` 也已完成。当前论文正式对比默认改为“模型专属冻结工件目录 + two_phase 预生成”，并将新的 `V2` 路线收敛为 `publicmix`：`Qwen 7B / Mistral 7B` 优先纳入经本地兼容性验证的公开 adapter，`Qwen 14B / Mistral-Nemo` 采用公开 adapter + 统一规则补齐到 `500`。下一步请先完成这套 `V2` 建库规则，再切到 `mistralai/Mistral-Nemo-Instruct-2407` 的论文主线 `TP=2 + PEFT+finetune + 500 adapters + representative r1000`。

---

## 15. 新会话里的优先级顺序

1. 先确认当前 Remote-SSH/TTY 会话是否 `State=active`
2. 读取并分析 `Qwen2.5-14B-Instruct` 两轮 `representative 1000 requests` 的完整结果
3. 当前 `14B r4000 @ 0.85` 已完成，可直接把 `0.85` 视为冻结默认参数
4. 当前 `Qwen2.5-7B-Instruct TP=2` 对照已完成，结论是保留 `TP=1` 为默认、`TP=2` 为正式对照
5. 当前第二家族 7B 档 **mistralai/Mistral-7B-Instruct-v0.3** 已完成，可直接作为第二家族小档位基线
6. 下一步推进 **mistralai/Mistral-Nemo-Instruct-2407**，并按论文主线直接使用 `TP=2 + PEFT+finetune + 500 adapters + representative r1000`
7. Gemma 继续挂在计划列表，不在当前轮次启动

补充说明：

- `mistralai/Mistral-7B-Instruct-v0.3` 与 `mistralai/Mistral-Nemo-Instruct-2407` 都是公开可下载的 instruct 模型。

---

## 16. 这份文档的定位

如果新会话要继续推进，优先参考：

1. 本文件：会话级交接
2. [docs/PROJECT_PROGRESS.md](/home/qhq/serverless_llm_experiment/docs/PROJECT_PROGRESS.md)：项目进度主文档
3. [docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](/home/qhq/serverless_llm_experiment/docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md)：技术实现主文档
4. [EXPERIMENT_GUIDE.md](/home/qhq/serverless_llm_experiment/EXPERIMENT_GUIDE.md)：运行与实验说明

本文件强调的是：

- 当前会话上下文
- 当前默认配置
- 当前互动习惯
- 当前下一步

而不是替代整个项目文档体系。
