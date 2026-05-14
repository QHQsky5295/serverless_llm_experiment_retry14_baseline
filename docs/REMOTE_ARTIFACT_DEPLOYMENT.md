# PrimeLoRA 两节点 Remote 工件配置

本文档记录一个可选的两节点工件链路：GPU 推理节点只运行 PrimeLoRA 和
baseline harness，另一个节点作为 remote artifact node 保存 LoRA adapter
目录。请求命中 REMOTE 层时，本地节点通过 HTTP 从 remote 节点拉取 adapter
到本地 NVMe 缓存，再继续原有 HOST/GPU admission 流程。

## 关键约束

- 默认关闭：不设置 `FAASLORA_REMOTE_ARTIFACT_ENABLED=1` 时，所有正式实验仍使用
  `configs/experiments.yaml` 中的本地冻结目录。
- 不改变已有论文结果：该能力只用于部署真实性和后续演示，不重跑、不覆盖
  `paper_results/final_v2/`。
- 不写入密码：SSH 密码、HTTP token 只通过交互输入或运行期环境变量传入，不能
  写入仓库、配置文件或 systemd 样例。
- 不静默回退：开关打开后，若远端缺少某个 adapter，本地会将该 adapter 视为
  remote fetch 失败，而不是偷偷读本地冻结目录。
- 公平对比口径：remote artifact 实验表示“同一远端 backing store + 各系统自己的
  本地缓存/加载机制”。所有系统 cold/first-touch 都必须从 remote materialize；
  materialize 后允许按 vLLM/SGLang/S-LoRA/ServerlessLLM/PrimeLoRA 的原机制使用
  本地 cache、HOST/main-memory pool 或 GPU residency。不要把它误解成每个请求
  都强制从远端重复下载同一个 adapter。

## 远端机器

当前计划的 remote artifact node：

```text
ssh/jump host: 10.199.227.174
ssh port: 8122
ssh user: lab14
artifact HTTP host on the experiment LAN: 192.168.4.174
GPU node on the experiment LAN: 192.168.4.178
artifact HTTP ports:
  - 18080: Llama-3.2 3B
  - 18081: Llama-2 7B
  - 18082: Llama-2 13B
```

密码不记录在本文档中。部署时使用交互式 SSH/SCP 输入。

## 远端目录

建议在 remote 节点上使用如下目录。当前真实两节点 smoke 与正式
true-remote 队列使用的是 `lab14` 用户目录；如果后续换成独立数据盘，
可保持同样的目录结构迁移到 `/data`。

```bash
/home/lab14/primelora_remote_artifacts/llama2_7b_a500_v2_publicmix
/home/lab14/primelora_remote_artifacts/llama2_13b_a500_v2_publicmix
/home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope
```

每个目录下面直接放 adapter 子目录，例如：

```text
llama32_3b_a500_v1_modelscope/
  code_lora_0135/
    adapter_config.json
    adapter_model.safetensors
  finance_lora_0169/
    ...
```

可以用本地脚本分发工件。该脚本不保存密码，SSH 密码需要交互输入：
脚本使用 `rsync -aL`，会把本地冻结工件中的绝对符号链接解引用为普通文件。
这是必要的，因为 3B/7B 冻结 adapter 目录里的 tokenizer/config support
files 可能是指向 GPU 节点模型目录的绝对 symlink，直接搬到 remote 节点后会失效。

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
scripts/stage_remote_artifacts.sh \
  --source artifacts/frozen/llama32_3b_a500_v1_modelscope \
  --remote-user lab14 \
  --remote-host 10.199.227.174 \
  --remote-port 8122 \
  --remote-dir /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope
```

如果只想先复制少量 adapter 做 smoke test，可准备一个 adapter id 列表：

```bash
printf "code_lora_0135\n" > /tmp/primelora_remote_subset.txt
scripts/stage_remote_artifacts.sh \
  --source artifacts/frozen/llama32_3b_a500_v1_modelscope \
  --remote-user lab14 \
  --remote-host 10.199.227.174 \
  --remote-port 8122 \
  --remote-dir /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --adapter-list /tmp/primelora_remote_subset.txt
```

## 远端启动

在 remote 节点上：

```bash
cd /path/to/PrimeLoRA
python3 remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 0.0.0.0 \
  --port 18080
```

如果需要 token：

```bash
export PRIME_REMOTE_TOKEN='<runtime-token-not-committed>'
python3 remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 0.0.0.0 \
  --port 18080
```

本轮 true-remote 正式复查同时启动三个只读 artifact endpoints：

```bash
# Llama-3.2 3B
python3 ~/primelora_remote/remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 0.0.0.0 --port 18080

# Llama-2 7B
python3 ~/primelora_remote/remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama2_7b_a500_v2_publicmix \
  --host 0.0.0.0 --port 18081

# Llama-2 13B
python3 ~/primelora_remote/remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama2_13b_a500_v2_publicmix \
  --host 0.0.0.0 --port 18082
```

systemd 样例位于：

```text
remote_artifact_node/systemd/primelora-remote-artifacts.service.example
```

## 本地 Smoke Test

在 GPU 推理节点：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
python3 scripts/remote_artifact_client.py health
python3 scripts/remote_artifact_client.py list --limit 5
python3 scripts/remote_artifact_client.py smoke --dst-root /tmp/primelora_remote_fetch
```

正式 true-remote 队列使用的 endpoint 对应关系为：

```bash
export REMOTE_FAIR_REAL_ENDPOINT_LLAMA2_7B=http://192.168.4.174:18081
export REMOTE_FAIR_REAL_ENDPOINT_LLAMA2_13B=http://192.168.4.174:18082
export REMOTE_FAIR_REAL_ENDPOINT_LLAMA32_3B=http://192.168.4.174:18080
```

本地环境如果设置了 `ALL_PROXY`，client 默认会绕过环境代理，直接连接 LAN
remote 节点。当前 SSH 入口是 `10.199.227.174:8122`，但 artifact HTTP
实测应走 LAN 地址 `192.168.4.174:18080`。

2026-05-13 true-remote 复查踩坑：部分 frozen adapter 目录包含指向本机模型缓存
的绝对 symlink，例如 `/home/qhq/serverless_llm_experiment/models/.../config.json`。
真实 remote 节点没有该路径时，artifact server 不能用 `resolve(strict=True)`
直接解引用，否则会对该 adapter 返回 `Empty reply` 并污染正式 replay。当前
`remote_artifact_node/server.py` 的规则是：打包时跳过不可解析或越过 artifact
root 的非可移植 symlink，只发送 LoRA adapter 权重、`adapter_config.json`
和本地真实支持文件。`scripts/stage_remote_artifacts.sh` 也默认使用 `rsync -aL`
在 staging 阶段 dereference 支持文件 symlink。两者共同保证真实两节点配置可
移植，不改变请求 trace、adapter subset 或实验统计口径。

如果远端开启 token：

```bash
export PRIME_REMOTE_TOKEN='<runtime-token-not-committed>'
```

## 本地 Opt-in 运行

只有显式设置下面两个变量时，runner 才会走真实远端传输：

```bash
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
```

然后按原实验入口运行即可。不开这个开关时，旧实验链路完全不变。

## Fairness Interpretation

本轮 remote-fair/true-remote 复查的目标是确认本地 250MiB/s remote-sim 与真实
两节点 remote artifact node 的趋势是否一致，同时修正“只有 PrimeLoRA 走 remote”
的不公平风险。当前 baseline harness 的行为如下：

- SGLang：`dynamic_remote` 模式在 adapter 首次触达时从 remote endpoint
  materialize 到本地 cache，然后通过 SGLang 的动态 LoRA loading 进入其 LoRA
  manager；后续请求可复用本地 cache/运行时状态。
- vLLM：7B/13B 默认 `static_remote`，启动前从 remote stage selected subset，
  staging 时间计入 lifecycle；Llama-3.2 3B 使用 `dynamic_remote`，避免小模型
  500-adapter 静态注册造成不必要的 host footprint。
- ServerlessLLM：默认 `dynamic` request-path remote materialization；如果改成
  `stage`，staging 时间也必须计入 lifecycle。两种方式都不能直接绕过 remote
  读取本地 frozen pool。
- S-LoRA：先从 remote stage selected subset，再进入其 host/main-memory
  adapter pool；这是对 S-LoRA 论文语义的 harness 适配，而不是给它额外的
  PrimeLoRA-style routing。
- PrimeLoRA：remote miss 通过 HTTP artifact node materialize 到本地 NVMe，
  然后继续原有 NVMe/HOST/GPU readiness、routing、scale-out warmup 和 GPU
  admission 流程。

因此，公平性不是“所有系统每次请求都远程拉取”，而是“所有系统从同一 remote
源开始，且不能在 cold/startup/first-touch 阶段偷读本地工件”。这样既保留各
baseline 自身机制，也能观察 PrimeLoRA 是否通过分层 residency 和 readiness-aware
控制减少重复 remote fetch 及 first-token path 上的 readiness gap。

这一口径也和公开系统实践一致，而不是为了 PrimeLoRA 单独设计的规则。KServe
`LocalModelCache` 的做法是从远端模型源下载到节点本地 NVMe/cache 后再服务；
vLLM 的 LoRA serving 支持通过 `LoRARequest`/resolver 使用 adapter 路径，并
可由 resolver 从远端源 materialize 到本地路径；SGLang 的 LoRA serving 暴露
`max_loras_per_batch`、`max_loaded_loras` 与动态加载语义；S-LoRA 原论文明确
把 adapters 放在主机内存，并把当前 running queries 使用的 adapter fetch 到
GPU；ServerlessLLM 的核心也是利用 GPU 服务器附近的多层本地存储减少远端
checkpoint download。因此，我们的 fair harness 要求所有系统的 cold/startup/
first-touch materialization 从 remote source 发生，但不剥夺它们原生的本地
cache、host-memory pool 或 runtime residency。对应公开资料包括：
`https://kserve.github.io/website/docs/model-serving/generative-inference/modelcache/localmodel`、
`https://docs.vllm.ai/en/latest/features/lora/`、
`https://sgl-project.github.io/advanced_features/lora.html`、
`https://arxiv.org/abs/2311.03285` 与
`https://github.com/ServerlessLLM/ServerlessLLM`。

## 代码入口

- `remote_artifact_node/server.py`：remote 节点 HTTP artifact server。
- `scripts/remote_artifact_client.py`：本地 smoke client。
- `scripts/stage_remote_artifacts.sh`：通过 SSH/rsync 将本地 adapter 目录分发
  到 remote 节点；不接收、不保存密码。
- `faaslora/storage/http_artifact_store.py`：本地 HTTP fetch/extract 实现。
- `faaslora/storage/remote_client.py`：新增 `storage.remote.provider: http` 的
  通用 storage client 支持。
- `scripts/run_all_experiments.py`：新增环境变量开关。默认关闭，不影响正式链路。
- `configs/remote_artifact_example.yaml`：示例配置，不被正式配置自动引用。

PrimeLoRA-SGLang backend portability 运行时另有一条启动约束：launch spec 必须
显式包含与 HTTP port 不同的 `nccl-port`。SGLang 默认随机选择内部 NCCL 端口，
而 HTTP port 会更晚绑定；在多轮实验中随机内部端口可能撞上 HTTP port，造成
`address already in use`。当前 `scripts/run_all_experiments.py` 会为
PrimeLoRA-SGLang 单独预留并写入 `nccl-port`，该行为不改变 remote artifact
传输逻辑。

PrimeLoRA-SGLang 还必须让 LoRA batch 容量与运行并发上限一致。SGLang 官方
`max-loras-per-batch` 表示 running batch 内可同时出现的 adapter 数；在我们的
100% LoRA-bound replay 中，它必须至少覆盖 `max-running-requests`。当前
`scripts/run_all_experiments.py` 会默认写入：

```text
max-running-requests = model.max_num_seqs
max-loras-per-batch = max(model.max_loras, max-running-requests)
```

这样做只修复 SGLang many-LoRA 参数一致性，不改变 remote/NVMe/HOST/GPU 工件
链路，也不改变 PrimeLoRA 的 routing、scale-out preparation、residency 或
GPU admission 逻辑。

## 2026-05-14 True-Remote Formal Validation

真实 remote 复查已完成，不覆盖 local-sim 闭环结果。新表图和快照：

- `figs_remote/`
- `figs/paper/main_remote_fair_real_remote_v1_7b3b/`
- `figs/paper/backend_portability_real_remote_v1_7b3b/`
- `paper_results/final_remote_fair_real_remote_v1/`

baseline true-remote round：

- Llama-2 7B：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- Llama-2 13B：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_074336_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- Llama-3.2 3B：
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`

结论：

- 7B 与 3B true-remote 主表中，PrimeLoRA-vLLM 保持 CE 第一。
- PrimeLoRA-SGLang 在 7B 与 3B matched-backend true-remote 对比中也保持 CE
  高于 standalone SGLang。
- 13B true-remote 数据有效但只作为诊断保留，不合并进主表。
- local-sim 与 true-remote 的差异主要来自跨节点 staging、first-touch 和
  readiness tail；本地 250MiB/s remote-sim 仍是可用近似，但 true-remote
  snapshot 可作为会议提交时的真实性增强证据。
