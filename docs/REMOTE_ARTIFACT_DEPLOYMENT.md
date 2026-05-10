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

## 远端机器

当前计划的 remote artifact node：

```text
host: 10.199.227.174
ssh port: 8122
ssh user: lab14
artifact service port: 18080
```

密码不记录在本文档中。部署时使用交互式 SSH/SCP 输入。

## 远端目录

建议在 remote 节点上使用如下目录：

```bash
/data/primelora_remote_artifacts/llama2_7b_a500_v2_publicmix
/data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope
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
  --root /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 0.0.0.0 \
  --port 18080
```

如果需要 token：

```bash
export PRIME_REMOTE_TOKEN='<runtime-token-not-committed>'
python3 remote_artifact_node/server.py \
  --root /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 0.0.0.0 \
  --port 18080
```

systemd 样例位于：

```text
remote_artifact_node/systemd/primelora-remote-artifacts.service.example
```

## 本地 Smoke Test

在 GPU 推理节点：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://10.199.227.174:18080
python3 scripts/remote_artifact_client.py health
python3 scripts/remote_artifact_client.py list --limit 5
python3 scripts/remote_artifact_client.py smoke --dst-root /tmp/primelora_remote_fetch
```

本地环境如果设置了 `ALL_PROXY`，client 默认会绕过环境代理，直接连接 LAN
remote 节点，避免把 `10.199.*` 流量误发到外部代理。

如果远端开启 token：

```bash
export PRIME_REMOTE_TOKEN='<runtime-token-not-committed>'
```

## 本地 Opt-in 运行

只有显式设置下面两个变量时，runner 才会走真实远端传输：

```bash
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://10.199.227.174:18080
```

然后按原实验入口运行即可。不开这个开关时，旧实验链路完全不变。

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
